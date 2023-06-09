import json
import os
import tempfile
from glob import glob
from typing import Union

import ffmpeg
from bson.objectid import ObjectId

from efemarai.console import console
from efemarai.fields.annotation_fields import InstanceField, Polygon
from efemarai.fields.base_fields import BaseField
from efemarai.fields.data_fields import Video, VideoFrame
from efemarai.spec import convert


class ModelOutput:
    @staticmethod
    def create_from(spec, datapoint, data, remap_class_ids=None):
        outputs = convert(spec, data)

        if not isinstance(outputs, list):
            outputs = [outputs]

        for output in outputs:
            if remap_class_ids and hasattr(output, "label"):
                output.label.id = remap_class_ids[output.label.id]

            if output.ref_field is not None:
                continue

            output.ref_field = [input.id for input in datapoint.inputs]

        return ModelOutput(outputs)

    def __init__(self, outputs):
        self.outputs = outputs

    def __len__(self):
        return len(self.outputs)


class Datapoint:
    @staticmethod
    def create_from(spec, data):
        datapoint = Datapoint(dataset=None)
        inputs, targets = convert(spec, data)

        if not isinstance(inputs, list):
            inputs = [inputs]

        if not isinstance(targets, list):
            targets = [targets]

        for target in targets:
            if target.ref_field is not None:
                continue

            target.ref_field = [input.id for input in inputs]

        datapoint.inputs.extend(inputs)
        datapoint.targets.extend(targets)

        return datapoint

    @staticmethod
    def create_targets_from(spec, data):
        targets = convert(spec, data)
        if not isinstance(targets, list):
            targets = [targets]
        return targets

    """
    Used to represent the information about a single piece of data, which is
    to be passed to the model. The datapoint holds the data in `inputs` (like
    images, text, etc) and the annotations (bounding boxes, polygons, etc).

    Args:
        dataset (efemarai.dataset.Dataset): Instance of a Dataset object
        inputs (list[BaseFields]): List of Basefields to represent the inputs to the datapoint.
        targets (list[BaseFields]): List of Basefields to represent the targets to the inputs datapoint.
        id (Optional[ObjectId]): A valid ObjectId for the datapoint
    Returns:
        :class: `efemarai.fields.datapoint.Datapoint`
    """

    def __init__(self, dataset, inputs=None, targets=None, id=None) -> None:
        self.inputs = self.sanitize_format(inputs)
        self.targets = self.sanitize_format(targets)
        self.dataset = dataset
        self.id = id if id is not None else ObjectId()

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  dataset={self.dataset}"
        res += f"\n  inputs={self.inputs}"
        res += f"\n  targets={self.targets}"
        res += "\n)"
        return res

    def __getattr__(self, name):
        def find_in(fields):
            res = [field for field in fields if field.key_name == name]
            return res[0] if res else None

        return find_in(self.inputs) or find_in(self.targets)

    @staticmethod
    def sanitize_format(data):
        if data is None:
            return []

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for k, v in data.items():
                v.key_name = k
            return list(data.values())
        raise AttributeError("Datapoint inputs and targets should be list or dict.")

    def add_target(self, _target: Union[BaseField, InstanceField]):
        """
        Used to add target to the datapoint.

        Args:
            _target: An instance of efemarai.fields.base_fields.BaseField.

        Returns:
            targets: list[BaseField]

        """
        if isinstance(_target, Polygon) and (
            not isinstance(_target.vertices, (list, tuple))
            or not len(_target.vertices) > 0
            or not isinstance(_target.vertices[0][0][0], (float, int))
        ):
            console.print(
                ":poop: We do not support empty polygons or RLE style polygons yet. Please pass a list of lists of floats.",
                style="red",
            )
            return None

        self.targets.append(_target)
        return self.targets

    def add_inputs(self, _input: Union[BaseField, InstanceField]):
        """
        Used to add inputs to the datapoint.

        Args:
            _input: An instance of efemarai.fields.base_fields.BaseField.

        Returns:
            inputs: list[BaseField]

        """
        self.inputs.append(_input)
        return self.inputs

    def _post(self, endpoint, json_obj=None, params=None):
        return self.dataset.project._session._post(endpoint, json_obj, params)

    def _put(self, endpoint, json_obj=None, params=None):
        return self.dataset.project._put(endpoint, json_obj, params)

    def _handle_video(self, ef_video):
        with tempfile.TemporaryDirectory() as TMP_VIDEO_FRAME_DIR:
            video_info = ffmpeg.probe(ef_video.file_path)["streams"]
            width = [info["width"] for info in video_info if info.get("width")][0]
            height = [info["height"] for info in video_info if info.get("height")][0]

            basename = os.path.basename(ef_video.file_path)
            out_path = os.path.join(TMP_VIDEO_FRAME_DIR, basename)
            os.makedirs(out_path, exist_ok=True)
            ffmpeg.input(ef_video.file_path).output(
                f"{out_path}/{basename}_frame_%08d.png", **{"qscale:v": 2}
            ).run(quiet=True)

            frames = sorted(glob(f"{out_path}/{basename}_frame_*.png"))
            # ffmpeg r_frame_rate is a string in decimal format -> "12/1"
            r_frame_rate = [
                info["r_frame_rate"] for info in video_info if info.get("width")
            ][0].split("/")
            fps = int(r_frame_rate[0]) / int(r_frame_rate[1])
            ef_video.width = width
            ef_video.height = height
            ef_video.fps = fps
            ef_video.frame_count = len(frames)
            for frame in frames:
                ef_video.frames.append(
                    VideoFrame(
                        index=frame.split("_")[-1].strip(".png"),
                        file_path=frame,
                        width=width,
                        height=height,
                    )
                )
                self.dataset.project._upload(
                    frame,
                    f"api/datapoint/{self.dataset.id}/upload",
                    verbose=False,
                )
                os.remove(frame)

    def upload(self, copy_files=True):
        """
        Used to upload a datapoint to the system.

        Args:
            copy_files (bool): Whether to copy the files or to upload them

        Returns:
            datapoint_id: ObjectId of the uploaded datapoint
        """
        files_uploaded = []
        # Upload files associated with inputs and targets
        for field in self.inputs + self.targets:
            if hasattr(field, "file_path") and isinstance(field.file_path, str):
                if not os.path.isfile(field.file_path):
                    console.print(
                        f":poop: Expecting a file at '{field.file_path}', but cannot locate it.",
                        style="red",
                    )
                    continue

                # Uploaded this file, so skipping
                if field.file_path in files_uploaded:
                    continue

                self.dataset.project._upload(
                    field.file_path,
                    f"api/datapoint/{self.dataset.id}/upload",
                    verbose=False,
                )

                files_uploaded.append(field.file_path)

                if isinstance(field, Video):
                    self._handle_video(field)

        serialized_targets = [
            json.loads(target._serialize()) for target in self.targets
        ]
        serialized_inputs = [json.loads(_input._serialize()) for _input in self.inputs]

        response = self._put(
            "api/datapoint/undefined",
            json_obj={
                "access_token": self.dataset.project._session.token,
                "datasetId": self.dataset.id,
                "targets": serialized_targets,
                "inputs": serialized_inputs,
            },
        )

        datapoint_id = response["id"]
        if not copy_files:
            return datapoint_id

        return datapoint_id

    def get_input(self, name):
        """
        Returns an input with that particular key name.

        Args:
            name (str): Key name of a specific field to be returned, or None if not found.

        Returns:
            An instance of BaseField
        """
        for _input in self.inputs:
            if _input.key_name == name:
                return _input

        return None

    def get_inputs_with_type(self, ef_type):
        """
        A way to filter only inputs of a particular ef_type.

        Args:
            ef_type (BaseField): An instance of `efemarai.fields.base_fields.BaseField`.

        Returns:
            list[BaseField] - a list of inputs of the ef_type

        Example:

        .. code-block:: python

            import efemarai as ef

            images = datapoints[0].get_inputs_with_type(ef.Image)
            # images is a list of all of the ef.Image elements in datapoints[0].
            # each of those images can be accessed by datapoints[0].get_input(images[0].key_name)
        """
        res = []
        for _input in self.inputs:
            if isinstance(ef_type, Video):
                res.append(_input)
                continue
            if isinstance(_input, ef_type):
                res.append(_input)
            if _input._cls == "Video":
                # TODO: Use frame.index when indexes start from 0. Currently they start at 1.
                for idx, frame in enumerate(_input.frames):
                    frame.set_data(_input.data[idx])
                    res.append(frame)
        return res

    def get_targets_with_type(self, ef_type):
        """
        A way to filter only targets of a particular ef_type.

        Args:
            ef_type (BaseField): An instance of `efemarai.fields.base_fields.BaseField`.

        Returns:
            list[BaseField] - a list of inputs of the ef_type
        """
        res = []
        for target in self.targets:
            if isinstance(target, ef_type):
                res.append(target)
        return res

    def get_input_refs(self, ids):
        """
        Returns a list of field in input that refer to id.

        Args:
            ids (List[ObjectId]): Ids when referenced to return object, or [] if not found.

        Returns:
            An instance of BaseField
        """
        refs = []
        for _input in self.inputs:
            if _input.id in ids:
                refs.append(_input)

        return refs
