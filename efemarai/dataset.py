import json
import os
import zipfile
from enum import Enum
from time import sleep

import boto3
from appdirs import user_data_dir
from botocore.errorfactory import ClientError

from efemarai.console import console
from efemarai.fields import AnnotationClass, sdk_deserialize
from efemarai.job_state import JobState
from efemarai.problem_type import ProblemType


class DatasetFormat(str, Enum):
    """Possible dataset formats."""

    COCO = "COCO"
    ImageRegression = "ImageRegression"
    ImageNet = "ImageNet"
    TFRecord = "tfrecord"
    Custom = "Custom"

    @classmethod
    def has(cls, value):
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def list(cls):
        return list(map(lambda x: x.value, cls._member_map_.values()))


class DatasetStage(str, Enum):
    Train = "train"
    Validation = "validation"
    Test = "test"

    @classmethod
    def has(cls, value):
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def list(cls):
        return list(map(lambda x: x.value, cls._member_map_.values()))


class Dataset:
    """
    Provides dataset related functionality.
    It can be created through the :class:`efemarai.project.Project.create_dataset` method.

    Example:

    .. code-block:: python
        :emphasize-lines: 2,5

        import efemarai as ef
        dataset = ef.Session().project("Name").create_dataset(...)
        # do something else
        dataset.reload()
        if dataset.finished:
            print('Dataset has finished loading')

    Example (2):

    .. code-block:: python

        import efemarai as ef
        project = ef.Session().project("Name")
        dataset = project.dataset("Dataset Name")
        print(f"Classes: {dataset.classes}")

    """

    @staticmethod
    def create(
        project,
        name,
        stage,
        format=None,
        data_url=None,
        annotations_url=None,
        credentials=None,
        upload=None,
        num_datapoints=None,
        mask_generation=None,
        min_asset_area=None,
    ):
        """Create a dataset. A more convenient way is to use `project.create_dataset(...)`."""
        if stage is not None and data_url is not None:
            # Remote dataset
            if not DatasetFormat.has(format):
                raise AssertionError(
                    f"Dataset format '{format}' should be in {DatasetFormat.list()}."
                )

            if not DatasetStage.has(stage):
                raise AssertionError(
                    f"Dataset stage '{stage}' should be in {DatasetStage.list()}."
                )

            if name is None or data_url is None:
                return None

            if mask_generation not in (None, "Simple", "Advanced"):
                raise AssertionError("Mask generation not in (None, Simple, Advanced)")

            response = project._put(
                "api/dataset",
                json={
                    "name": name,
                    "format": format,
                    "stage": stage,
                    "data_url": data_url,
                    "annotations_url": annotations_url,
                    "access_token": credentials,
                    "upload": upload,
                    "projectId": project.id,
                },
            )
            dataset_id = response["id"]

            if upload:
                endpoint = f"api/dataset/{dataset_id}/upload"
                if annotations_url is not None:
                    project._upload(annotations_url, endpoint)
                project._upload(data_url, endpoint)
                project._post(
                    endpoint,
                    json={
                        "num_samples": num_datapoints,
                        "mask_generation": mask_generation,
                        "min_asset_area": min_asset_area,
                    },
                )

            return Dataset(
                project=project,
                name=name,
                stage=stage,
                id=dataset_id,
                format=format,
                data_url=data_url,
                annotations_url=annotations_url,
                state="NotStarted",
                classes=[],
            )
        response = project._put(
            "api/dataset",
            json={
                "name": name,
                "format": format,
                "stage": stage,
                # "data_url": data_url,
                # "annotations_url": annotations_url,
                "access_token": credentials,
                "upload": upload,
                "projectId": project.id,
                "create_only": True,
            },
        )
        dataset_id = response["id"]

        if upload:
            endpoint = f"api/dataset/{dataset_id}/upload"
            if annotations_url is not None:
                project._upload(annotations_url, endpoint)
            project._upload(data_url, endpoint)
            project._post(
                endpoint,
                json={
                    "num_samples": num_datapoints,
                    "mask_generation": mask_generation,
                    "min_asset_area": min_asset_area,
                },
            )

        # Custom dataset to be uploaded
        return Dataset(
            project=project,
            name=name,
            stage=stage,
            id=dataset_id,
            state="NotStarted",
        )

    @staticmethod
    def _parse_classes(dataset_classes):
        if dataset_classes == []:
            return []

        # Support previous versions
        id_access_key = "id" if "id" in dataset_classes[0] else "index"

        classes = []
        for c in dataset_classes:
            classes.append(
                AnnotationClass(
                    id=c[id_access_key],
                    name=c["name"],
                    category=c.get("category"),
                    color=c.get("color"),
                )
            )

        return classes

    def __init__(
        self,
        project,
        name,
        stage,
        id=None,
        format=None,
        data_url=None,
        annotations_url=None,
        state=None,
        state_message=None,
        classes=None,
        count=None,
    ):
        self.project = (
            project  #: (:class:`efemarai.project.Project`) Associated project.
        )
        self.stage = stage  #: (str) Stage of the dataset.
        self.name = name  #: (str) Name of the dataset.
        self.classes = (
            classes if classes is not None else []
        )  #: (list [str]) List of names of the classes. `None` is used for those ids that don't have a corresponding name.
        self.id = id
        self.format = format  #: (str) Format of the dataset.
        self.data_url = data_url
        self.annotations_url = annotations_url
        self.state = JobState(state)
        self.state_message = state_message
        self.count = count

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  name={self.name}"
        res += f"\n  format={self.format}"
        res += f"\n  stage={self.stage}"
        res += f"\n  count={self.count}" if self.count else ""
        res += f"\n  data_url={self.data_url}" if self.data_url else ""
        res += (
            f"\n  annotations_url={self.annotations_url}"
            if self.annotations_url
            else ""
        )
        res += f"\n  state={self.state}"
        res += f"\n  state_message={self.state_message}"
        res += f"\n  classes={self.classes}"
        res += "\n)"
        return res

    @staticmethod
    def deserialize_datapoint(data):
        """Returns a datapoint from the data field."""
        datapoint = sdk_deserialize(data)
        datapoint.inputs = [sdk_deserialize(json.loads(i)) for i in datapoint.inputs]
        # Dereference target ref_fields
        for i, target in enumerate(datapoint.targets):
            data = json.loads(target)
            if "ref_field" in data and len(data["ref_field"]) > 0:
                data["ref_field"] = [
                    di
                    for ref in data["ref_field"]
                    for di in datapoint.inputs
                    if str(di.id) in str(ref)
                ]
            datapoint.targets[i] = data

        datapoint.targets = [sdk_deserialize(t) for t in datapoint.targets]
        return datapoint

    def __get_data(self, skip, limit):
        response = self.project._post(
            "api/datapointsIter",
            json={
                "datasetId": self.id,
                "skip": skip,
                "limit": limit,
                "showOnlyAllowed": True,
            },
        )

        datapoints = []
        for data in response["objects"]:
            datapoint = Dataset.deserialize_datapoint(data)
            datapoints.append(datapoint)

        return datapoints, response["count"]

    def __len__(self):
        if self.count is None:
            _, datapoints_count = self.__get_data(skip=0, limit=1)
            self.count = datapoints_count

        return self.count

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Note: Currently ignoring the step in the slice
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            datapoints, _ = self.__get_data(skip=start, limit=stop - start)
            return datapoints
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(
                    f"The index ({key}) is out of range (size: {len(self)})."
                )
            datapoints, _ = self.__get_data(skip=key, limit=1)
            return datapoints[0]
        raise TypeError(f"Invalid argument type ({type(key)}).")

    @property
    def finished(self):
        """Returns true if the dataset loading has finished.

        :rtype: bool
        """
        return self.state == JobState.Finished

    @property
    def failed(self):
        """Returns true if the dataset loading has failed.

        :rtype: bool
        """
        return self.state == JobState.Failed

    @property
    def running(self):
        """Returns true if the is still being loaded - not failed or finished.

        :rtype: bool
        """
        return self.state not in (JobState.Finished, JobState.Failed)

    def add_annotation_class(self, id, name, category=None, color=None):
        annotation = AnnotationClass(id=id, name=name, category=category, color=color)
        self.classes.append(annotation)
        return annotation

    def get_annotation_class(self, *, id=None, name=None):
        if name is not None:
            annotations = list(filter(lambda x: x.name == name, self.classes))

        if id is not None:
            annotations = list(filter(lambda x: x.id == id, self.classes))

        if len(annotations) != 1:
            console.print(
                f":poop: Received not 1 annotation with name:{name}, id: {id} - '{annotations}' len({len(annotations)})."
                f"Class data: {self.classes}",
                style="red",
            )
            return None

        return annotations[0]

    def finalize(self, min_asset_area=15, mask_generation=None):
        self.project._put(
            f"api/dataset/{self.id}/finalize",
            json={
                "classes": [_cls._serialize() for _cls in self.classes],
                "asset": {
                    "min_asset_area": min_asset_area,
                    "mask_generation": mask_generation,
                },
            },
        )
        self.reload()

        return self

    def delete(self, delete_dependants=False):
        """
        Deletes the dataset.

        You cannot delete an object that is used in a stress test
        or a baseline (delete those first). This cannot be undone.
        """
        if self.id is None:
            return

        self.project._delete(
            f"api/dataset/{self.id}/{delete_dependants}",
        )

    def reload(self):
        """
        Reloads the dataset *in place* from the remote endpoint and return it.

        Returns:
            The updated dataset object.
        """
        if self.id is None:
            return None

        endpoint = f"api/dataset/{self.id}"
        dataset_details = self.project._get(endpoint)

        self.name = dataset_details["name"]
        self.format = dataset_details["format"]
        self.stage = dataset_details["stage"]
        self.data_url = dataset_details.get("data_url")
        self.annotations_url = dataset_details.get("annotations_url")
        self.state = JobState(dataset_details["states"][-1]["name"])
        self.state_message = dataset_details["states"][-1].get("message")
        self.classes = self._parse_classes(dataset_details["classes"])

        return self

    def download(
        self,
        num_samples=None,
        dataset_format=None,
        path=None,
        unzip=True,
        ignore_cache=False,
    ):
        """
        Download the dataset locally.

        Args:
            num_samples (int): Number of samples to download. Leave `None` for all.
            dataset_format (str): What format to download the dataset. Currently supported include `COCO`, `YOLO`, `VOC`, `ImageNet`, `CSV`.
            path (str): The path where to download the data.
            unzip (bool): Should the downloaded zip be unzipped.
            ignore_cache (bool): Force regeneration of the dataset by ignoring the cache. May lead to slower subsequent calls.
        """
        if self.id is None:
            return

        if self.running:
            console.print(":poop: Dataset has not finished loading.", style="red")
            return None

        if path is None:
            path = user_data_dir(appname="efemarai")

        path = os.path.join(path, self.id)

        if dataset_format is None:
            if self.project.problem_type == ProblemType.Classification:
                dataset_format = "imagenet"
            elif self.project.problem_type == ProblemType.ObjectDetection:
                dataset_format = "coco"
            elif self.project.problem_type == ProblemType.InstanceSegmentation:
                dataset_format = "coco"
            elif self.project.problem_type == ProblemType.Keypoints:
                dataset_format = "coco"

        if dataset_format is None:
            console.print(":poop: Unsupported problem type.", style="red")
            return None

        if not ignore_cache:
            name = os.path.join(path, f"dataset_{dataset_format}")

            if num_samples:
                name += f"_{num_samples}"

            if os.path.exists(name):
                return name

            name += ".zip"
            if os.path.exists(name + ".zip"):
                return name

        access = self.project._post(
            "api/downloadDataset",
            json={
                "id": self.id,
                "format": dataset_format,
                "num_samples": num_samples,
                "async_download": True,
            },
        )

        s3 = boto3.client(
            "s3",
            aws_access_key_id=access["AccessKeyId"],
            aws_secret_access_key=access["SecretAccessKey"],
            aws_session_token=access["SessionToken"],
            endpoint_url=access["Url"],
        )

        with console.status(f"Preparing '{self.name}' dataset download"):
            while True:
                try:
                    response = s3.head_object(
                        Bucket=access["Bucket"], Key=access["ObjectKey"]
                    )
                    size = response["ContentLength"]
                    break
                except ClientError:
                    sleep(1)

        with self.project._session._progress_bar(verbose=True) as progress:
            task = progress.add_task("Downloading dataset ", total=float(size))

            def callback(num_bytes):
                return progress.advance(task, num_bytes)

            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, os.path.basename(access["ObjectKey"]))

            s3.download_file(
                access["Bucket"], access["ObjectKey"], filename, Callback=callback
            )

        if unzip:
            with console.status("Unzipping dataset"):
                dirname = os.path.splitext(filename)[0]
                with zipfile.ZipFile(filename, "r") as f:
                    f.extractall(dirname)

                os.remove(filename)

                filename = dirname

        console.print(
            (f":heavy_check_mark: Downloaded '{self.name}' dataset to\n  {filename}"),
            style="green",
        )

        return filename

    def enhance(self, domain, samples_per_datapoint, new_name=None):
        response = self.project._post(
            "api/datasetEnhance",
            json={
                "datasetId": self.id,
                "domainId": domain.id,
                "samplesPerDatapoint": samples_per_datapoint,
                "newName": new_name,
            },
        )
        new_dataset_id = response["value"]
        return self.project.dataset(new_dataset_id)
