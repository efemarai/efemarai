import math
from typing import Dict

import cv2
import numpy as np
from bson.objectid import ObjectId
from PIL import Image as pil_image

from efemarai.fields.annotation_fields import AnnotationClass, InstanceField, Polygon
from efemarai.fields.base_fields import (
    BaseField,
    create_polygons_from_mask,
    sdk_serialize,
)


class Image(BaseField):
    """
    Represents an image that can be attached to any other field and can
    act as an input or a target. Either the `file_path` or the `data` field
    needs to be passed to be a valid element.

    Args:
        file_path (str): The location of the image to be loaded.
        data (np.ndarray): The numpy array with the loaded image of type `uint8`.
        width (Optional[int]): Width of the image (autofield if data is passed).
        height (Optional[int]): Height of the image (autofield if data is passed).
        description (Optional[str]): A description of the object or annotation information.
        ref_field (Optional[List[object]]): A list of objects the current BaseField is refering.
        key_name (Optional[str]): A string key, by which the field can be identified.
        confidence (Optional[float]): Confidence level of the field.
        user_attributes (Optional[Dict[str, object]]): Optional dictionary containing additional
            data regarding the field.

    Returns:
        :class: `efemarai.fields.data_fields.Image`: An Image object.

    Example:

    .. code-block:: python

        import efemarai as ef
        import numpy as np

        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:30, 10:50] = 255

        ef_image = ef.Image(data=image)
        print(ef_image)

        ef_image2 = ef.Image(file_path="./image.png")
        print(ef_image2)
    """

    def __init__(
        self,
        file_path: str = None,
        data: np.ndarray = None,
        width: int = None,
        height: int = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        BaseField.__init__(
            self,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        if file_path is None and data is None:
            raise AssertionError(
                f"You need to specify either a 'file_path'({file_path}) or 'data'({data})."
            )

        self.file_path = file_path
        self.width = width
        self.height = height
        self._raw = data

        # Skip check for 0-1 images (TODO: give warning)
        if isinstance(self._raw, list):
            self._raw = np.asarray(self._raw, dtype=np.uint8)
        if self._raw is not None and self._raw.dtype != np.uint8:
            raise AttributeError(
                f"Expected format {np.uint8}. Received {self._raw.dtype}."
            )

        if height is None:
            if self._raw is not None:
                self.height = self._raw.shape[0]
            else:
                img = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
                self.height = img.shape[0]
                self.width = img.shape[1]

        if width is None and self._raw is not None:
            self.width = self._raw.shape[1]

    @property
    def data(self):
        if self._raw is None:
            self._raw = np.asarray(pil_image.open(self.file_path))
            self.height = self._raw.shape[0]
            self.width = self._raw.shape[1]
        return self._raw

    def set_data(self, data: np.ndarray):
        if self._raw.dtype != np.uint8:
            raise AssertionError(f"Expected format {np.uint8}. Received {data.dtype}.")

        self._raw = data
        self.height = self._raw.shape[0]
        self.width = self._raw.shape[1]

    @data.setter
    def data(self, data):
        self.set_data(data)

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  file_path={self.file_path}"
        res += "\n  data=" + (
            f"{self._raw.shape}, {self._raw.dtype}, min({np.min(self._raw)}), max({np.max(self._raw)})"
            if self._raw is not None
            else "<not-loaded>"
        )
        res += f"\n  width={self.width}"
        res += f"\n  height={self.height}"
        res += f"\n  description={self.description}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += "\n)"
        return res


class Mask(Image):
    """Used for Semantic Segmentation. Same as Image class, but whenever rescaled, Image.Nearest strategy is used."""

    pass


class InstanceMask(Image, InstanceField):
    """
    Represents an instance mask that can be attached to any other field and can
    act as an input or a target.

    Args:
        file_path (str): The location of the image to be loaded.
        data (np.ndarray): The numpy array with the loaded image.
        width (Optional[int]): Width of the image.
        height (Optional[int]): Height of the image.
        instance_id (id): Id of the instance within the dataset
        label:( Optional[AnnotationClass]): An AnnotationClass associated with the instance
        description (str): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (str): A string key, by which the field can be identified.
        confidence (float): Confidence level of the field.
        user_attributes (Dict[str, object]): Optional dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.data_fields.InstanceMask`: An InstanceMask object.
    """

    def __init__(
        self,
        instance_id: int = None,
        label: AnnotationClass = None,
        file_path: str = None,
        data: np.ndarray = None,
        width: int = None,
        height: int = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        """The MRO is >>> ef.InstanceMask.__mro__
        (<class 'efemarai.fields.data_fields.InstanceMask'>,
         <class 'efemarai.fields.data_fields.Image'>,
         <class 'efemarai.fields.annotation_fields.InstanceField'>,
         <class 'efemarai.fields.base_fields.BaseField'>,
         <class 'object'>)
        """
        Image.__init__(
            self,
            file_path=file_path,
            data=data,
            width=width,
            height=height,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )
        InstanceField.__init__(
            self,
            instance_id=instance_id,
            label=label,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  file_path={self.file_path}"
        res += "\n  data=" + (
            f"{self._raw.shape}, {self._raw.dtype}, min({np.min(self._raw)}), max({np.max(self._raw)})"
            if self._raw is not None
            else "<not-loaded>"
        )
        res += f"\n  confidence={self.confidence}"
        res += f"\n  label={self.label}"
        res += f"\n  width={self.width}"
        res += f"\n  height={self.height}"
        res += f"\n  description={self.description}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += "\n)"
        return res

    def to_polygon(self):
        """
        Converts `efemarai.fields.data_fields.InstanceMask` to `efemarai.fields.annotation_fields.Polygon`

        Returns:
            :class: `efemarai.fields.annotation_fields.Polygon`

        Example: See ef.Polygon for an example usage.
        """
        vertices, area = create_polygons_from_mask(self.data)
        return Polygon(
            vertices=vertices,
            area=area,
            instance_id=self.instance_id,
            label=self.label,
            confidence=self.confidence,
            id=self.id,
            description=self.description,
            ref_field=self.ref_field,
            key_name=self.key_name,
        )

    @staticmethod
    def bool_to_uint8(data):
        if data.dtype != np.uint8:
            data = data.astype(np.uint8) * 255
        return data


class VideoFrame(Image):
    def __init__(
        self,
        index: int,
        file_path: str,
        width: int = None,
        height: int = None,
        data: np.ndarray = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        Image.__init__(
            self,
            file_path=file_path,
            data=data,
            width=width,
            height=height,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.index = index
        self.width = width
        self.height = height
        self._raw = data


class Video(BaseField):
    def __init__(
        self,
        file_path: str,
        data: np.ndarray = None,
        frames: list = None,
        frame_count: int = None,
        fps: int = None,
        width: int = None,
        height: int = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        if frames is None:
            frames = []
        BaseField.__init__(
            self,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.file_path = file_path
        self.frames = frames
        self.frame_count = frame_count
        self.fps = fps
        self.width = width
        self.height = height
        self._raw = data
        self._index = None

    def __iter__(self):
        for frame in self.frames:
            yield frame

    def __len__(self):
        return len(self)

    def __getitem__(self, i):
        if isinstance(i, slice):
            self._index = list(range(i.start, i.stop, i.step if i.step else 1))
        elif isinstance(i, (int, list, tuple)):
            self._index = i
        else:
            raise AttributeError(
                f"List indices must be int, list or slice. Got {type(i)}"
            )
        return self

    @property
    def data(self):
        if self._raw is None:
            self._raw = np.stack([frame.data for frame in self.frames])
        return self._raw

    def set_data(self, data):
        self._raw = data


class Tensor(BaseField):
    def __init__(
        self,
        data: np.ndarray,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.data = data


class Text(BaseField):
    def __init__(
        self,
        text: str,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.text = text


class DataFrame(BaseField):
    def __init__(
        self,
        data,
        target_column: str,
        file_path: str = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.data = data
        self.file_path = file_path
        self.target_column = target_column

    def split_inputs_targets(self, n_rows=1):
        iterations = math.ceil(self.data.shape[0] / n_rows)
        inputs, targets = [], []
        target_column = self._check_column(self.data, self.target_column)
        for chunk in range(iterations):
            rows = self.data.loc[chunk * n_rows : ((chunk + 1) * n_rows - 1)]

            # exclude target column
            inputs.append(rows.loc[:, rows.columns != target_column])
            targets.append(rows[target_column].tolist())

        return inputs, targets

    def _serialize(self):
        self.data = self.data.to_dict("list")
        return sdk_serialize(self)

    @staticmethod
    def _check_column(data, target_column):
        if target_column not in data.columns:
            raise AttributeError(
                f"Target column '{target_column}' not found in dataframe columns: {data.columns.tolist()}."
                "Please provide a valid column name."
            )
        return target_column


class JSON(BaseField):
    def __init__(
        self,
        data: dict,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.data = data


class Audio(BaseField):
    def __init__(
        self,
        data,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.data = data
