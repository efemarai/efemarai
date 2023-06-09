from typing import Dict, List, Tuple

import cv2
import numpy as np
import slugify
from bson.objectid import ObjectId

from efemarai.fields.base_fields import (
    BaseField,
    create_polygons_from_mask,
    sdk_serialize,
)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class AnnotationClass:
    """
    Represents an annotation that can be attached to any other field and can
    act as a label. When returned by a model output, either name or id can be
    specified. When creating a dataset, both are required.

    Args:
        name (str): Name of the class.
        id (int): Id of the class.
        confidence (Optional[float]): Confidence level. Prefer using the DataField confidence in general.
        color (Optional[str]): Color of the class in hex string. Used for visualization.
        text_color (Optional[str]): Color of the text of the annotation. Used for visualization.
        category (Optional[str]): Category of the class.
        slug (Optional[str]): Slugified name of the class that can be used for queries etc.

    Returns:
        :class: `efemarai.fields.annotation_fields.AnnotationClass`: An AnnotationClass object.

    Example:

    When your code needs to return a target that has a label, you should create one with
    the name _or_ id that your model returns.

    .. code-block:: python

        import efemarai as ef
        output = {"class_id": 1, "name": "person", "value": 0.77}
        label = ef.AnnotationClass(id=output["class_id"], name=output["name"])
        label2 = ef.AnnotationClass(id=output["class_id"])
        print(label, label2)

    If you are appending ground truth data to a dataset, you need to resolve all of the information.
    You can do that through the dataset object to obtain the annotation.

    .. code-block:: python

        label = dataset.get_annotation_class(id=output["class_id"])
        label = dataset.get_annotation_class(name=output["name"])
    """

    def __init__(
        self,
        name: str = None,
        id: int = None,
        confidence: float = None,
        color: str = None,
        text_color: str = None,
        category: str = None,
        slug: str = None,
    ):
        if name is None and id is None:
            raise AssertionError("'name' and 'id' cannot be None at the same time.")

        self.name = name
        self.id = int(id)
        self.confidence = confidence
        self.category = category
        self.color = color if color is not None else self.get_random_color()
        self.text_color = (
            text_color if text_color is not None else self.get_random_color()
        )
        self.slug = slug if slug is not None else self.get_slug()

    def get_slug(self):
        return slugify.slugify(self.name) if self.name is not None else None

    @staticmethod
    def get_random_color():
        import random

        def r():
            return random.randint(0, 255)

        return f"#{r():02X}{r():02X}{r():02X}"

    def _serialize(self):
        return sdk_serialize(self)

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  name={self.name}"
        res += f"\n  id={self.id}"
        res += f"\n  confidence={self.confidence}"
        res += f"\n  color={self.color}" if self.color else ""
        res += f"\n  text_color={self.text_color}" if self.text_color else ""
        res += f"\n  category={self.category}" if self.category else ""
        res += f"\n  slug={self.slug}"
        res += "\n)"
        return res


class Tag(BaseField):
    """
    Represents an annotation that can be used for classification.

    Args:
        label (AnnotationClass): An instance of AnnotationClass
        probabilities (np.ndarray): The softmax probabilities for a given prediction.
        confidence (Optional[float]): Confidence level of the field (autocalculated from probabilities).
        id (Optional[ObjectID]): Id of the field.
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Optional dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.Tag`: A Tag object.

    Example:

    .. code-block:: python

        import efemarai as ef
        import numpy as np

        outputs = [0.1, 0.2, 0.15, 0.55]

        label = ef.AnnotationClass(id=np.argmax(outputs))
        tag = ef.Tag(label=label, probabilities=outputs)
        print(label, tag)
    """

    def __init__(
        self,
        label: AnnotationClass,
        probabilities: np.ndarray = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        probs = np.asarray(probabilities)
        # Get confidence from passed arg, or from probabilities if they exist
        confidence = (
            confidence
            if confidence is not None
            else float(
                (-1 * probs * np.log2(probs)).sum(axis=probs.ndim - 1)
            )  # Could we use multidim data?
            if probs.any()
            else None
        )
        super().__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.probabilities = probabilities
        self.label = label

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  label={self.label}"
        res += f"\n  probabilities={self.probabilities}"
        res += f"\n  confidence={self.confidence}"
        res += f"\n  description={self.description}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += "\n)"
        return res


class Value(BaseField):
    """
    Represents an annotation that can be used for regression.

    Args:
        value (float): A float to represent the annotation value
        confidence (Optional[float]): An optional confidence associated with the value
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.Value`: A Value object.

    Example:

    .. code-block:: python

        import efemarai as ef
        import numpy as np

        output = 0.55

        value = ef.Value(value=output)
        print(value)
    """

    def __init__(
        self,
        value: float,
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

        self.value = value

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  value={self.value}"
        res += f"\n  description={self.description}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += "\n)"
        return res


class InstanceField(BaseField):
    """
    This is a base class. Represents an annotation of an instance that can be
    attached to any other field and can act as a label.

    Args:
        instance_id (id): Id of the instance within the dataset.
        label (AnnotationClass): An AnnotationClass associated with the instance.
        confidence (float): Confidence level of the field.
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.InstanceField`: An InstanceField object.
    """

    def __init__(
        self,
        instance_id: int,
        label: AnnotationClass = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super(InstanceField, self).__init__(
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.instance_id = instance_id
        self.label = (
            label
            if isinstance(label, (AnnotationClass, type(None)))
            else AnnotationClass(**label)
        )


class BoundingBox(InstanceField):
    XYWH_ABSOLUTE = 0
    XYXY_ABSOLUTE = 1
    CENTERWH_ABSOLUTE = 2

    @staticmethod
    def convert(box, source_format=None, target_format=None):
        if source_format is None:
            source_format = BoundingBox.XYXY_ABSOLUTE

        if target_format is None:
            target_format = BoundingBox.XYXY_ABSOLUTE

        if source_format == target_format:
            return box

        if source_format != BoundingBox.XYXY_ABSOLUTE:
            box = BoundingBox._convert_to_xyxy_absolute(box, source_format)

        if target_format != BoundingBox.XYXY_ABSOLUTE:
            box = BoundingBox._convert_from_xyxy_absolute(box, target_format)

        return box

    @staticmethod
    def _convert_to_xyxy_absolute(box, source_format):
        if source_format == BoundingBox.XYWH_ABSOLUTE:
            x, y, w, h, *rest = box
            return x, y, x + w - 1, y + h - 1, *rest
        if source_format == BoundingBox.CENTERWH_ABSOLUTE:
            x, y, w, h, *rest = box
            return x - w / 2, y - h / 2, x + w / 2 - 1, y + h / 2 - 1, *rest

    @staticmethod
    def _convert_from_xyxy_absolute(box, target_format):
        if target_format == BoundingBox.XYWH_ABSOLUTE:
            x1, y1, x2, y2, *rest = box
            return x1, y2, x2 - x1 + 1, y2 - y1 + 1, *rest

    """
    Represents a bounding box annotation.

    Args:
        xyxy (Tuple[float]): Coordinates of the bounding box in absolute values in format x1,y1,x2,y2.
        label (AnnotationClass): An AnnotationClass associated with the instance
        area (Optional[float]): Area of the bounding box
        instance_id (Optional[id]): Id of the instance within the dataset
        confidence (Optional[float]): Confidence level of the field.
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.BoundingBox`: A BoundingBox object.
    """

    def __init__(
        self,
        xyxy: tuple,
        label: AnnotationClass,
        area: float = None,
        instance_id: int = None,
        description: str = None,
        confidence: float = None,
        id: ObjectId = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            instance_id=instance_id,
            label=label,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        self.xyxy = tuple(
            i if not isinstance(i, (np.int64, np.int32)) else i.item() for i in xyxy
        )
        self.area = (
            area
            if area is not None
            else (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
        )

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  xyxy={self.xyxy}"
        res += f"\n  area={self.area}"
        res += f"\n  instance_id={self.instance_id}"
        res += f"\n  description={self.description}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  label={self.label}"
        res += f"\n  key_name={self.key_name}"
        res += "\n)"
        return res

    @property
    def width(self):
        return self.xyxy[2] - self.xyxy[0] + 1

    @property
    def height(self):
        return self.xyxy[3] - self.xyxy[1] + 1

    @property
    def size(self):
        return self.width * self.height

    @property
    def ratio(self):
        return self.width / self.height


class Polygon(InstanceField):
    """
    Represents a polygon annotation.

    Args:
        vertices List[List[Tuple[float]]]: Coordinates of the polygon. The
            values in are in format [[[point]...],[[point]...]] in image dimensions,
            where the first list contains all contours of the polygon, the internal
            ones - lists of single points of that contour.
        label (AnnotationClass): An AnnotationClass associated with the instance
        area (Optional[float]): Area of the polygon
        instance_id (Optional[id]): Id of the instance within the dataset
        confidence (Optional[float]): Confidence level of the field.
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Dictionary containing
            additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.Polygon`: A Polygon object.

    You can construct a polygon from the output of a model that is a mask:

    .. code-block:: python

        import efemarai as ef
        import numpy as np

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:50] = 255

        label = ef.AnnotationClass(id=1)
        polygon = ef.InstanceMask(label=label, data=mask).to_polygon()
        print(label, polygon)

        new_poly = ef.Polygon(label=label, vertices=polygon.vertices)
        print(new_poly)
    """

    def __init__(
        self,
        vertices: List[List[Tuple[float, float]]],
        label: AnnotationClass,
        area: float = None,
        instance_id: int = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            instance_id=instance_id,
            label=label,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )

        if not (
            isinstance(vertices, (list, tuple))
            and all(
                (isinstance(v, (list, tuple)))
                and all(
                    (isinstance(p, (tuple, list)))
                    and len(p) == 2
                    and all(isinstance(f, (float, int)) for f in p)
                    for p in v
                )
                for v in vertices
            )
        ):
            raise ValueError(
                "Polygon vertices are not of type List[List[List[float/int, float/int], ...]] or Tuple[Tuple[Tuple[float/int, float/int], ...]]"
            )

        self.vertices = vertices
        self.area = area
        self._raw_data = None

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  vertices={self.vertices}"
        res += f"\n  area={self.area}"
        res += f"\n  instance_id={self.instance_id}"
        res += f"\n  description={self.description}"
        res += f"\n  label={self.label}"
        res += f"\n  confidence={self.confidence}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += (
            f"\n  _raw_data={self._raw_data.shape}"
            if self._raw_data is not None
            else f"{self._raw_data}"
        )
        res += "\n)"
        return res

    def load_raw_data(self, width, height):
        self._raw_data = np.zeros((height, width), np.uint8)
        cv2.drawContours(
            self._raw_data,
            [
                np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
                for contour in self.vertices
            ],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

    def get_mask(self, width, height):
        if self._raw_data is None or self._raw_data.shape[:2] != (height, width):
            self.load_raw_data(width, height)
        return self._raw_data

    def set_mask(self, mask):
        self._raw_data = mask

    def set_vertices(self):
        polygons, polygons_area = create_polygons_from_mask(self._raw_data)
        self.vertices = polygons
        self.area = polygons_area


class Keypoint(InstanceField):
    """
    Represents a keypoint annotation.

    Args:
        x (float): X coordinate of the keypoint in absolute value.
        y (float): Y coordinate of the keypoint in absolute value.
        label (AnnotationClass): An AnnotationClass associated with the instance
        name (Optional[str]): Name of the keypoint, typically provided within a Skeleton.
        index (Optional[int]): Index of the keypoint within a Skeleton.
        occluded (Optional[bool]): Whether the keypoint is occluded.
        annotated (Optional[bool]): In case the keypoint is occluded, if it is annotated or with default coords.
        instance_id (Optional[id]): Id of the instance within the dataset
        confidence (Optional[float]): Confidence level of the field.
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.Keypoint`: A Keypoint object.
    """

    def __init__(
        self,
        x: float,
        y: float,
        label: AnnotationClass,
        name: str = None,
        index: int = None,
        annotated: bool = None,
        occluded: bool = None,
        instance_id: int = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            instance_id=instance_id,
            label=label,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )
        self.x = x if not isinstance(x, (np.int64, np.int32)) else x.item()
        self.y = y if not isinstance(y, (np.int64, np.int32)) else y.item()
        self.name = name
        self.index = index
        self.annotated = annotated
        self.occluded = occluded

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  x={self.x}"
        res += f"\n  y={self.y}"
        res += f"\n  occluded={self.occluded}"
        res += f"\n  instance_id={self.instance_id}"
        res += f"\n  description={self.description}"
        res += f"\n  label={self.label}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += f"\n  id={self.id}"
        res += "\n)"
        return res

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


class Skeleton(InstanceField):
    """
    Represents a skeleton annotation. A sekeleton is a group of keypoints that
    represent an instance of an object.

    Args:
        keypoints (List[Keypoint]): List of the Keypoints in the class
        label (AnnotationClass): An AnnotationClass associated with the instance
        edges: (Optional[List[Tuple[int]]]) List of edges between keypoints. Each edge is a tuple (from_keypoint_index, to_keypoint_index) and is used for visualization.
        instance_id (Optional[id]): Id of the instance within the dataset
        confidence (Optional[float]): Confidence level of the field.
        description (Optional[str]): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering to.
        key_name (Optional[str]): A string key, by which the field can be identified.
        user_attributes (Optional[Dict[str, object]]): Dictionary containing additional data regarding the field.

    Returns:
        :class: `efemarai.fields.annotation_fields.Skeleton`: A Skeleton object.


    Example of how to convert Mediapipe facemesh into the efemarai system:

    .. code-block:: python

        import efemarai as ef
        import cv2
        import mediapipe as mp
        from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

        model = mp.solutions.face_mesh
        # >> The image is an ef.Image variable

        results = model.process(cv2.cvtColor(image.data, cv2.COLOR_BGR2RGB))
        sdk_outputs = []
        label = ef.AnnotationClass(name="face")

        # For each face in the image
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            keypoints = []
            for j, idx in enumerate(face_landmarks.landmark):
                cord = _normalized_to_pixel_coordinates(idx.x, idx.y, image.width, image.height)
                keypoints.append(
                    ef.Keypoint(
                        x=cord[0],
                        y=cord[1],
                        name=model.name[j],
                        index=j,
                        annotated=True,
                        occluded=False,
                        instance_id=i,
                        ref_field=image,
                        label=label,
                        confidence=1,
                    )
                )

            sdk_outputs.append(
                ef.Skeleton(
                    label=label,
                    keypoints=keypoints,
                    ref_field=image,
                    instance_id=i,
                    confidence=1,
                )
            )
    """

    def __init__(
        self,
        keypoints: List[Keypoint],
        label: AnnotationClass,
        edges: list = None,
        instance_id: int = None,
        confidence: float = None,
        id: ObjectId = None,
        description: str = None,
        ref_field: list = None,
        key_name: str = None,
        user_attributes: Dict[str, object] = None,
    ):
        super().__init__(
            instance_id=instance_id,
            label=label,
            confidence=confidence,
            id=id,
            description=description,
            ref_field=ref_field,
            key_name=key_name,
            user_attributes=user_attributes,
        )
        self.keypoints = keypoints
        self.edges = [
            (f, t)
            if not isinstance(f, (np.int64, np.int32))
            or not isinstance(t, (np.int64, np.int32))
            else (int(f), int(t))
            for (f, t) in edges
        ]

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  keypoints={self.keypoints}"
        res += "\n)"
        return res


class PolarVector:
    def __init__(self, angle: float, length: float):
        self.angle = angle
        self.length = length
