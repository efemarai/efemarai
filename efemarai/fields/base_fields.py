import base64
import json
from typing import Dict, List

import numpy as np
from bson.objectid import ObjectId


def _decode_ndarray(xs):
    data = base64.b64decode(xs["data"])
    shape = xs["shape"]
    dtype = np.dtype(xs["dtype"])
    return np.frombuffer(data, dtype=dtype).reshape(*shape)


def _encode_ndarray(xs):
    data = xs.data if xs.flags["C_CONTIGUOUS"] else xs.tobytes()
    return {
        "data": base64.b64encode(data).decode("ascii"),
        "shape": xs.shape,
        "dtype": str(xs.dtype),
    }


def sdk_serialize(obj):
    return json.dumps(obj, default=_serialize_function)


def _serialize_function(x):
    if isinstance(x, np.ndarray):
        res = _encode_ndarray(x)
        return res
    if isinstance(x, ObjectId):
        return str(x)
    return x.__dict__


class BaseField:
    """
    At the core of each data field, the BaseField creates a common representation of an
    element in Efemarai. It is used to store the id, description, reference field, name,
    confidence, and any user added attributes.

    Args:
        id (ObjectId): A BaseField unique identifier.
        description (str): A description of the object or annotation information.
        ref_field (List[object]): A list of objects the current BaseField is refering.
        key_name (str): A string key, by which the field can be identified.
        confidence (float): Confidence level of the field.
        user_attributes (Dict[str, object]): Optional dictionary containing additional
            data regarding the field.

    Returns:
        :class:`efemarai.fields.base_field.BaseField`: A BaseField object.
    """

    def __init__(
        self,
        id: ObjectId = None,
        description: str = None,
        ref_field: List[object] = None,
        key_name: str = None,
        confidence: float = None,
        user_attributes: Dict[str, object] = None,
    ):

        self.id = id if id is not None else ObjectId()
        self.description = description
        self.key_name = key_name
        self.confidence = confidence
        self._cls = self.__class__.__name__
        self.ref_field = ref_field
        self.user_attributes = user_attributes

        # Propagate ref_field to frames in video
        if ref_field is not None:
            if not isinstance(ref_field, list):
                if hasattr(ref_field, "_cls") and ref_field._cls == "Video":
                    self.frames = (
                        [ref_field._index] if ref_field._index is not None else []
                    )
                self.ref_field = [
                    ref_field.id if hasattr(ref_field, "id") else ref_field
                ]
            else:
                for ref in ref_field:
                    if hasattr(ref, "_cls") and ref._cls == "Video":
                        if not hasattr(self, "frames") or not self.frames:
                            self.frames = []
                        self.frames.append([ref._index for ref in ref_field])
                self.ref_field = [
                    ref.id if hasattr(ref, "id") else ref for ref in ref_field
                ]

    def _serialize(self):
        return sdk_serialize(self)

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  description={self.description}"
        res += f"\n  ref_field={self.ref_field}"
        res += f"\n  key_name={self.key_name}"
        res += f"\n  confidence={self.confidence}"
        res += f"\n  user_attributes={self.user_attributes}"
        res += "\n)"
        return res
