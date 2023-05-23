import inspect
import sys

import numpy as np

from efemarai.fields.annotation_fields import (
    AnnotationClass,
    BoundingBox,
    Keypoint,
    Polygon,
    Skeleton,
    Tag,
    Value,
)
from efemarai.fields.base_fields import (
    BaseField,
    _decode_ndarray,
    _encode_ndarray,
    sdk_serialize,
)
from efemarai.fields.data_fields import (
    DataFrame,
    Image,
    InstanceMask,
    Mask,
    Text,
    Video,
    VideoFrame,
    create_polygons_from_mask,
)
from efemarai.fields.datapoint import Datapoint, ModelOutput

SDK_CLASS_STRUCTURE = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))


def sdk_deserialize(data):
    if "_cls" not in data:
        print(f":poop: Cannot locate class from the data {data.keys()}.")
        return None

    c = SDK_CLASS_STRUCTURE[data["_cls"]]
    sdk_type = data.pop("_cls")

    # "data" stores the loaded image in an np.ndarray
    if (
        "data" in data
        and isinstance(data["data"], dict)
        and all(k in data["data"].keys() for k in ("data", "shape", "dtype"))
    ):
        data["data"] = _decode_ndarray(data["data"])

    # Find id
    if "_id" in data:
        if isinstance(data["_id"], dict):
            # Get only the oid value, not the whole dict
            data["id"] = data["_id"]["$oid"]
        else:
            data["id"] = data["_id"]
        data.pop("_id")
    elif "id" in data:
        if isinstance(data["id"], dict):
            data["id"] = data["id"]["$oid"]
    else:
        print(f":poop: Cannot resolve id from data. ({data.keys()})")

    if sdk_type == "Video":
        for frame in data["frames"]:
            frame.pop("_raw", None)
            frame.pop("_cls", None)
            frame["id"] = frame["id"]["$oid"]  # TODO: Find a smarter way to do this
        data["frames"] = [VideoFrame(**f) for f in data["frames"]]

    try:
        obj = c(**data)
    except Exception:
        print(
            f"Cannot create an instance of class '{c}' with keys {data.keys()} and data {data}"
        )
    return obj
