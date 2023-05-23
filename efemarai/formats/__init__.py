import numpy as np
from PIL import Image as PIL_Image

from efemarai.fields import AnnotationClass, BoundingBox, Image
from efemarai.spec import call, create


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


COCO_INPUT = create(Image, data=np.array, key_name="'image'")

COCO_TARGET = [
    create(
        BoundingBox,
        label=create(AnnotationClass, id="category_id"),
        xyxy=call(
            BoundingBox.convert,
            box="bbox",
            source_format=BoundingBox.XYWH_ABSOLUTE,
        ),
    )
]

COCO_DATASET = (COCO_INPUT, COCO_TARGET)

DEFAULT_INPUT_FORMAT = {".image": {".data": PIL_Image.fromarray}}

TORCHVISION_DETECTION = {
    call(zip, "boxes", "labels"): [
        create(
            BoundingBox,
            xyxy={0: tensor_to_numpy},
            label=create(AnnotationClass, id={1: tensor_to_numpy}),
        )
    ]
}
