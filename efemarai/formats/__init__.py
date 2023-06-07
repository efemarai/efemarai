import numpy as np
from PIL import Image as PIL_Image

from efemarai.fields import AnnotationClass, BoundingBox, Image, InstanceMask, Polygon, Text
from efemarai.spec import call, create


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()

def mask_to_polygon(mask):
    polygon = mask.to_polygon()
    polygon.load_raw_data(mask.width, mask.height)
    return polygon


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

TEXT_EQA_INPUT = create(
    lambda *args: list(args),
    create(Text, text="context", key_name="'context'"),
    create(Text, text="question", key_name="'question'"),
)

TEXT_EQA_OUTPUT = create(Text, text="answer", key_name="'answer'")

TEXT_EQA_DATASET = (TEXT_EQA_INPUT, TEXT_EQA_OUTPUT)

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

ULTRALYTICS_DETECTION = {
    ".boxes": {
        call(zip, ".xyxy", ".cls", ".conf"): [
            create(
                BoundingBox,
                xyxy={0: lambda x: x.tolist()},
                label=create(AnnotationClass, id="[1]", confidence="[2]"),
            )
        ]
    }
}

DEFAULT_INPUT_NP_FORMAT = {".image": {".data": np.array}}

SUPERVISION_TARGET = {
    call(zip, ".xyxy", ".class_id"): [
        create(
            BoundingBox,
            xyxy={0: lambda x: x},
            label=create(
                AnnotationClass,
                id={1: lambda x: x},
            ),
        )
    ]
}

SUPERVISION_DETECTION = (COCO_INPUT, SUPERVISION_TARGET)

ROBOFLOW_DETECTION = {
    "predictions": [
        create(
            BoundingBox,
            xyxy=call(
                BoundingBox.convert,
                box=lambda x: (x["x"], x["y"], x["width"], x["height"]),
                source_format=BoundingBox.CENTERWH_ABSOLUTE,
            ),
            label=create(AnnotationClass, id="class", confidence="confidence"),
        ),
    ]
}

COCO_TARGET_INSTANCE = [
    create(
        Polygon,
        vertices="segmentation",
        label=create(AnnotationClass, id="category_id"),
    ),
]

COCO_INSTANCE_DATASET = (COCO_INPUT, COCO_TARGET_INSTANCE)

DETECTRON_INSTANCE_DETECTION = {
    "._fields": {
        call(zip, "scores", "pred_classes", "pred_masks"): [
            call(
                mask_to_polygon,
                create(
                    InstanceMask,
                    data=call(InstanceMask.bool_to_uint8, {2: tensor_to_numpy}),
                    label=create(
                        AnnotationClass,
                        id={1: tensor_to_numpy},
                        confidence={0: tensor_to_numpy},
                    ),
                ),
            )
        ]
    }
}
