import numpy as np
from PIL import Image as PIL_Image

from efemarai.fields import AnnotationClass, BoundingBox, Image, Text
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

VQA_INPUT = create(
    lambda *args: list(args),
    create(Text, text="question", key_name="'question'"),
    create(Image, data=lambda x: x["image"], key_name="'image'"),
)

VQA_OUTPUT = create(Text, text="answers", key_name="'answer'")

VQA_DATASET = (VQA_INPUT, VQA_OUTPUT)
