import glob
import os
from itertools import islice

import cv2

from efemarai.dataset import DatasetFormat
from efemarai.fields import Datapoint, Image, Tag
from efemarai.formats.config import CLASS_COLORS, IMAGE_EXTENSIONS


def get_image_width_height(image_path):
    img = cv2.imread(image_path)
    return img.shape[1], img.shape[0]


def create_imagenet_dataset(
    project,
    name,
    stage,
    data_url,
    num_datapoints,
):
    """Upload an ImageNet style dataset in the system."""
    dataset = project.create_dataset(
        name=name, stage=stage, format=DatasetFormat.Custom
    )

    # Creating classes based on folders
    classes = sorted(next((os.walk(data_url.replace("file://", "", 1))), None)[1])

    for i, class_name in enumerate(classes):
        dataset.add_annotation_class(
            id=i,
            name=class_name,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
        )

    # Loading images into datapoints for each folder
    for class_id, class_name in enumerate(classes):
        for ext in IMAGE_EXTENSIONS:
            for image_file in islice(
                glob.iglob(
                    f"{data_url.replace('file://', '', 1)}/{class_name}/*.{ext}"
                ),
                num_datapoints,
            ):
                label = dataset.get_annotation_class(id=class_id)

                width, height = get_image_width_height(image_file)

                image = Image(
                    file_path=image_file,
                    width=width,
                    height=height,
                    user_attributes={
                        "extension": ext,
                        "original_data": os.path.join(class_name, image_file),
                    },
                )
                tag = Tag(label=label, ref_field=image)

                datapoint = Datapoint(
                    dataset=dataset, inputs={"image": image}, targets={"class": tag}
                )
                datapoint.upload()

    dataset.finalize()

    return dataset
