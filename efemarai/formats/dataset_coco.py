import itertools
import os
from copy import copy

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from tqdm import tqdm

from efemarai.dataset import DatasetFormat
from efemarai.fields import (
    BoundingBox,
    Datapoint,
    Image,
    Keypoint,
    Polygon,
    Skeleton,
    create_polygons_from_mask,
)
from efemarai.formats.config import CLASS_COLORS


def create_mask(size, annotation):
    mask = np.zeros(size[:2], np.uint8)
    if type(annotation["segmentation"]["counts"]) is list:
        # uncompressed RLE
        rle = coco_mask.frPyObjects(
            annotation["segmentation"], mask.shape[0], mask.shape[1]
        )
    else:
        # plain RLE
        rle = annotation["segmentation"]

    mask = coco_mask.decode(rle) * 255
    return mask


def create_coco_dataset(
    project,
    name,
    stage,
    data_url,
    annotations_url,
    num_datapoints,
    mask_generation,
    min_asset_area,
):
    """Upload a COCO style dataset in the system."""
    dataset = project.create_dataset(
        name=name, stage=stage, format=DatasetFormat.Custom
    )

    coco = COCO(annotations_url)

    for i, category in enumerate(coco.loadCats(coco.getCatIds())):
        dataset.add_annotation_class(
            id=category["id"],
            name=category["name"],
            category=category.get("supercategory", "supercategory"),
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
        )

    for cocoImg in tqdm(coco.loadImgs(coco.getImgIds())[:num_datapoints]):
        attr_data = copy(cocoImg)
        attr_data.pop("width", None)
        attr_data.pop("height", None)
        image = Image(
            file_path=os.path.join(data_url, cocoImg["file_name"]),
            width=cocoImg["width"],
            height=cocoImg["height"],
            user_attributes={"original_data": attr_data},
        )

        datapoint = Datapoint(
            dataset=dataset,
            inputs={"image": image},
        )

        annotations = coco.loadAnns(coco.getAnnIds(imgIds=cocoImg["id"]))
        for instance_id, cocoAnn in enumerate(annotations):
            # "category_id", "bbox", "area", "segmentation", "keypoints"

            # implicitly ensures only registered classes are added
            label = dataset.get_annotation_class(id=cocoAnn["category_id"])

            # or in case class name is provided
            # label = dataset.get_annotation_class(name=annotation["category_name"]),

            if "bbox" in cocoAnn:
                datapoint.add_target(
                    BoundingBox(
                        xyxy=[
                            cocoAnn["bbox"][0],
                            cocoAnn["bbox"][1],
                            cocoAnn["bbox"][0] + cocoAnn["bbox"][2],
                            cocoAnn["bbox"][1] + cocoAnn["bbox"][3],
                        ],
                        label=label,
                        instance_id=instance_id,
                        ref_field=image,
                    )
                )

            if "segmentation" in cocoAnn:
                if isinstance(cocoAnn["segmentation"], dict):
                    # RLE encoding -> create polygon
                    mask_img = np.zeros((cocoImg["width"], cocoImg["height"]))
                    mask_img = create_mask(mask_img.shape, cocoAnn)
                    polygons, polygons_area = create_polygons_from_mask(mask_img)
                    polygons = [list(itertools.chain(*polygon)) for polygon in polygons]
                    cocoAnn["segmentation"] = polygons
                    cocoAnn["area"] = polygons_area

                vertices = []
                for polygon in cocoAnn["segmentation"]:
                    vertices.append(list(zip(polygon[:-1][::2], polygon[1:][::2])))

                datapoint.add_target(
                    Polygon(
                        label=label,
                        vertices=vertices,
                        area=cocoAnn["area"],
                        instance_id=instance_id,
                        ref_field=image,
                    )
                )

            # The person category has fully annotated skeleton which looks like
            # "id": 1,
            # "name": "person",
            # "keypoints": [
            #     "nose","left_eye","right_eye","left_ear","right_ear",
            #     "left_shoulder","right_shoulder","left_elbow","right_elbow",
            #     "left_wrist","right_wrist","left_hip","right_hip",
            #     "left_knee","right_knee","left_ankle","right_ankle"
            # ],
            # "skeleton": [
            #     [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            #     [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            # ]
            # Add single skeleton if info available
            cocoCat = coco.loadCats(label.id)[0]
            if "skeleton" in cocoCat and "keypoints" in cocoCat:
                keypoints = [
                    Keypoint(
                        x=x,
                        y=y,
                        name=cocoCat["keypoints"][i],
                        index=i,
                        annotated=v > 0,
                        occluded=v == 1,
                        instance_id=instance_id,
                        ref_field=image,
                        label=label,
                    )
                    for i, (x, y, v) in enumerate(
                        Keypoint.chunks(cocoAnn["keypoints"], 3)
                    )
                ]

                datapoint.add_target(
                    Skeleton(
                        label=label,
                        keypoints=keypoints,
                        edges=cocoCat["skeleton"],
                        ref_field=image,
                        instance_id=instance_id,
                    )
                )
            elif "keypoints" in cocoAnn:
                # else just add multiple keypoints, without grouping them
                for i, (x, y, v) in enumerate(Keypoint.chunks(cocoAnn["keypoints"], 3)):
                    datapoint.add_target(
                        Keypoint(
                            label=label,
                            x=x,
                            y=y,
                            occluded=v == 1,
                            annotated=v > 0,
                            index=i,
                            instance_id=instance_id,
                            ref_field=image,
                        )
                    )

        datapoint.upload()

    dataset.finalize(mask_generation=mask_generation, min_asset_area=min_asset_area)
    return dataset
