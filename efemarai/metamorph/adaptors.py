import random
from copy import deepcopy
from functools import wraps

import albumentations as A
import numpy as np

from efemarai.fields import BoundingBox, Image, Keypoint, Polygon, Skeleton, Text


def apply_nlpaug():
    def decorator(create_operator):
        supported_data_types = {Text}
        supported_annotation_types = {Text}

        @wraps(create_operator)
        def wrapper(*args, **kwargs):
            def apply_operator(datapoint):
                # Specify how a single field is to be transformed
                def transform(field, annotations, field_metadata=None):
                    operator = create_operator(*args, **kwargs)
                    transformed_field = deepcopy(field)
                    output = operator(transformed_field.text)
                    transformed_field.text = output[0]

                    return transformed_field, annotations

                result = transform_datapoint(
                    transform,
                    datapoint,
                    supported_data_types,
                    supported_annotation_types,
                )

                return (result,)

            return apply_operator

        return wrapper

    return decorator


def apply_albumentation(filter_instances=True):
    def decorator(create_operator):
        supported_data_types = {Image}
        supported_annotation_types = {BoundingBox, Polygon, Keypoint, Skeleton}

        @wraps(create_operator)
        def wrapper(*args, **kwargs):
            def apply_operator(datapoint):
                # Specify how a single field is to be transformed
                def transform(field, annotations, field_metadata=None):
                    operator = create_operator(*args, **kwargs)
                    input_targets = prepare_targets(field, annotations)
                    output_targets = operator(**input_targets)

                    if filter_instances:
                        output_targets = filter_targets(output_targets)

                    image, annotations = prepare_fields(output_targets)
                    image.key_name = field.key_name

                    return image, annotations

                result = transform_datapoint(
                    transform,
                    datapoint,
                    supported_data_types,
                    supported_annotation_types,
                )

                return (result,)

            return apply_operator

        return wrapper

    return decorator


def apply_paste():
    def decorator(create_operator):
        supported_data_types = {Image}
        supported_annotation_types = {BoundingBox, Polygon, Keypoint, Skeleton}

        @wraps(create_operator)
        def wrapper(*args, **kwargs):
            def apply_operator(asset, datapoint):
                # Specify how a single field is to be transformed
                def transform(field, annotations, field_metadata):
                    paste = create_operator(*args, **kwargs)
                    image, annotations = paste(
                        asset, field, annotations, field_metadata
                    )
                    return image, annotations

                result = transform_datapoint(
                    transform,
                    datapoint,
                    supported_data_types,
                    supported_annotation_types,
                )

                return (result,)

            return apply_operator

        return wrapper

    return decorator


def transform_datapoint(
    transform,
    datapoint,
    supported_data_types,
    supported_annotation_types,
):
    result = datapoint.__class__(
        dataset=datapoint.dataset,
        # TODO: Figure out how to handle metadata
        # metadata=deepcopy(datapoint.metadata),
        # synthetic=True,
    )

    seed = random.randint(0, (1 << 32) - 1)

    id_swaps = {}
    processed_annotations = set()

    for field in datapoint.inputs:
        # print("Field:::", field)
        # print("annotation:::", datapoint.targets)
        annotations = [
            annotation
            for annotation in datapoint.targets
            if field.id in annotation.ref_field
        ]

        # Make sure to process annotations just once
        annotations = [
            annotation
            for annotation in annotations
            if annotation.id not in processed_annotations
        ]
        processed_annotations.update([annotation.id for annotation in annotations])

        random.seed(seed)

        field_metadata = [
            # metadata_field
            # for metadata_field in datapoint.metadata
            # if field.id in metadata_field.ref_field
        ]

        transformed_field, transformed_annotations = transform_field(
            transform,
            field,
            annotations,
            field_metadata,
            supported_data_types,
            supported_annotation_types,
        )

        id_swaps[field.id] = transformed_field.id

        result.inputs.append(transformed_field)
        result.targets.extend(transformed_annotations)

    for target in result.targets:
        target.ref_field = [id_swaps.get(ref, ref) for ref in target.ref_field]

    random.seed()

    return result


def transform_field(
    transform,
    field,
    annotations,
    field_metadata,
    supported_data_types,
    supported_annotation_types,
):
    # Skip fields that are not to be transformed
    field_supported = type(field) in supported_data_types
    requires_transform = field._requires_transform
    if not field_supported or not requires_transform:
        return deepcopy(field), deepcopy(annotations)

    # Split annotations into un/supported
    supported_annotations = []
    unsupported_annotations = []
    for annotation in annotations:
        if type(annotation) in supported_annotation_types:
            supported_annotations.append(annotation)
        else:
            unsupported_annotations.append(annotation)

    # Transform field and supported annotations
    transformed_field, transformed_annotations = transform(
        field, supported_annotations, field_metadata
    )

    # Update ref_field of unsupported annotations
    for annotation in unsupported_annotations:
        annotation = deepcopy(annotation)
        transformed_annotations.append(annotation)

    return transformed_field, transformed_annotations


def prepare_targets(image, annotations):
    """Convert an image and its annotations to an albumentation targets."""
    height, width = image.data.shape[:2]

    box_fields = [field for field in annotations if type(field) is BoundingBox]
    boxes = [
        A.core.bbox_utils.convert_bbox_to_albumentations(
            bbox=box.xyxy,
            source_format="pascal_voc",
            rows=height,
            cols=width,
        )
        for box in box_fields
    ]

    polygon_fields = [field for field in annotations if type(field) is Polygon]
    masks = [polygon.get_mask(image.width, image.height) for polygon in polygon_fields]

    keypoint_fields = [field for field in annotations if type(field) is Keypoint]
    skeleton_fields = [field for field in annotations if type(field) is Skeleton]

    keypoints = [
        A.core.keypoints_utils.convert_keypoint_to_albumentations(
            keypoint=keypoint.to_xy(),
            source_format="xy",
            rows=height,
            cols=width,
        )
        for keypoint in keypoint_fields
    ]

    for skeleton in skeleton_fields:
        keypoints.extend(
            A.core.keypoints_utils.convert_keypoints_to_albumentations(
                keypoints=skeleton.to_xy(),
                source_format="xy",
                rows=height,
                cols=width,
            )
        )

    targets = {"image": image.data}

    if boxes:
        targets["bboxes"] = boxes
        targets["box_fields"] = box_fields

    if masks:
        targets["masks"] = masks
        targets["polygon_fields"] = polygon_fields

    if keypoints:
        targets["keypoints"] = keypoints
        targets["keypoint_fields"] = keypoint_fields
        targets["skeleton_fields"] = skeleton_fields

    return targets


def filter_targets(targets):
    """Remove annotations that are empty or outside of the image."""
    removed_instance_ids = remove_instances(targets)

    filtered_targets = {"image": targets["image"]}

    if "bboxes" in targets:
        boxes = []
        box_fields = []

        for bbox, field in zip(targets["bboxes"], targets["box_fields"]):
            if field.instance_id not in removed_instance_ids:
                boxes.append(bbox)
                box_fields.append(field)

        filtered_targets["bboxes"] = boxes
        filtered_targets["box_fields"] = box_fields

    if "masks" in targets:
        masks = []
        polygon_fields = []
        for mask, field in zip(targets["masks"], targets["polygon_fields"]):
            if field.instance_id not in removed_instance_ids:
                masks.append(mask)
                polygon_fields.append(field)

        filtered_targets["masks"] = masks
        filtered_targets["polygon_fields"] = polygon_fields

    if "keypoints" in targets:
        keypoints = []
        keypoint_fields = []
        skeleton_fields = []

        for keypoint, field in zip(targets["keypoints"], targets["keypoint_fields"]):
            if field.instance_id not in removed_instance_ids:
                keypoints.append(keypoint)
                keypoint_fields.append(field)

        start_index = len(targets["keypoint_fields"])
        for field in targets["skeleton_fields"]:
            end_index = start_index + len(field.keypoints)

            if field.instance_id not in removed_instance_ids:
                keypoints.extend(targets["keypoints"][start_index:end_index])
                skeleton_fields.append(field)

            start_index = end_index

        filtered_targets["keypoints"] = keypoints
        filtered_targets["keypoint_fields"] = keypoint_fields
        filtered_targets["skeleton_fields"] = skeleton_fields

    return filtered_targets


def remove_instances(targets):
    """Get the IDs of instances that are outside of the image or have empty annotations."""
    height, width = targets["image"].shape[:2]

    removed_instance_ids = set()

    # Remove instances with bounding boxes outside of the image
    if "box_fields" in targets:
        for bbox, field in zip(targets["bboxes"], targets["box_fields"]):
            x1, y1, x2, y2 = bbox
            if x2 < 0 or width < x1 or y2 < 0 or height < y1:
                removed_instance_ids.add(field.instance_id)

    # Remove instances with empty masks
    if "polygon_fields" in targets:
        for mask, field in zip(targets["masks"], targets["polygon_fields"]):
            if np.count_nonzero(mask) == 0:
                removed_instance_ids.add(field.instance_id)

    # Remove instances with all keypoints outside of the image
    if "keypoint_fields" in targets:
        inside_instance_ids = set()
        outside_instance_ids = set()
        for keypoint, field in zip(targets["keypoints"], targets["keypoint_fields"]):
            x, y, *_ = keypoint
            if not field.annotated:
                continue

            if x < 0 or width < x or y < 0 or height < y:
                outside_instance_ids.add(field.instance_id)
            else:
                inside_instance_ids.add(field.instance_id)

        for instance_id in outside_instance_ids:
            if instance_id not in inside_instance_ids:
                removed_instance_ids.add(instance_id)

    # Remove instances with skeletons completely outside of the image
    if "skeleton_fields" in targets:
        start_index = len(targets["keypoint_fields"])
        for skeleton in targets["skeleton_fields"]:
            end_index = start_index + len(skeleton.keypoints)

            # Get all annotated keypoints
            keypoints = np.array(
                [
                    keypoint
                    for keypoint, field in zip(
                        targets["keypoints"][start_index:end_index], skeleton.keypoints
                    )
                    if field.annotated
                ]
            )

            # Do not remove an instance based on keypoints if none of them are annotated
            if len(keypoints) == 0:
                continue

            x1, y1, *_ = keypoints.min(axis=0)
            x2, y2, *_ = keypoints.max(axis=0)

            if x2 < 0 or width < x1 or y2 < 0 or height < y1:
                removed_instance_ids.add(skeleton.instance_id)

            start_index = end_index

    return removed_instance_ids


def prepare_fields(targets):
    """Convert albumentation targets to datapoint fields."""
    height, width = targets["image"].shape[:2]
    image = Image(width=width, height=height, data=targets["image"])

    annotations = []
    annotations.extend(prepare_box_fields(image, targets))
    annotations.extend(prepare_polygon_fields(image, targets))
    annotations.extend(prepare_keypoint_fields(image, targets))
    annotations.extend(prepare_skeleton_fields(image, targets))

    return image, annotations


def prepare_box_fields(image, targets):
    if "bboxes" not in targets:
        return []

    box_fields = []

    for bbox, field in zip(targets["bboxes"], targets["box_fields"]):
        x1, y1, x2, y2 = A.core.bbox_utils.convert_bbox_from_albumentations(
            bbox=bbox,
            target_format="pascal_voc",
            rows=image.height,
            cols=image.width,
        )

        # Make sure boxes raimain within the image
        def clip_x(x):
            return max(0, min(x, image.width))

        def clip_y(y):
            return max(0, min(y, image.height))

        x1, y1, x2, y2 = clip_x(x1), clip_y(y1), clip_x(x2), clip_y(y2)

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box_fields.append(
            BoundingBox(
                xyxy=[x1, y1, x2, y2],
                area=area,
                ref_field=[image.id],
                instance_id=field.instance_id,
                label=field.label,
            )
        )

    return box_fields


def prepare_polygon_fields(image, targets):
    if "masks" not in targets:
        return []

    polygon_fields = []

    for mask, field in zip(targets["masks"], targets["polygon_fields"]):
        polygon = Polygon(
            vertices=[],
            ref_field=[image.id],
            instance_id=field.instance_id,
            label=field.label,
        )
        polygon.set_mask(mask)
        polygon.set_vertices()
        polygon_fields.append(polygon)

    return polygon_fields


def prepare_keypoint_fields(image, targets):
    if "keypoints" not in targets:
        return []

    keypoint_fields = targets["keypoint_fields"]
    keypoints = A.core.keypoints_utils.convert_keypoints_from_albumentations(
        keypoints=targets["keypoints"][: len(keypoint_fields)],
        target_format="xy",
        rows=image.height,
        cols=image.width,
    )

    return [
        Keypoint(
            ref_field=[image.id],
            instance_id=field.instance_id,
            label=field.label,
            x=x,
            y=y,
            name=field.name,
            index=field.index,
            annotated=field.annotated,
            occluded=field.occluded,
        )
        for (x, y), field in zip(keypoints, keypoint_fields)
    ]


def prepare_skeleton_fields(image, targets):
    if "keypoints" not in targets:
        return []

    skeleton_fields = []

    # Skip keypoints corresponding to plain keypoint fields
    start_index = len(targets["keypoint_fields"])

    for skeleton in targets["skeleton_fields"]:
        end_index = start_index + len(skeleton.keypoints)
        keypoints = A.core.keypoints_utils.convert_keypoints_from_albumentations(
            keypoints=targets["keypoints"][start_index:end_index],
            target_format="xy",
            rows=image.height,
            cols=image.width,
        )
        start_index = end_index

        skeleton_fields.append(
            Skeleton(
                ref_field=[image.id],
                instance_id=skeleton.instance_id,
                label=skeleton.label,
                keypoints=[
                    Keypoint(
                        ref_field=[image.id],
                        instance_id=field.instance_id,
                        label=field.label,
                        x=x,
                        y=y,
                        name=field.name,
                        index=field.index,
                        annotated=field.annotated,
                        occluded=(
                            field.occluded
                            or x < 0
                            or image.width < x
                            or y < 0
                            or image.height < y
                        ),
                    )
                    for (x, y), field in zip(keypoints, skeleton.keypoints)
                ],
                edges=skeleton.edges,
            )
        )

    return skeleton_fields
