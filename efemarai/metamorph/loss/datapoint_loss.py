from collections import defaultdict

from efemarai import Datapoint
from efemarai.metamorph.loss.bounding_box_loss import bounding_box_loss, preprocess_bbox
from efemarai.metamorph.loss.classification_loss import (
    classification_loss,
    preprocess_tags,
)
from efemarai.metamorph.loss.instance_segmentation_loss import (
    mask_loss,
    preprocess_polygon,
)
from efemarai.metamorph.loss.keypoint_loss import (
    keypoint_loss,
    preprocess_keypoints,
    preprocess_skeleton,
)
from efemarai.metamorph.loss.regression_loss import regression_loss
from efemarai.metamorph.loss.text_loss import preprocess_text, text_loss

INSTANCE_PREPROCESSORS = {
    "Polygon": preprocess_polygon,
    "BoundingBox": preprocess_bbox,
    "Keypoint": preprocess_keypoints,
    "Skeleton": preprocess_skeleton,
}
INSTANCE_LOSSES = {
    "Polygon": mask_loss,
    "BoundingBox": bounding_box_loss,
    "Keypoint": keypoint_loss,
    "Skeleton": keypoint_loss,
}


def load_mask(fields):
    base_fields = fields.targets if isinstance(fields, Datapoint) else fields.outputs
    for target in base_fields:
        if target._cls == "Polygon":
            # Skip loading raw data if it is already loaded
            if target._raw_data is not None:
                continue
            inputs = (
                fields.inputs if isinstance(fields, Datapoint) else fields.inputs.inputs
            )
            ref_inputs = list(
                filter(
                    lambda _input: _input.id in target.ref_field
                    and _input._cls == "Image",
                    inputs,
                )
            )
            for ref_input in ref_inputs:
                target.load_raw_data(ref_input.width, ref_input.height)

        # TODO Handle masks
        if "Mask" in target._cls:
            print("Loading mask: ", target.data.shape)


def group_fields_by_type(fields):
    """
    Args:
        fields (dict): orm.models.Datapoint or orm.models.ModelOutput.

        key (str): Dict key corresponding to the objects datapoints and model_outputs.

    Return:
        fields grouped by type (dict(ef.field : dict(ref_field:[ef.BaseField]))):
    """
    fields_grouped_by_type = defaultdict(lambda: defaultdict(list))

    for field in fields:
        field.ref_field.sort()
        # Loss is not calculated if "require_loss" is added as False in the user_attributes.
        if field._requires_loss:
            fields_grouped_by_type[field._cls][str(field.ref_field)].append(field)
    return fields_grouped_by_type


def update_loss_dict(source_dict, target_dict, weight=1):
    for key in source_dict.keys():
        if key not in target_dict:
            target_dict.update({key: source_dict[key]})

        if isinstance(target_dict[key], (dict, list)):
            target_dict[key].extend(source_dict[key])

        else:
            target_dict[key] += (
                source_dict[key] * weight
                if "failure_score" in key
                else source_dict[key]
            )

    return target_dict


def datapoint_loss(
    datapoint, model_output, class_weights, field_weights, confusion_weights=None
):
    if confusion_weights is None:
        confusion_weights = defaultdict(lambda: (1 / 3, 1 / 3, 1 / 3))

    loss = {
        "pred_instances_info": [],
        "gt_instance_info": [],
        "gt_scores_info": [],
        "pred_scores_info": [],
        "fp_scores_info": [],
        "fn_scores_info": [],
        "failure_score_unnormalized": 0,
        "failure_score_normalization_constant": 0,
    }

    load_mask(datapoint)
    load_mask(model_output)

    grouped_target_fields = group_fields_by_type(datapoint.targets)
    grouped_output_fields = group_fields_by_type(model_output.outputs)

    field_types = set(
        list(grouped_target_fields.keys()) + list(grouped_output_fields.keys())
    )

    # No gt and no pred.
    if not field_types:
        loss["failure_score_normalization_constant"] = 1
        return loss

    # No target but prediciton or no prediction but target
    if (not grouped_target_fields and grouped_output_fields) or (
        grouped_target_fields and not grouped_output_fields
    ):
        loss["failure_score_unnormalized"] += 1
        loss["failure_score_normalization_constant"] += 1
        return loss

    # TODO: Instantiate the metric only once before reaching the loss calculation.
    if "Text" in field_types:
        from torchmetrics.text.bert import BERTScore

        # Single process for the dataloader inside the master process. Required because we use mp.Pool
        bertscore = BERTScore(num_threads=0, verbose=True)

    # TODO: filter for field.user_attributes.get("loss") == False
    for field_type in field_types:
        loss_type_weight = field_weights[field_type]
        field_loss = {
            "pred_instances_info": [],
            "gt_instance_info": [],
            "gt_scores_info": [],
            "pred_scores_info": [],
            "fp_scores_info": [],
            "fn_scores_info": [],
            "failure_score_unnormalized": 0,
            "failure_score_normalization_constant": 0,
        }
        for target_ref in grouped_target_fields[field_type]:
            # TODO: Think of a better way to handle this case. Is that the case?
            if target_ref not in grouped_output_fields[field_type]:
                field_loss["failure_score_unnormalized"] += 1
                field_loss["failure_score_normalization_constant"] += 1
                continue

            # Object Detection, Instance Segmentation, Keypoints
            if field_type in ["BoundingBox", "Polygon", "Keypoint", "Skeleton"]:
                (
                    gt_instances,
                    gt_classes,
                    pred_instances,
                    pred_classes,
                    pred_confidence,
                ) = INSTANCE_PREPROCESSORS[field_type](
                    grouped_target_fields[field_type][target_ref],
                    grouped_output_fields[field_type][target_ref],
                )

                tmp_loss = INSTANCE_LOSSES[field_type](
                    gt_instances,
                    gt_classes,
                    pred_instances,
                    pred_classes,
                    pred_confidence,
                    class_weights,
                    confusion_weights,
                )

            # TODO: Add class weights
            # Regression
            elif field_type == "Value":
                tmp_loss = regression_loss(
                    grouped_target_fields[field_type][target_ref],
                    grouped_output_fields[field_type][target_ref],
                )

            # Classification
            elif field_type == "Tag":
                gt_classes, pred_confidence = preprocess_tags(
                    grouped_target_fields[field_type][target_ref],
                    grouped_output_fields[field_type][target_ref],
                )
                tmp_loss = classification_loss(
                    gt_classes, pred_confidence, class_weights
                )

            # Text
            elif field_type == "Text":
                gt_classes, pred_confidence = preprocess_text(
                    grouped_target_fields[field_type][target_ref],
                    grouped_output_fields[field_type][target_ref],
                )
                tmp_loss = text_loss(
                    metric=bertscore, targets=gt_classes, preds=pred_confidence
                )

            else:
                # TODO  compute loss for other fields
                print(f"Loss not implemented for field_type '{field_type}'.")
                tmp_loss = {
                    "failure_score_unnormalized": 0,
                    "failure_score_normalization_constant": 1,
                }

            field_loss = update_loss_dict(tmp_loss, field_loss)
        loss = update_loss_dict(field_loss, loss, weight=loss_type_weight)
    return loss
