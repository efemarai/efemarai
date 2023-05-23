# import numpy as np

from efemarai.metamorph.loss.datapoint_loss import datapoint_loss

# from efemarai.metamorph.loss.bounding_box_loss import bounding_box_loss
# from efemarai.metamorph.loss.instance_segmentation_loss import mask_loss
# from efemarai.metamorph.loss.keypoint_loss import (
#     keypoint_loss,
#     scale_coords_to_zero_one,
# )

# from orm import DatasetStore, ProblemType

MASK_OUT_KEYS = ["area", "info"]


def failure_loss(
    datapoint, model_output, class_weights, field_weights, confusion_weights=None
):
    """
    Loss function for custom problem type.
    Args:
        datapoint (ef.Datapoint): Datapoint object containting targets.

        model_output (ef.ModelOutput): ModelOutput object containing predictions.

        class_weights (dict[int,float]): Dict mapping class id to class weight. Class weights are normalized such that sum of weights is 1.

        field_weights (dict[str
        q,float]): Dict mapping ef.base_field._cls to weight. Field weights are normalized such that sum of weights is 1.

        confusion_weights (dict[int,tuple]): Dict mapping class id to confusion weights (tp, fp, fn). Confusion weights are normalized such that sum of weights is 1.

    Returns:
        loss (dict): A dict containing the failure score as well as score break down dicts showing
    """
    return datapoint_loss(
        datapoint,
        model_output,
        class_weights,
        field_weights,
        confusion_weights,
    )


def filter_loss(loss, mask_out_keys=None):
    if mask_out_keys is None:
        mask_out_keys = MASK_OUT_KEYS

    return {
        key: val
        for key, val in loss.items()
        if all(mask_key not in key for mask_key in mask_out_keys)
    }


def subtract_losses(loss_a, loss_b):
    difference_loss = {}

    for key in filter_loss(loss_a).keys():
        if key.endswith("_normalization_constant"):
            difference_loss[key] = max(loss_a[key], loss_b[key])
        else:
            difference_loss[key] = loss_a[key] - loss_b[key]

    return difference_loss


def normalize_loss(loss):
    normalized_loss = {}
    for key, value in loss.items():
        if key.endswith("_normalization_constant"):
            continue

        if key.endswith("_unnormalized"):
            key = key.replace("_unnormalized", "", 1)
            denominator = loss[key + "_normalization_constant"]
            value = value / denominator if denominator > 0 else 0

        normalized_loss[key] = value

    return normalized_loss


def aggregate_loss(loss):
    return sum(normalize_loss(filter_loss(loss)).values())
