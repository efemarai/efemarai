from collections import defaultdict

import numpy as np

from efemarai.metamorph.loss.matching import greedy_entity_matching

DEFAULT_SEGM_CONFIDENCE = 1.0


def preprocess_polygon(targets, outputs):
    gt_object, gt_class, pred_object, pred_class, pred_confidence = [], [], [], [], []

    for target in targets:
        target_obj = target._raw_data
        target_cls = (
            target.label.id
            if hasattr(target, "label") and hasattr(target.label, "id")
            else None
        )
        gt_object.append(target_obj)
        gt_class.append(target_cls)

    for output in outputs:
        output_obj = output._raw_data
        output_cls = (
            output.label.id
            if hasattr(output, "label") and hasattr(output.label, "id")
            else None
        )
        output_conf = (
            output.confidence
            if not hasattr(output, "label") or not hasattr(output.label, "confidence")
            else output.label.confidence
        )
        pred_object.append(output_obj)
        pred_class.append(output_cls)
        pred_confidence.append(
            output_conf if output_conf is not None else DEFAULT_SEGM_CONFIDENCE
        )

    return (
        np.array(gt_object),
        np.array(gt_class),
        np.array(pred_object),
        np.array(pred_class),
        np.array(pred_confidence),
    )


class MaskInfo:
    """Holds all info about a mask.

    Attributes:
        mask (np.ndarray): Array holding mask information (mask).
        label (int): Index of the mask class.
        confidence (Optional[float]): Confidence score of the mask.
    """

    def __init__(self, mask, label, confidence=None):
        self.mask = mask
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        """Returns string representation of the bounding box inscribing the mask.

        The format of the returned string is
            "[x1, y1, x2, y2], class-index, confidence"

        where 'confidence' is included only if available.
        """
        # TODO: Add option to return string representation of the polygons
        coords = np.argwhere(np.asarray(self.mask))

        res = "{"

        if coords.sum() > 0:
            top, left = coords.min(axis=0)
            bottom, right = coords.max(axis=0)
            res += f"({top} {left} {bottom} {right}), {int(self.label)}"

        else:
            res += f"(0 0 0 0), {int(self.label)}"

        if self.confidence is not None:
            res += f", {self.confidence:.4f}"

        res += "}"

        return res


def mask_loss(
    gt_masks,
    gt_classes,
    pred_masks,
    pred_classes,
    pred_confidence,
    class_weights,
    confusion_weights=None,
):
    """Calculates the failure score for a set of ground truth and prediction masks.

    The score is bounded in [0; 1] with 0 indicating that the predictions are good.

    Args:
        gt_masks (np.array): Ground truth PIL Image with shape [NxM] when loaded where N and M are the width and height of the mask.

        gt_classes (np.array): Labels of the ground truth PIL Image with shape [K] when loaded.

        pred_masks (np.array): Prediction PIL Image with shape [NxM] when loaded where N and M are the width and height of the mask.

        pred_classes (np.array): Labels of the prediction PIL Image with shape [N] when loaded.

        pred_confidence (np.array): Predictions confidence scores with shape [N].

        class_weights (dict[int,float]): Dict mapping class id to class weight. Class weights are normalized such that sum of weights is 1.

        confusion_weights (dict[int,tuple]): Dict mapping class id to confusion weights (tp, fp, fn). Confusion weights are normalized such that sum of weights is 1.

    Returns:
        A dict containing the failure score as well as score break down dicts showing
        how individual masks have been scored.
    """
    np.testing.assert_almost_equal(sum(class_weights.values()), 1)

    if confusion_weights is None:
        confusion_weights = defaultdict(lambda: (1 / 3, 1 / 3, 1 / 3))

    from pycocotools import mask as cocomask

    threshold = 0.5

    gt_m = [cocomask.encode(np.asfortranarray(m > threshold)) for m in gt_masks]
    pred_m = [cocomask.encode(np.asfortranarray(m > threshold)) for m in pred_masks]

    if len(gt_masks) == 0 or len(pred_masks) == 0:
        pairwise_iou = np.zeros((len(gt_masks), len(pred_masks)))

    else:
        pairwise_iou = np.asarray(cocomask.iou(gt_m, pred_m, [0] * len(pred_m)))
        # Ignore IoUs of polygons with different classes
        pairwise_iou[~np.equal.outer(gt_classes, pred_classes)] = 0

    # Multiply each column with confidence score of respective prediction
    pairwise_iou *= pred_confidence

    best_gt, best_pred, best_iou = greedy_entity_matching(pairwise_iou)

    gt_masks_info = [MaskInfo(mask, label) for mask, label in zip(gt_masks, gt_classes)]

    pred_masks_info = [
        MaskInfo(mask, label, confidence)
        for mask, label, confidence in zip(pred_masks, pred_classes, pred_confidence)
    ]

    gt_scores = {
        (polygon_info, repr(polygon_info)): (
            best_iou[(i, best_pred[i])],
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][0],  # 0 -> tp
        )
        for i, polygon_info in enumerate(gt_masks_info)
        if i in best_pred
    }

    pred_scores = {
        (polygon_info, repr(polygon_info)): (
            best_iou[(best_gt[i], i)],
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][0],  # 0 -> tp
        )
        for i, polygon_info in enumerate(pred_masks_info)
        if i in best_gt
    }

    fp_scores = {
        repr(polygon_info): (
            -1.0 * confidence,
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][1],  # 1 -> fp
        )
        for i, (polygon_info, confidence) in enumerate(
            zip(pred_masks_info, pred_confidence)
        )
        if i not in best_gt
    }

    fn_scores = {
        repr(polygon_info): (
            -1.0,
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][2],  # 2 -> fn
        )
        for i, polygon_info in enumerate(gt_masks_info)
        if i not in best_pred
    }

    weighted_scores = []
    weighted_scores += list(gt_scores.values())
    weighted_scores += list(pred_scores.values())
    weighted_scores += list(fp_scores.values())
    weighted_scores += list(fn_scores.values())

    loss = {
        "pred_masks_info": [repr(polygon_info) for polygon_info in pred_masks_info],
        "gt_masks_info": [repr(polygon_info) for polygon_info in gt_masks_info],
        "gt_scores_info": {k[1]: v[0] for k, v in gt_scores.items()},
        "pred_scores_info": {k[1]: v[0] for k, v in pred_scores.items()},
        "fp_scores_info": {k[1]: v[0] for k, v in fp_scores.items()},
        "fn_scores_info": {k[1]: v[0] for k, v in fn_scores.items()},
        "failure_score_unnormalized": sum(
            (1 - (s + 1) / 2) * w for (s, w) in weighted_scores
        ),
        "failure_score_normalization_constant": sum(w for (_, w) in weighted_scores),
    }
    return loss
