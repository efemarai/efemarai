from collections import defaultdict

import numpy as np

from efemarai.metamorph.loss.matching import greedy_entity_matching

DEFAULT_BBOX_CONFIDENCE = 1.0


def preprocess_bbox(targets, outputs):
    gt_object, gt_class, pred_object, pred_class, pred_confidence = [], [], [], [], []

    for target in targets:
        target_obj = target.xyxy
        target_cls = (
            target.label.id
            if hasattr(target, "label") and hasattr(target.label, "id")
            else None
        )
        gt_object.append(target_obj)
        gt_class.append(target_cls)

    for output in outputs:
        output_obj = output.xyxy
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
            output_conf if output_conf is not None else DEFAULT_BBOX_CONFIDENCE
        )

    return (
        np.array(gt_object),
        np.array(gt_class),
        np.array(pred_object),
        np.array(pred_class),
        np.array(pred_confidence),
    )


class BoxInfo:
    """Holds all info about a bounding box.

    Attributes:
        box (np.ndarray): Array holding box coordinates (xmin, ymin, xmax, ymax).
        label (int): Index of the box class.
        confidence (Optional[float]): Confidence score of the bounding box.
    """

    def __init__(self, box, label, confidence=None):
        self.box = box
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        """Returns string  representation of the bounding box.

        The format of the returned string is
            "[(xmin, ymin), (xmax, ymax)], class-index, confidence"

        where 'confidence' is included only if available.
        """
        xmin, ymin, xmax, ymax = [round(c) for c in self.box.tolist()]
        res = "{"

        res += f"({xmin}, {ymin}), ({xmax}, {ymax}), {int(self.label)}"

        if self.confidence is not None:
            res += f", {self.confidence:.4f}"

        res += "}"

        return res


def box_iou(box1, box2):
    """Computes the intersection over union of two set of boxes.

    Each box needs to be represented as (xmin, ymin, xmax, ymax).

    Args:
        box1 (np.ndarray): bounding boxes, sized [N, 4].
        box2 (np.ndarray): bounding boxes, sized [M, 4].

    Return:
        np.ndarray: iou, sized [N, M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
      https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    """
    N = box1.shape[0]
    M = box2.shape[0]

    if N == 0 or M == 0:
        return np.empty((N, M))

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def bounding_box_loss(
    gt_boxes,
    gt_classes,
    pred_boxes,
    pred_classes,
    pred_confidence,
    class_weights,
    confusion_weights=None,
):
    """Calculates the failure score for a set of ground truth and prediction boxes.

    The score is bounded in [0; 1] with 0 indicating that the predictions are good.

    Args:
        gt_boxes (np.array): Ground truth boxes with shape [Nx4]. Each box is
            represented as (xmin, ymin, xmax, ymax).

        gt_classes (np.array): Labels of the ground truth boxes with shape [N].

        pred_boxes (np.array): Prediction boxes with shape [Mx4]. Each box is
            represented as (xmin, ymin, xmax, ymax).

        pred_classes (np.array): Labels of the prediction boxes with shape [N].

        pred_confidence (np.array): Predictions confidence scores with shape [N].

        class_weights (dict[int,float]): Dict mapping class id to class weight. Class weights are normalized such that sum of weights is 1.

        confusion_weights (dict[int,tuple]): Dict mapping class id to confusion weights (tp, fp, fn). Confusion weights are normalized such that sum of weights is 1.


    Returns:
        A dict containing the failure score as well as score break down dicts showing
        how individual boxes have been scored.
    """
    np.testing.assert_almost_equal(sum(class_weights.values()), 1)

    if confusion_weights is None:
        confusion_weights = defaultdict(lambda: (1 / 3, 1 / 3, 1 / 3))

    # Ground Truth x Predictions
    IoUs = box_iou(gt_boxes, pred_boxes)

    # Ignore IoUs of boxes with different classes
    IoUs[~np.equal.outer(gt_classes, pred_classes)] = 0

    # Multiply each column with confidence score of respective prediction
    IoUs *= pred_confidence

    best_gt, best_pred, best_iou = greedy_entity_matching(IoUs)

    gt_boxes_info = [BoxInfo(box, label) for box, label in zip(gt_boxes, gt_classes)]

    pred_boxes_info = [
        BoxInfo(box, label, confidence)
        for box, label, confidence in zip(pred_boxes, pred_classes, pred_confidence)
    ]

    gt_scores = {
        (box_info, repr(box_info)): (
            best_iou[(i, best_pred[i])],
            class_weights[box_info.label]
            * confusion_weights[box_info.label][0],  # 0 -> tp
        )
        for i, box_info in enumerate(gt_boxes_info)
        if i in best_pred
    }

    pred_scores = {
        (box_info, repr(box_info)): (
            best_iou[(best_gt[i], i)],
            class_weights[box_info.label]
            * confusion_weights[box_info.label][0],  # 0 -> tp
        )
        for i, box_info in enumerate(pred_boxes_info)
        if i in best_gt
    }

    fp_scores = {
        repr(box_info): (
            -1.0 * confidence,
            class_weights[box_info.label]
            * confusion_weights[box_info.label][1],  # 1 -> fp
        )
        for i, (box_info, confidence) in enumerate(
            zip(pred_boxes_info, pred_confidence)
        )
        if i not in best_gt
    }

    fn_scores = {
        repr(box_info): (
            -1.0,
            class_weights[box_info.label]
            * confusion_weights[box_info.label][2],  # 2 -> fn
        )
        for i, box_info in enumerate(gt_boxes_info)
        if i not in best_pred
    }

    weighted_scores = []
    weighted_scores += list(gt_scores.values())
    weighted_scores += list(pred_scores.values())
    weighted_scores += list(fp_scores.values())
    weighted_scores += list(fn_scores.values())

    loss = {
        "pred_boxes_info": [repr(box_info) for box_info in pred_boxes_info],
        "gt_boxes_info": [repr(box_info) for box_info in gt_boxes_info],
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
