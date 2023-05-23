import numpy as np


def preprocess_tags(targets, outputs):
    gt_class, pred_confidence = [], []

    for target in targets:
        target_cls = (
            target.label.id
            if hasattr(target, "label") and hasattr(target.label, "id")
            else None
        )
        gt_class.append(target_cls)

    for output in outputs:
        if output.probabilities is not None:
            probabilities = output.probabilities

        pred_confidence.append(probabilities)

    return (
        np.array(gt_class),
        np.array(pred_confidence),
    )


def one_hot_encode(ids, num_classes=None):
    if isinstance(ids, list):
        ids = np.asarray(ids)

    if num_classes is None:
        max_id = ids.max() + 1
    else:
        max_id = num_classes

    one_hot = np.zeros((ids.size, max_id), dtype=int)
    rows = np.arange(ids.size)
    one_hot[rows, ids] = 1
    return one_hot


def classification_loss(gt_classes, pred_confidence, class_weights):

    loss = {
        "softmax_info": pred_confidence
        if not hasattr(pred_confidence, "tolist")
        else pred_confidence.tolist(),
        "failure_score_normalization_constant": 1,
    }
    one_hot = one_hot_encode(gt_classes, num_classes=len(class_weights))

    weights = np.zeros(len(class_weights))
    for index, weight in class_weights.items():
        weights[index] = weight

    conf_diff = np.abs(one_hot - pred_confidence) * weights

    loss["failure_score_unnormalized"] = conf_diff.sum().item()

    return loss


def cross_entropy_loss(gt_classes, pred_confidence, class_weights):

    loss = {
        "softmax_info": pred_confidence
        if not hasattr(pred_confidence, "tolist")
        else pred_confidence.tolist(),
        "failure_score_normalization_constant": 1,
    }
    one_hot = one_hot_encode(gt_classes, num_classes=len(class_weights))

    cr_e = np.log(pred_confidence) * one_hot
    cr_e = cr_e.sum().item() * -1
    loss["failure_score_unnormalized"] = cr_e

    return loss
