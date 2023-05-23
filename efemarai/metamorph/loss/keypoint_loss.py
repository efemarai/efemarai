from collections import defaultdict

import numpy as np

from efemarai.metamorph.loss.matching import (
    calculate_pairwise_distances,
    greedy_entity_matching,
)

DEFAULT_KEYPOINT_CONFIDENCE = 1.0
DEFAULT_SKELETON_CONFIDENCE = 1.0


def preprocess_keypoints(targets, outputs):
    """
    Extract the needed data from ef.Field objects.
    Args:
        targets (dict): Ground truth serialized ef.BaseFields

        outputs (dict): Predicted serialized ef.BaseFields
    Return:
        gt_instances (np.array): Ground truth instances.

        gt_classes (np.array): Labels of the ground truth instances with shape [K].

        pred_instances (np.array): Prediction instances.

        pred_classes (np.array): Labels of the prediction instances with shape [N].

        pred_confidence (np.array): Predictions confidence scores with shape [N].

    """
    gt_object, pred_object, pred_confidence = [], [], []
    target_ids = {target.instance_id for target in targets}
    for skeleton_id in target_ids:
        gt_object.append(
            [
                (target.x, target.y, float(target.occluded))
                for target in targets
                if target.instance_id == skeleton_id
            ]
        )

    output_ids = {output.instance_id for output in outputs}
    for skeleton_id in output_ids:
        pred_object.append(
            [
                (output.x, output.y, float(output.occluded))
                for output in outputs
                if output.instance_id == skeleton_id
            ]
        )
        # TODO: Think of how to aggregate the keypoint score per group
        pred_confidence.append(
            # np.array(
            #     [
            #         (
            #             output.confidence
            #             if not hasattr(output, "label")
            #             or not hasattr(output.label, "confidence")
            #             else output.label.confidence
            #         )
            #         for output in outputs
            #         if output.instance_id == skeleton_id
            #     ]
            # ).mean()
            # np.random.rand()
            DEFAULT_KEYPOINT_CONFIDENCE
        )

    return (
        np.array(gt_object),
        np.array(list(target_ids)),
        np.array(pred_object),
        np.array(list(output_ids)),
        np.array(pred_confidence),
    )


def preprocess_skeleton(targets, outputs):
    """
    Extract the needed data from ef.Field objects.
    Args:
        targets (dict): Ground truth serialized ef.BaseFields

        outputs (dict): Predicted serialized ef.BaseFields
    Return:
        gt_instances (np.array): Ground truth instances.

        gt_classes (np.array): Labels of the ground truth instances with shape [K].

        pred_instances (np.array): Prediction instances.

        pred_classes (np.array): Labels of the prediction instances with shape [N].

        pred_confidence (np.array): Predictions confidence scores with shape [N].

    """
    gt_object, pred_object, pred_confidence = [], [], []
    target_ids = {target.instance_id for target in targets}
    for skeleton in targets:
        gt_object.append(
            [
                (keypoint_field.x, keypoint_field.y, float(keypoint_field.occluded))
                for keypoint_field in skeleton.keypoints
                if keypoint_field.instance_id == skeleton.instance_id
            ]
        )

    output_ids = {output.instance_id for output in outputs}
    for skeleton in outputs:
        pred_object.append(
            [
                (keypoint_field.x, keypoint_field.y, float(keypoint_field.occluded))
                for keypoint_field in skeleton.keypoints
                if keypoint_field.instance_id == skeleton.instance_id
            ]
        )
        # TODO: Think of how to aggregate the keypoint score per group
        pred_confidence.append(DEFAULT_SKELETON_CONFIDENCE)

    return (
        np.array(gt_object),
        np.array(list(target_ids)),
        np.array(pred_object),
        np.array(list(output_ids)),
        np.array(pred_confidence),
    )


class KeypointInfo:
    """Holds all info about a keypoint.

    Attributes:
        keypoint (np.ndarray): Array holding keypoint information (keypoint).
        label (int): Index of the keypoint class.
        confidence (Optional[float]): Confidence score of the keypoint.
    """

    def __init__(self, keypoint, label, confidence=None):
        self.keypoint = keypoint
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        """Returns string representation of the bounding box inscribing the keypoint.

        The format of the returned string is
            "[x, y, (z)], class-index, confidence"

        where 'confidence' is included only if available.
        """
        res = "{"
        res += f"({self.keypoint}"
        res += f") {int(self.label)}"

        if self.confidence is not None:
            res += f", {self.confidence:.4f}"

        res += "}"

        return res


def scale_coords_to_zero_one(keypoints, image_size):
    if keypoints.ndim == 3 and (
        keypoints[:, :, 0].max() > 1 or keypoints[:, :, 1].max() > 1
    ):
        keypoints[:, :, 0] /= image_size[0]
        keypoints[:, :, 1] /= image_size[1]
    return keypoints


def filter_empty_groups(keypoints):
    """Returns the keypoints with filtered out zero groups.

    A zero group is a group which consists of only [0,0,0] elements.

    Args:
        keypoints (np.array): Keypoints with shape [n_groups, n_elements, N] where
    N == 3 when the image is 2d [x, y, visibility]
    N == 4 when the image is 3d [x, y, z, visibility]

    Returns:
        A numpy ndarray with filtered out empty groups.

    """
    return keypoints[np.unique(keypoints.nonzero()[0])]


def keypoint_loss(
    gt_keypoints,
    gt_classes,
    pred_keypoints,
    pred_classes,
    pred_confidence,
    class_weights,
    confusion_weights=None,
):
    """Calculates the failure score for a set of ground truth and prediction keypoints.

    The score is bounded in [0; 1] with 0 indicating that the predictions are good.

    Args:
        gt_keypoints (np.array): Ground truth keypoints with shape [NxMxP] where N and M are the groups and keypoints in group.
    P can be either 3 or 4, for 2d images and 3d images respectively. P is [x, y, (z), visibility].

        gt_classes (np.array): Labels of the ground truth keypoints groups with shape [K].

        pred_keypoints (np.array): Prediction keypoints with shape [NxMxP] where N and M are the groups and keypoints in group.
    P can be either 3 or 4, for 2d images and 3d images respectively. P is [x, y, (z), confidence].

        pred_classes (np.array): Labels of the prediction keypoint groups with shape [N].

        pred_confidence (np.array): Predictions confidence scores with shape [N].

        class_weights (dict[int,float]): Dict mapping class id to class weight. Class weights are normalized such that sum of weights is 1.

        confusion_weights (dict[int,tuple]): Dict mapping class id to confusion weights (tp, fp, fn). Confusion weights are normalized such that sum of weights is 1.

    Returns:
        A dict containing the failure score as well as score break down dicts showing
        how individual keypoints have been scored.
    """
    np.testing.assert_almost_equal(sum(class_weights.values()), 1)

    gt_keypoints = filter_empty_groups(gt_keypoints)

    if confusion_weights is None:
        confusion_weights = defaultdict(lambda: (1 / 3, 1 / 3, 1 / 3))

    # calculate pairwise distance between groups of keypoints
    distances = calculate_pairwise_distances(gt_keypoints, pred_keypoints)

    # use how close the groups are
    closeness = 1 - distances

    # Multiply each column with confidence score of respective prediction
    closeness *= pred_confidence

    best_gt, best_pred, best_closeness = greedy_entity_matching(closeness)

    gt_keypoints_info = [
        KeypointInfo(keypoint, label)
        for keypoint, label in zip(gt_keypoints, gt_classes)
    ]

    pred_keypoints_info = [
        KeypointInfo(keypoint, label, confidence=confidence)
        for keypoint, label, confidence in zip(
            pred_keypoints, pred_classes, pred_confidence
        )
    ]

    gt_scores = {
        (polygon_info, repr(polygon_info)): (
            best_closeness[(i, best_pred[i])],
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][0],  # 0 -> tp
        )
        for i, polygon_info in enumerate(gt_keypoints_info)
        if i in best_pred
    }

    pred_scores = {
        (polygon_info, repr(polygon_info)): (
            best_closeness[(best_gt[i], i)],
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][0],  # 0 -> tp
        )
        for i, polygon_info in enumerate(pred_keypoints_info)
        if i in best_gt
    }

    fp_scores = {
        repr(polygon_info): (
            -1.0 * confidence,
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][1],  # 1 -> fp
        )
        for i, (polygon_info, confidence) in enumerate(
            zip(pred_keypoints_info, pred_confidence)
        )
        if i not in best_gt
    }

    fn_scores = {
        repr(polygon_info): (
            -1.0,
            class_weights[polygon_info.label]
            * confusion_weights[polygon_info.label][2],  # 2 -> fn
        )
        for i, polygon_info in enumerate(gt_keypoints_info)
        if i not in best_pred
    }

    weighted_scores = []
    weighted_scores += list(gt_scores.values())
    weighted_scores += list(pred_scores.values())
    weighted_scores += list(fp_scores.values())
    weighted_scores += list(fn_scores.values())

    loss = {
        "pred_keypoints_info": [
            repr(polygon_info) for polygon_info in pred_keypoints_info
        ],
        "gt_keypoints_info": [repr(polygon_info) for polygon_info in gt_keypoints_info],
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
