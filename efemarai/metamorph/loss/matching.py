import math

import numpy as np
from scipy import spatial


def greedy_entity_matching(IoUs):
    """Matches predictions to ground truth classes in a greedy fashion.

    Ground truth and prediction entities are matched together if they have
    the largest IoU amongst all non-matched, but overlapping entities, and also
    have the same class label. This is done iteratively until no other non-matched
    overlapping pair of entities exists.

    Args:
        IoUs (np.ndarray): Pairwise IoUs between ground truth and prediction entities
            with shape [number of ground truth entities, number of predicted entities].

    Returns:
        A tuple of (best_gt, best_pred, best_iou) which are dicts mapping
        every matched prediction box index to the index of its ground truth box
        (best_gt) and the other way round - each ground truth box index to its
        prediction box index (best_pred). best_iou is a dict mapping a frozenset
        of {gt_index, pred_index} to the IoU of these two matched entities.
    """
    # Avoid modifying input array
    IoUs = IoUs.copy()

    best_pred = {}
    best_gt = {}
    best_iou = {}

    while IoUs.size > 0 and IoUs.max() > 0:
        indices = np.where(IoUs == IoUs.max())

        gt_index = indices[0][0]
        pred_index = indices[1][0]

        best_gt[pred_index] = gt_index
        best_pred[gt_index] = pred_index

        best_iou[(gt_index, pred_index)] = IoUs[gt_index, pred_index].item()
        IoUs[gt_index] = 0
        IoUs[:, pred_index] = 0

    return best_gt, best_pred, best_iou


def calculate_pairwise_distances(gt, pred):
    """Matches ground truth keypoint group to prediction keypoint group based on minimal distance.

    Computes distance between groups of keypoints based on pairwise distances computed between
    keypoints inside groups. The final distances are represented as a mean of all distances inside the
    group. The matched groups are the ones which have the least distance between them.

    Args:
        gt (np.ndarray): Ground truth keypoints with shape [number of keypoint groups,
            number of keypoint vertices, 3], where the last dimension is [x, y, (z), v].
            x & y & z (optional) are bound in [0, 1], v: 0-unlabeled, 1-invisible, 2-visible.
        pred (np.ndarray): Prediction keypoints with shape [number of keypoint groups,
            number of keypoint vertices, 3], where the last dimension is [x, y, (z), c].
            x & y & z (optional) & c are bound in [0, 1].

    Returns:
        distances (np.ndarray): Pairwise distances between ground truth and prediction entities
            with shape [number of ground truth entities, number of predicted entities].
    """
    distances = np.empty(shape=[gt.shape[0], pred.shape[0]], dtype=float)

    for gt_group_idx, gt_group in enumerate(gt):
        for pred_group_idx, pred_group in enumerate(pred):

            group_distances = []

            # For each GT keypoint, calculate the pairwise (idx to idx) distance to the PRED keypoint
            for keypoint_gt, keypoint_pred in zip(gt_group, pred_group):

                if keypoint_pred[-1] == 0:
                    # If no prediction and GT is 0 -> correct
                    if keypoint_gt[-1] == 0:
                        group_distances.append(0)
                        continue

                # TODO: Export this as an option in the UI -> should the loss be calculated on hidden?
                elif keypoint_gt[-1] == 2 or keypoint_gt[-1] == 1:
                    # Calculate distance between visible and predicted
                    group_distances.append(
                        spatial.distance.euclidean(keypoint_gt[:-1], keypoint_pred[:-1])
                        / math.sqrt(
                            len(keypoint_gt[:-1])
                        )  # normalize to a max distance of 1
                    )
                    continue

                # Add max error for FP and FN.
                group_distances.append(1)

            distances[gt_group_idx, pred_group_idx] = np.array(group_distances).mean()

    return distances
