import numpy as np


def regression_loss(targets, preds):
    from scipy import spatial

    u = [ef_field.value for ef_field in targets]
    v = [ef_field.value for ef_field in preds]

    # TODO: Add class weights
    dist = spatial.distance.euclidean(u, v)
    norm = np.sqrt(len(u))

    loss = {
        "failure_score_unnormalized": dist,
        "failure_score_normalization_constant": norm,
    }
    return loss
