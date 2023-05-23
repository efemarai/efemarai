def preprocess_text(targets, outputs):
    return (
        [target.text for target in targets],
        [output.text for output in outputs],
    )


def text_loss(metric, targets, preds):
    # TODO: Take up all targets and preds. Option: ef.Text.text is a list of sentences
    targets = [targets[0]]
    score = metric(targets, preds)
    # {"f1":float, "recall":float, "precision":float}
    score = score["f1"] if isinstance(score["f1"], float) else score["f1"].mean()
    loss = {
        "failure_score_unnormalized": 1 - score,
        "failure_score_normalization_constant": 1,
    }
    return loss
