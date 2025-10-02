import keras.ops as ops


def logk_accuracy(y_true, f_x):
    p = ops.sigmoid(f_x)
    preds = ops.cast(p > 0.5, "float32")
    return ops.mean(ops.cast(ops.equal(preds, ops.cast(y_true, "float32")), "float32"))


def k_accuracy(y_true, y_pred, alpha=2.0):
    y_true = ops.cast(y_true, dtype='float32')
    pred_labels = ops.cast(ops.greater(y_pred, 1), dtype='float32')
    correct = ops.equal(pred_labels, y_true)
    return ops.mean(ops.cast(correct, dtype='float32'), axis=-1)
