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


def multimodel_accuracy(y_true, f_x):
    batch_size = ops.shape(f_x)[0]

    # Reconstruct full f vector including f0=0
    f0 = ops.zeros((batch_size, 1), dtype=f_x.dtype)
    fx_full = ops.concatenate([f0, f_x], axis=1)  # shape (batch_size, N+1)

    # Predicted model is argmax of logK
    preds = ops.argmin(fx_full, axis=1)

    # Compute accuracy
    correct = ops.equal(preds, ops.cast(y_true[:, 0], "int32"))
    return ops.mean(ops.cast(correct, f_x.dtype))
