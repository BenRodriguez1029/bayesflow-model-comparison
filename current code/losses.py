# losses.py
from keras import ops


def cross_entropy_loss(y_true, y_pred):
    return -y_true * ops.log(y_pred + 1e-7) - (1.0 - y_true) * ops.log(1.0 - y_pred + 1e-7)


def squared_loss(y_true, y_pred):
    return ops.square(y_pred - y_true)


def polynomial_loss(y_true, y_pred, alpha=2.0):
    return y_true * ops.power(1.0 - y_pred, alpha) + (1.0 - y_true) * ops.power(y_pred, alpha)


def exponential_loss(y_true, y_pred):
    return ops.exp((0.5 - y_true) * y_pred)


def logistic_loss(y_true, y_pred):
    return ops.log(1.0 + ops.exp((1.0 - 2.0 * y_true) * y_pred))


def alpha_exponential_loss(y_true, y_pred, alpha=2.0):
    return ops.power(1.0 + ops.exp((1.0 - 2.0 * y_true) * y_pred), alpha)


def alpha_log_exponential_loss(y_true, y_pred, alpha=2.0):
    return ops.power(y_pred, (0.5 - y_true) * alpha)

# l-POP transform


def J_alpha(x, alpha=2.0):
    return x + x * ops.power(ops.abs(x), alpha - 1)


def lpop_exponential_loss(y_true, y_pred, alpha=2.0):
    return ops.exp((0.5 - y_true) * J_alpha(y_pred, alpha))


def multimodel_exponential_loss(y_true, f_x):
    target_index = ops.cast(y_true[:, 0], "int32")  # get target model
    batch_size = ops.shape(f_x)[0]

    # prepend f0 = 0
    f0 = ops.zeros((batch_size, 1), dtype=f_x.dtype)
    fx_full = ops.concatenate([f0, f_x], axis=1)

    # select f_m for each sample
    f_m = fx_full[ops.arange(batch_size), target_index]
    diffs = fx_full - ops.expand_dims(f_m, axis=1)

    mask = ops.ones_like(diffs) * \
        (1.0 - ops.one_hot(target_index, ops.shape(diffs)[1]))
    diffs_masked = diffs * mask

    losses = ops.exp(-0.5 * diffs_masked)
    return ops.mean(ops.sum(losses, axis=1))  # mean over batch
