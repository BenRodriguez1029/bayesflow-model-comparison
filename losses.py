# losses.py
from keras import ops

"""m should always be 0 or 1"""

def squared_loss(y_true, y_pred):
    return ops.mean(ops.square(y_pred - y_true), axis=-1)

def polynomial_loss(y_true, y_pred, alpha=2.0):
    return ops.mean(y_true * ops.power(1.0 - y_pred, alpha) +
                    (1.0 - y_true) * ops.power(y_pred, alpha), axis=-1)

def exponential_loss(y_true, y_pred):
    return ops.mean(ops.exp((0.5 - y_true) * y_pred), axis=-1)

def logistic_loss(y_true, y_pred):
    return ops.mean(ops.log(1.0 + ops.exp((1.0 - 2.0 * y_true) * y_pred)), axis=-1)

def alpha_exponential_loss(y_true, y_pred, alpha=2.0):
    return ops.mean(ops.power(1.0 + ops.exp((1.0 - 2.0 * y_true) * y_pred), alpha), axis=-1)

def alpha_log_exponential_loss(y_true, y_pred, alpha=2.0):
    return ops.mean(y_pred * ops.power(0.5 - y_true, alpha), axis=-1)

# l-POP transform
def J_alpha(x, alpha=2.0):
    return x + x * ops.power(ops.abs(x), alpha - 1)

def lpop_exponential_loss(y_true, y_pred, alpha=2.0):
    return ops.mean(ops.exp((0.5 - y_true) * J_alpha(y_pred, alpha)), axis=-1)