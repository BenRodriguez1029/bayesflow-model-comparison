# losses.py
from keras import ops

"""m should always be 0 or 1"""


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


