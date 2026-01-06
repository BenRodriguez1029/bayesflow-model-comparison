from math import sqrt
import random
import numpy as np


def autoregress(r, rho, v):
    return (rho * r) + sqrt(1-rho**2) * v * random.gauss(0, 1)


def random_walk(n_steps):
    r_t = 0
    rho_t = random.uniform(-1, 1)
    v_t = random.uniform(0, 0.006)

    returns = []
    models = []

    for t in range(n_steps):
        if random.random() < 0.01:
            rho_t = random.uniform(-1, 1)
            v_t = random.uniform(0, 0.006)
            models.append(1)
        else:
            rho_t = np.clip(rho_t + random.gauss(0, 0.05), -1, 1)
            v_t = np.clip(v_t + random.gauss(0, 0.0001), 0, 0.006)
            models.append(0)

        r_t = autoregress(r_t, rho_t, v_t)
        returns.append(r_t)

    return returns, models
