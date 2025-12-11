from math import sqrt
import random

def epsilon():
    return random.gauss(0, 1)

def autoregress(r, rho, v):
    return (rho * r) + sqrt(1-rho**2) * v * epsilon() 

def simulate_ar_series(r_0, rho_dist, v_dist, n, fixed=False):
    if fixed:
        rho = rho_dist()
        v = v_dist()

    series = [[r_0]]
    for t in range(1, n):
        if not fixed:
            rho = rho_dist()
            v = v_dist()
        r_t = autoregress(series[t-1][0], rho, v)
        series.append([r_t])

    return series