import numpy as np
from scipy import arange

# unique quantum probes


def gauss_function(x, sigma = 1., mu = 0):
    return 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-1 / 2. * (x - mu) ** 2 / sigma ** 2)


def gauss(n, sigma = 1., mu = 0):
    return [gauss_function(x, sigma, mu) for x in arange(-n/2., n/2.+1)]


def individual_all_decoherence_state(n, gamma = 0, t = 0):
    sigma = 1 / 2 / (n * (np.exp(gamma * t) - 1)) ** (1. / 4)
    return gauss(n, sigma)