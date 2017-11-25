import numpy as np
from numpy import arange
from scipy.sparse import dia_matrix, csc_matrix
from scipy.linalg import expm


def get_z_projections(n):
    return [m for m in arange(-n / 2, (n + 1) / 2)]

def j_z(n, shape = (0, 0)):
    m_list = get_z_projections(n)
    if shape[0] == 0: shape = (n+1, n+1)
    return dia_matrix((m_list, 0), shape=shape)

def j_plus_values(n):
    j = n / 2.
    m_list = get_z_projections(n)
    return [np.sqrt(j * (j + 1) - m * (m + 1)) for m in m_list]

def j_plus(n, shape = (0, 0)):
    if shape[0] == 0: shape = (n+1, n+1)
    return dia_matrix((j_plus_values(n), -1), shape=shape)

def j_minus_values(n):
    j = n / 2.
    m_list = get_z_projections(n)
    return [np.sqrt(j * (j + 1) - m * (m - 1)) for m in m_list]

def j_minus(n, shape = (0, 0)):
    if shape[0] == 0: shape = (n+1, n+1)
    return dia_matrix((j_minus_values(n), 1), shape=shape)

def j_x(n, shape = (0, 0)):
    if shape[0] == 0: shape = (n + 1, n + 1)
    return (j_plus(n, shape) + j_minus(n, shape)) / 2.

def j_y(n, shape = (0, 0)):
    if shape[0] == 0: shape = (n + 1, n + 1)
    return (j_plus(n, shape) - j_minus(n, shape)) / 2. * 1j


def mean(state, j = j_z):
    n = state.shape[0] - 1
    out = state.conj().dot(j(n).dot(state))
    return out.real


def means(state):
    return [mean(state, j_x), mean(state, j_y), mean(state, j_z)]


def MSD(state, for_bloch=False):
    if for_bloch:
        m = means_bloch(state)
    else:
        m = means(state)
    return m/np.linalg.norm(m)


def means_bloch(state):
    m_x, m_y, m_z = means(state)
    return [m_y, m_x, m_z]


def rotation(state, angle, j):
    n = state.shape[0] - 1
    return expm(-1j * angle * csc_matrix(j(n))).dot(state)


def pi_pulse_x(state):
    return rotation(state, np.pi / 2, j_x)


def pi_pulse_y(state):
    return rotation(state, np.pi / 2, j_y)


def css_ground(n):
    state = np.zeros(n + 1)
    state[0] = 1.
    return state