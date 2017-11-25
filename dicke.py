import numpy as np
from scipy.misc import factorial
from scipy.sparse import csr_matrix

from angular_momentum import j_plus, css_ground


def dicke_state(j, m):
    a = np.sqrt(factorial(j - m) / factorial(j + m) / factorial(2 * j))
    b = csr_matrix(j_plus(2 * j)).__pow__(j + m)
    state = a * b
    return state.dot(css_ground(int(2 * j)))