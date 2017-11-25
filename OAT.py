from scipy.linalg import expm
from scipy.sparse import csc_matrix

from angular_momentum import j_x, css_ground


def hamiltonian_oat(n, sqeezing_angle = 0):
    return sqeezing_angle / 2 * (j_x(n) * j_x(n))


def unitary_oat(n, sqeezing_angle = 0):
    m = -1j * hamiltonian_oat(n, sqeezing_angle=sqeezing_angle)
    m = csc_matrix(m)
    return expm(m)


def oat_state(n, squeezing_angle = 0):
    return unitary_oat(n, sqeezing_angle=squeezing_angle).dot(css_ground(n))