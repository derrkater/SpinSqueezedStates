import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.colors

from angular_momentum import *
from OAT import *
from dicke import dicke_state
from durkin import *
from qutip import wigner, coherent, coherent_dm
from scipy.linalg import expm
import qutip


def optimal_theta_review(n):
    return 12 ** (1 / 6) * (n / 2.) ** (-2 / 3)

def optimal_theta(n):
    return 2 * np.e / (1 - 2 * np.e + 2 * np.e ** 2 - n + np.e * n)


def plot_1_blochball(add_helper_lines = True):

    bloch = qutip.Bloch()

    theta = np.pi / 3
    phi = - np.pi / 4

    vec = [np.exp(1j * phi) * np.cos(theta / 2), np.sin(theta / 2)]
    qubit = vec[0] * qutip.basis(2, 0) + vec[1] * qutip.basis(2, 1)

    bloch.add_states(qubit)

    if add_helper_lines:
        fig = plt.figure(figsize=(7.4,7))
        ax = fig.add_subplot(111, projection='3d')

        bloch.axes = ax
        bloch.fig = fig

        bloch_vec = bloch.vectors[0].copy()
        bloch_vec[1] = -bloch_vec[1]
        point_below = bloch_vec.copy()
        point_below[2] = 0
        help_line_1 = [[p1, p2] for p1, p2 in zip(bloch_vec, point_below)]
        help_line_2 = [[0, p] for p in bloch_vec]
        help_line_3 = [[0, p] for p in point_below]
        bloch.axes.plot(help_line_1[0], help_line_1[1], help_line_1[2])
        bloch.axes.plot(help_line_2[0], help_line_2[1], help_line_2[2])
        bloch.axes.plot(help_line_3[0], help_line_3[1], help_line_3[2])
    else:
        bloch.view = [-75, 22]

    bloch.save("/Users/derrkater/Desktop/mgr/mgr/img/plot1bloch")

def plot_states(n = 20):
    xvec = np.linspace(-3., 3., 200)

    coherent_state = qutip.Qobj(oat_state(n, 0))
    # squeezed_state_1 = qutip.Qobj(oat_state(n, optimal_theta(n)))
    # squeezed_state_2 = qutip.Qobj(oat_state(n, optimal_theta_review(n)))

    wigner_coherent = wigner(coherent_state, xvec, xvec)
    # wigner_squeezed_1 = wigner(squeezed_state_1, xvec, xvec)
    # wigner_squeezed_2 = wigner(squeezed_state_2, xvec, xvec)

    fig, axes = plt.subplots(1, 1, figsize =(7,7))
    axes.contourf(xvec, xvec, wigner_coherent, 100, cmap=cm.Purples)
    # axes.set_title("Coherent state")
    # axes[1].contourf(xvec, xvec, wigner_squeezed_1, 100)
    # axes[1].set_title("Squeezed state")
    # axes[2].contourf(xvec, xvec, wigner_squeezed_2, 100)
    # axes[2].set_title("Squeezed state")

    plt.show()




if __name__ == '__main__':

    plot_states()

    # plot_1_blochball(False)

    # plot_bloch_evolution()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.plot([0,1], [0,1], [0,1])
    # plt.show()





