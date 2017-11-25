import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
import qutip
from scipy import linspace

from OAT import oat_state
from angular_momentum import pi_pulse_x, pi_pulse_y, MSD, rotation, j_z
from arrows_3d import Rx, Ry, Rz, Arrow3D


arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
PHI = .85 * np.pi


def Ry_bloch(arc):
    return Ry(-arc)


def get_spin_wigner(state, theta, phi):
    state_quantum_object = qutip.Qobj(state)
    wigner, _, _ = qutip.spin_wigner(state_quantum_object, theta, phi)
    wigner = wigner.real
    wigner /= wigner.max()
    return wigner


def get_spin_wigner_colormap(wigner, cmap = cm.Purples):
    # https://matplotlib.org/devdocs/gallery/images_contours_and_fields/image_transparency_blend.html

    # Create an alpha channel based on weight values
    # Any value whose absolute value is > .0001 will have zero transparency
    alphas = matplotlib.colors.Normalize(0, .3, clip=True)(np.abs(wigner))
    alphas = np.clip(alphas, .0, .7)  # alpha value clipped at the bottom at .0

    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
    colors = matplotlib.colors.Normalize(0, 1)(wigner)
    colors = cmap(colors)

    # Now set the alpha channel to the one we created above
    colors[..., -1] = alphas

    return colors

psi = 20 * np.pi / 180


def make_arc(alpha0, alpha1, r, plane = "zy", resolution = 50):
    # plane = xy, z-x (goes to -z), zy (string)
    arc = np.linspace(alpha0, alpha1, resolution)
    arc -= np.pi / 2 # stupid geometry compatibility issues
    p = np.array([np.cos(arc), np.sin(arc), arc * 0]) * r
    if plane == "xy":
        pass
    elif plane == "z-x":
        p = Ry(-np.pi / 2).dot(p)
        p = Rx(-np.pi / 2).dot(p)
    elif plane == "zy":
        p = Ry(-np.pi / 2).dot(p)
        p = Rx(-np.pi / 2).dot(p)
        p = Rz(-np.pi / 2).dot(p)
    # p = np.array([np.cos(arc), np.sin(arc), arc * 0]) * r
    return p


def plot_arc_on_bloch(arc, bloch):
    bloch.axes.plot(arc[0, :], arc[1, :], arc[2, :], 'k--')
    # o = arc[0]
    # v = [1, 0, 0]
    # t = [x + y for x, y in zip(o, v)]
    # arrow = Arrow3D([o[0], t[0]], [o[1], t[1]], [o[2], t[2]], **arrow_prop_dict)
    # bloch.axes.add_artist(arrow)


def plot_bloch_evolution(resolution=30, n=20, plot_wigner=True, plot_msd=True):
    # strange geometry issues that I don't have time to deal with arise between modules
    # fix rules:
    # Rz ok, Ry does -Rx rotation, Rx does Ry rotation
    # Arcs start at 0
    # pi_pulse_x realises y


    # initialize bloch ball
    bloch = qutip.Bloch()
    # bloch.xlabel = ["$y$", ""]
    # bloch.ylabel = ["", "$x$"]

    bloch.fig = plt.figure(figsize=(7, 7))
    bloch.axes = bloch.fig.add_subplot(111, projection='3d')
    bloch.axes.view_init(45, 45)

    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)

    # add quantum states
    coherent = oat_state(n, 0.1)
    rotated_pi_1 = pi_pulse_x(coherent)
    rotated_phi = rotation(rotated_pi_1, PHI, j_z)
    rotated_pi_2 = pi_pulse_x(rotated_phi)

    states = [coherent, rotated_pi_1]

    # create the sphere surface of r = 1
    r = 1
    x = r * np.outer(np.cos(phi), np.sin(theta))
    y = r * np.outer(np.sin(phi), np.sin(theta))
    z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))

    if plot_wigner:
        # calculate wigner
        wigners = []
        for s in states:
            wigners.append(get_spin_wigner(s, theta, phi))
            print("msd: {}".format(MSD(s)))


        # plot wigners on bloch ball
        wigner = sum(wigners)
        wigner /= wigner.max()
        colors = get_spin_wigner_colormap(wigner)
        bloch.axes.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=colors)

    if plot_msd:
        # plot MSDs of states
        MSDs = [MSD(s, for_bloch=True) for s in states]
        bloch.add_vectors(MSDs)
        bloch.vector_width = 10
        bloch.vector_mutation = 50

    # experimental arrows



    r = 0.3

    # 1st vector
    arc = make_arc(np.pi / 2, np.pi, r, plane="z-x")
    plot_arc_on_bloch(arc, bloch)

    # 2nd vector
    # arc = make_arc(np.pi, np.pi + PHI, r, plane="xy")
    # plot_arc_on_bloch(arc, bloch)

    # 3rd vector
    # arc = make_arc(-np.pi, -np.pi/2, r, plane="z-x")
    # plot_arc_on_bloch(arc, bloch)

    # arc = np.arange(0, 116) * np.pi / 180 - (np.pi / 2)
    # p = np.array([np.cos(arc), np.sin(arc), arc * 0]) * r
    # bloch.axes.plot(p[0, :], p[1, :], p[2, :], 'b--')
    # p = Rz(np.pi / 2).dot(p)
    # bloch.axes.plot(p[0, :], p[1, :], p[2, :], 'r--')
    # p = Ry(- np.pi / 2).dot(p)
    # bloch.axes.plot(p[0, :], p[1, :], p[2, :], 'k--')

    # arc = np.arange(-5, 105) * np.pi / 180
    # p = np.array([np.sin(arc), arc * 0, np.cos(arc)])
    # p = Rz(psi).dot(p)
    # bloch.axes.plot(p[0, :], p[1, :], p[2, :], 'k--')

    # arc = linspace(0, 0.87 * np.pi, resolution) + np.pi / 2
    # p = np.array([np.cos(arc), np.sin(arc), arc * 0]) * 0.4
    # bloch.axes.plot(p[0, :], p[1, :], p[2, :], 'k--')

    # o = [0, 0, 0]
    # t = [0, 0, 1]
    # t = Rx(np.pi / 2).dot(t)
    # a = Arrow3D([o[0], t[0]], [o[1], t[1]], [o[2], t[2]], **arrow_prop_dict)
    # bloch.axes.add_artist(a)



    bloch.show()
    # bloch.save("/Users/derrkater/Desktop/mgr/mgr/img/coherent_3")


plot_bloch_evolution(100, plot_wigner=True)
