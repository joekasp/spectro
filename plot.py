import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'mathtext.default': 'regular'})
import fits

# "Plot" functions return axes objects
# "Show" functions display plots

FIG_SIZE = 4  # Default size for single-panel figures


def plot_image(ax, img, w1, w3, colormap='jet'):
    """
    Plot an image (data or component) and return axes object.
    :param ax: axes object
    :param img: image to plot, X x Y numpy array
    :param w1: x-axis, X x 1 numpy array
    :param w3: y-axis, Y x 1 numpy array
    :param colormap: optional colormap to use
    :return: the input axes object with the input image plotted on it
    """
    w1grid, w3grid = np.meshgrid(w3, w1)
    ax.pcolormesh(w1grid, w3grid, img,
                  vmin=-np.abs(img).max(), vmax=np.abs(img).max(), cmap=colormap)
    ax.set_xlabel('$\omega_1/2\pi c\ (cm^{-1})$', fontsize=14)
    ax.set_ylabel('$\omega_3/2\pi c\ (cm^{-1})$', fontsize=14)
    ax.set_xlim(w1grid.min(), w1grid.max())
    ax.set_ylim(w3grid.min(), w3grid.max())
    return ax


def show_data(data, w1, w3, tau2, time):
    """
    Show a plot of the first image from the input data with time after the input time.
    :param data: numpy array, set of Z images, each X x Y
    :param w1: x-axis, X x 1 numpy array
    :param w3: y-axis, Y x 1 numpy array
    :param tau2: list of times, Z x 1 numpy array
    :param time: target time
                 This function finds the first image after the target time.
    :return displays a figure
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    i = time_to_index(tau2, time)
    ax = plot_image(ax, data[:, :, i], w1, w3)
    ax.set_title("t = " + str(tau2[i][0]) + " fs", fontsize=14)
    plt.show()


def show_3_data(data, w1, w3, tau2, times=[0, 0, 0]):
    """
    Show 3 plots of images from the input data corresponding to the input times.
    :param data: numpy array, set of Z images, each X x Y
    :param w1: x-axis, X x 1 numpy array
    :param w3: y-axis, Y x 1 numpy array
    :param tau2: list of times, Z x 1 numpy array
    :param times: list of 3 target times in fs
    :return: displays a figure
    """
    fig, axarr = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.set_size_inches(11, 3)

    w1grid, w3grid = np.meshgrid(w3, w1)
    plt.setp(axarr.flat, adjustable='box-forced',
             aspect=(w1grid.max()-w1grid.min()) / (w3grid.max()-w3grid.min()))

    for i in range(3):
        ax = axarr.flatten()[i]
        time_i = time_to_index(tau2, times[i])
        plot_image(ax, data[:, :, time_i], w1, w3)
        ax.set_title("t = " + str(tau2[time_i][0]) + " fs")
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)


def show_component(comp, w1, w3, comp_num):
    """
    Show a plot of the specified component image.
    :param comp: numpy array, set of n_comp images, each X x Y
    :param w1: x-axis, X x 1 numpy array
    :param w3: y-axis, Y x 1 numpy array
    :param comp_num: component number to plot (1 through n_comp)
    :return: displays a figure
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax = plot_image(ax, comp[:, :, comp_num-1], w1, w3, 'bwr')
    ax.set_title("Component " + str(comp_num), fontsize=14)
    plt.show()


def show_3_components(comp, w1, w3):
    """
    Show a plot of the first 3 component images.
    :param comp: numpy array, set of n_comp images, each X x Y
    :param w1: x-axis, X x 1 numpy array
    :param w3: y-axis, Y x 1 numpy array
    :return: displays a figure
    """
    fig, axarr = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.set_size_inches(11, 3)

    w1grid, w3grid = np.meshgrid(w3, w1)
    plt.setp(axarr.flat, adjustable='box-forced',
             aspect=(w1grid.max()-w1grid.min()) / (w3grid.max()-w3grid.min()))

    for i in range(3):
        ax = axarr.flatten()[i]
        plot_image(ax, comp[:, :, i], w1, w3, 'bwr')
        ax.set_title("Component " + str(i+1))
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)


def plot_contribution(ax, tau2, proj, comp_num=1):
    """
    Plot projection vs time for a specified component, return axes object.
    :param ax: axes object
    :param tau2: list of times, Z x 1 numpy array
    :param proj: projection of data onto components, Z x n_comp numpy array
    :param comp_num: component number to plot, default 1
    :return: axes object with scatter plot of projection vs time
    """
    buffer = tau2[-1][0]/100
    ax.plot([-buffer, tau2[-1] + buffer], [0, 0], c='black')  # zero line
    ax.scatter(tau2, proj[:, comp_num-1])
    ax.set_xlabel("Time (fs)", fontsize=14)
    ax.set_xlim(-buffer, tau2[-1] + buffer)
    ax.set_title("Component " + str(comp_num), fontsize=14)
    return ax


def show_contribution(tau2, proj, comp_num):
    """
    Show scatter plot of dynamics for specified component.
    :param tau2: list of times, Z x 1 numpy array
    :param proj: projection of data onto components, Z x n_comp numpy array
    :param comp_num: component to plot
    :return: shows a figure
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax = plot_contribution(ax, tau2, proj, comp_num)
    plt.show()


def plot_exp_fit(ax, popt, tau2):
    """
    Plot exponential fit function, return axes object.
    :param ax: axes object
    :param popt: list of fit parameters
                 must be from either fits.my_exponential or fits.my_double_exp
    :param tau2: list of times, Z x 1 numpy array
    :return: axes object with plot of exponential function
    """
    if popt.shape[0] == 3:  # single exponential
        ax.plot(tau2, fits.my_exponential(tau2, popt[0], popt[1], popt[2]))
    elif popt.shape[0] == 5:
        ax.plot(tau2, fits.my_double_exp(tau2, popt[0], popt[1], popt[2], popt[3], popt[4]))
    return ax


def show_exp_fit(tau2, proj, comp_num, popt, T_SCALE):
    """
    Show scatter plot and exponential fit for dynamics for specified component.
    :param tau2: list of times, Z x 1 numpy array
    :param proj: projection of data onto components, Z x n_comp numpy array
    :param comp_num: component to plot
    :param popt: list of fit parameters from exponential fit of the component
                 must be from either fits.my_exponential or fits.my_double_exp
    :param T_SCALE: scaling parameter used for fitting
                    1000 for ps, 1 for fs
    :return: shows a figure
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax = plot_contribution(ax, tau2/T_SCALE, proj, comp_num)
    ax = plot_exp_fit(ax, popt, tau2/T_SCALE)
    time_units = ""
    if T_SCALE == 1000:
        time_units = "(ps)"
    elif T_SCALE == 1:
        time_units = "(fs)"
    elif T_SCALE == 1000000:
        time_units = "(ns)"
    ax.set_xlabel("Time " + time_units)

    buffer = tau2[-1][0]/T_SCALE / 100
    ax.set_xlim(-buffer, tau2[-1]/T_SCALE + buffer)
    plt.show()


def time_to_index(tau2, target):
    """Return the index after where target time would occur in tau2"""
    for time in tau2:
        if time >= target:
            return np.where(tau2 == time)[0][0]
    return tau2.shape[0] - 1