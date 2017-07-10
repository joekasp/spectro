import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'mathtext.default':'regular'})

# "Plot" functions return axes objects
# "Show" functions display plots

FIG_SIZE = 4  # Default size for single-panel figures

def plot_image(ax, img, w1, w3, colormap='jet'):
    """Plot data image or component, return axes object"""
    w1grid, w3grid = np.meshgrid(w3, w1)
    ax.pcolormesh(w1grid, w3grid, img,
                  vmin=-np.abs(img).max(), vmax=np.abs(img).max(),cmap=colormap)
    ax.set_xlabel('$\omega_1/2\pi c\ (cm^{-1})$', fontsize=14)
    ax.set_ylabel('$\omega_3/2\pi c\ (cm^{-1})$', fontsize=14)
    ax.set_xlim(w1grid.min(), w1grid.max())
    ax.set_ylim(w3grid.min(), w3grid.max())
    return ax


def show_data(data, w1, w3, tau2, time):
    """Show plot of single data image nearest to time"""
    fig, ax = plt.subplots(figsize=(FIG_SIZE,FIG_SIZE))
    i = time_to_index(tau2, time)
    ax = plot_image(ax, data[:, :, i], w1, w3)
    ax.set_title("t = " + str(tau2[i][0]) + " fs", fontsize=14)
    plt.show()


def show_3_data(data, w1, w3, tau2, times=[0,0,0]):
    """Show plot of 3 spectra at 3 times"""
    fig, axarr = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.set_size_inches(11, 3)

    w1grid, w3grid = np.meshgrid(w3, w1)
    plt.setp(axarr.flat, adjustable='box-forced',
             aspect=(w1grid.max()-w1grid.min()) / (w3grid.max()-w3grid.min()))

    for i in range(3):
        ax = axarr.flatten()[i]
        time_i = time_to_index(tau2, times[i])
        plot_image(ax, data[:,:,time_i], w1, w3)
        ax.set_title("t = " + str(tau2[time_i][0]) + " fs")
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)


def show_component(comp, w1, w3, i):
    """Show plot of single component i"""
    fig, ax = plt.subplots(figsize=(FIG_SIZE,FIG_SIZE))
    ax = plot_image(ax, comp[:,:,i-1], w1, w3, 'bwr')
    ax.set_title("Component " + str(i), fontsize=14)
    plt.show()


def show_3_components(comp, w1, w3):
    fig, axarr = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.set_size_inches(11, 3)

    w1grid, w3grid = np.meshgrid(w3, w1)
    plt.setp(axarr.flat, adjustable='box-forced',
             aspect=(w1grid.max()-w1grid.min()) / (w3grid.max()-w3grid.min()))

    for i in range(3):
        ax = axarr.flatten()[i]
        plot_image(ax, comp[:,:,i], w1, w3, 'bwr')
        ax.set_title("Component " + str(i+1))
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)



def plot_contribution(ax, tau2, proj, n_comp):
    ax.plot([-1000, tau2[-1] + 1000], [0,0], c='black')  # zero line
    ax.scatter(tau2, proj[:, n_comp-1])
    ax.set_xlabel("Time (fs)", fontsize=14)
    ax.set_xlim(-1000, tau2[-1] + 1000)
    ax.set_title("Component " + str(n_comp), fontsize=14)
    return ax


def show_contribution(tau2, proj, n_comp):
    """Plot the contribution of the selected component vs time"""
    fig, ax = plt.subplots(figsize=(FIG_SIZE,FIG_SIZE))
    ax = plot_contribution(ax, tau2, proj, n_comp)
    plt.show()


def time_to_index(tau2, target):
    """Return the index after where target time would occur in tau2"""
    for time in tau2:
        if time >= target:
            return np.where(tau2==time)[0][0]
    return tau2.shape[0] - 1

