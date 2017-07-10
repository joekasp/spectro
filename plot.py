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

def plotImage(ax, img, w1, w3):
    """Plot data image or component, return axes object"""
    w1grid, w3grid = np.meshgrid(w3, w1)
    ax.pcolormesh(w1grid, w3grid, img,
                  vmin=-np.abs(img).max(), vmax=np.abs(img).max(),cmap=cm.bwr)
    ax.set_xlabel('$\omega_1/2\pi c\ (cm^{-1})$', fontsize=14)
    ax.set_ylabel('$\omega_3/2\pi c\ (cm^{-1})$', fontsize=14)
    ax.set_xlim(w1grid.min(), w1grid.max())
    ax.set_ylim(w3grid.min(), w3grid.max())
    return ax

def showComponent(comp, w1, w3, i):
    """Show plot of single component i"""
    fig, ax = plt.subplots(figsize=(5,5))
    ax = plotImage(ax, comp[:,:,i-1], w1, w3)
    ax.set_title("Component " + str(i), fontsize=14)
    plt.show()

def showData(data, w1, w3, tau2, i):
    """Show plot of single data image i"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax = plotImage(ax, data[:, :, i], w1, w3)
    ax.set_title("t = " + str(tau2[i][0]) + " fs", fontsize=14)
    plt.show()


def show3Components(comp, w1, w3):
    fig, axarr = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    fig.set_size_inches(11, 3)

    w1grid, w3grid = np.meshgrid(w3, w1)
    plt.setp(axarr.flat, adjustable='box-forced',
             aspect=(w1grid.max()-w1grid.min()) / (w3grid.max()-w3grid.min()))

    for i in range(3):
        ax = axarr.flatten()[i]
        plotImage(ax, comp[:,:,i], w1, w3)
        ax.set_title("Component " + str(i+1))
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)


# to do: add zero line, change font sizes
def plotContribution(tau2, proj, n_comp):
    fig, ax = plt.subplots()
    plt.scatter(tau2, proj[:, n_comp-1])
    ax.set_title("Component " + str(n_comp))
    ax.set_xlabel("Time (fs)")
    plt.show()

