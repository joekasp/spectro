import math
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def doPCA(data,w1,w3,tau2,n_comp=10):
    data_r = np.zeros((data.shape[2],data.shape[0]*data.shape[1]))
    for i in range(data.shape[2]):
        data_r[i] = np.nan_to_num(data[:,:,i]).ravel()

    pca = PCA(n_components=n_comp)
    pca.fit(data_r)
    comp = np.zeros((data.shape[0],data.shape[1],pca.components_.shape[0]))
    for i in range(pca.components_.shape[0]):
        comp[:,:,i] = pca.components_[i].reshape(data.shape[0],data.shape[1])

    # Plot a series of components
    w1grid, w3grid = np.meshgrid(w3, w1)
    fig, axarr = plt.subplots(3,3, figsize=(9,9), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3,wspace=0.4)
    for i in range(9):
        ax = axarr.flatten()[i]
        img = comp[:,:,i]
        ax.pcolormesh(w1grid, w3grid, img)
        ax.set_title('Component ' + str(i))
        ax.set_xlim(w1grid.min(), w1grid.max())
        ax.set_ylim(w3grid.min(), w3grid.max())
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)
    plt.show()
