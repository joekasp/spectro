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

    data_c = pca.transform(data_r)

    # Plot against time
    plt.figure(figsize=(5,5))
    plt.scatter(tau2, data_c[:,0], color='red', label='C0') # component 0
    plt.scatter(tau2, data_c[:,1], color='orange', label='C1')
    plt.scatter(tau2, data_c[:,2], color='green', label='C2')
    plt.scatter(tau2, data_c[:,3], color='blue', label='C3')
    plt.xlabel('Time')
    plt.legend(loc='lower right')
    
    # Plot value of component contribution vs. component for selected times
    plt.figure(figsize=(5,5))
    comp_num = list(range(10))
    plt.plot(comp_num, (data_c[0]), color='red', label='t=0') # component 0
    plt.plot(comp_num, (data_c[3]), color='orange', label='t=270')
    plt.plot(comp_num, (data_c[6]), color='green', label='t=540')
    plt.plot(comp_num, (data_c[9]), color='blue', label='t=5000')
    plt.plot(comp_num, (data_c[12]), color='purple', label='t=20000')
    plt.xlim(-0.5,10.5)
    plt.ylim(-5, 6.6)
    plt.xlabel('Component')
    plt.ylabel('Component contribution')
    plt.legend(loc='upper right')
    plt.show()
