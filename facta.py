import math
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from fits import *
from plot3d import *

def doFactA(data,w1,w3,tau2,n_comp=10):
    data_r = np.zeros((data.shape[2],data.shape[0]*data.shape[1]))
    for i in range(data.shape[2]):
        data_r[i] = np.nan_to_num(data[:,:,i]).ravel()

    # Standardize
#    data_r = (data_r - np.mean(data_r,axis=0))/np.std(data_r,ddof=1,axis=0)
#    data_r = normalize(data_r,norm='l2',axis=0)

    facta = FactorAnalysis(n_components=n_comp)
    facta.fit(data_r)
    comp = np.zeros((data.shape[0],data.shape[1],facta.components_.shape[0]))
    for i in range(facta.components_.shape[0]):
        comp[:,:,i] = facta.components_[i].reshape(data.shape[0],data.shape[1])

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

    #create surface plots
    surf3d(w1,w3,comp[:,:,0])
    surf3d(w1,w3,comp[:,:,1])
    surf3d(w1,w3,comp[:,:,2])

    data_c = facta.transform(data_r)

    # Plot filtered contours
    w1grid, w3grid = np.meshgrid(w3, w1)
    fig, axarr = plt.subplots(3,3, figsize=(9,9), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3,wspace=0.4)
    for j in range(9):
        ax = axarr.flatten()[j]
        i = 2*j
        img = comp[:,:,0]*data_c[i,0] + comp[:,:,1]*data_c[i,1] + \
            comp[:,:,2]*data_c[i,2] + comp[:,:,3]*data_c[i,3] + comp[:,:,4]*data_c[i,4] 
        ax.pcolormesh(w1grid, w3grid, img)
        ax.set_title('Time ' + str(tau2[i]))
        ax.set_xlim(w1grid.min(), w1grid.max())
        ax.set_ylim(w3grid.min(), w3grid.max())
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)
    plt.show()

    # Plot against time
    plt.figure(figsize=(5,5))
    plt.scatter(tau2, data_c[:,0], color='red', label='C0') # component 0
    plt.scatter(tau2, data_c[:,1], color='orange', label='C1')
    plt.scatter(tau2, data_c[:,2], color='green', label='C2')
    plt.scatter(tau2, data_c[:,3], color='blue', label='C3')
    plt.xlabel('Time (fs)')
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

    #print(tau2.shape)
    p0 = .1,.1,.1 
    popt, pcov = curve_fit(my_exponential,tau2.ravel()*0.001,data_c[:,0],p0,maxfev=1000)
    print(popt) 
