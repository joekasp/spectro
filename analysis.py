import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis


"""Do PCA, ICA, or factor analysis and return components or coefficients."""

def doPCA(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Returns PCA object"""
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshapeImage(data)
    pca = PCA(n_components=n_comp)
    pca.fit(data_r)
    return pca

def getPCAComponents(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Return reshaped PCA components"""
    data = np.nan_to_num(data)
    pca = doPCA(data, w1, w3, tau2, normalize, n_comp)
    comp = np.zeros((data.shape[0], data.shape[1], pca.components_.shape[0]))
    for i in range(pca.components_.shape[0]):
        comp[:, :, i] = pca.components_[i].reshape(data.shape[0], data.shape[1])
    return comp

def getPCAProjections(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Return coefficients of projections onto new components"""
    data = np.nan_to_num(data)
    pca = doPCA(data, w1, w3, tau2, normalize, n_comp)
    data_r = reshapeImage(data)
    return pca.transform(data_r)



def doICA(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Returns ICA object"""
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshapeImage(data)
    ica = FastICA(n_components=n_comp, whiten=True)
    ica.fit(data_r)
    return ica

def getICAComponents(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Return reshaped ICA components"""
    data = np.nan_to_num(data)
    ica = doICA(data, w1, w3, tau2, normalize, n_comp)
    comp = np.zeros((data.shape[0], data.shape[1], ica.components_.shape[0]))
    for i in range(ica.components_.shape[0]):
        comp[:, :, i] = ica.components_[i].reshape(data.shape[0], data.shape[1])
    return comp

def getICAProjections(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Return coefficients of projections onto new components"""
    data = np.nan_to_num(data)
    ica = doICA(data, w1, w3, tau2, normalize, n_comp)
    data_r = reshapeImage(data)
    return ica.transform(data_r)



def doFactA(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Returns FactA object"""
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshapeImage(data)
    facta = FactorAnalysis(n_components=n_comp)
    facta.fit(data_r)
    return facta

def getFactAComponents(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Return reshaped FactA components"""
    data = np.nan_to_num(data)
    facta = doFactA(data, w1, w3, tau2, normalize, n_comp)
    comp = np.zeros((data.shape[0], data.shape[1], facta.components_.shape[0]))
    for i in range(facta.components_.shape[0]):
        comp[:, :, i] = facta.components_[i].reshape(data.shape[0], data.shape[1])
    return comp

def getFactAProjections(data, w1, w3, tau2, normalize=False, n_comp=10):
    """Return coefficients of projections onto new components"""
    data = np.nan_to_num(data)
    facta = doFactA(data, w1, w3, tau2, normalize, n_comp)
    data_r = reshapeImage(data)
    return facta.transform(data_r)




def reshapeImage(data):
    """
    Reshape each 2D image in the stack into 1D
    example: 109 x 109 x 13 original matrix (stack of 13 109x109 images)
        becomes 13 x 11881 (since 11881 = 109 x 109)
    """
    data_r = np.zeros((data.shape[2], data.shape[0] * data.shape[1]))
    for i in range(data.shape[2]):
        data_r[i] = data[:,:,i].ravel()
    return data_r
