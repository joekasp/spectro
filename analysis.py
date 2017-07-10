import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis


"""Do PCA, ICA, or factor analysis and return components or coefficients."""


def do_pca(data, normalize=False, n_comp=10):
    """
    Do PCA on the input data and return a PCA object.
    :param data: numpy array, set of images to be analyzed
                 set of Z images, each X x Y
    :param normalize: boolean, True to normalize data before doing PCA
    :param n_comp: int, number of components to generate
    :return: PCA object
    """
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshape_image(data)
    pca = PCA(n_components=n_comp)
    pca.fit(data_r)
    return pca


def get_pca_components(data, normalize=False, n_comp=10):
    """
    Do PCA on the input data and return set of component images.
    :param data: numpy array, set of images to be analyzed
                 set of Z images, each X x Y
    :param normalize: boolean, True to normalize data before doing PCA
    :param n_comp: int, number of components to generate
    :return: numpy array with dimensions (X, Y, n_comp)
             set of n_comp images, each X x Y
    """
    data = np.nan_to_num(data)
    pca = do_pca(data, normalize, n_comp)
    comp = np.zeros((data.shape[0], data.shape[1], pca.components_.shape[0]))
    for i in range(pca.components_.shape[0]):
        comp[:, :, i] = pca.components_[i].reshape(data.shape[0], data.shape[1])
    return comp


def get_pca_projections(data, normalize=False, n_comp=10):
    """
    Do PCA on the input data and return projection of original data onto components.
    :param data: numpy array, set of images to be analyzed
                 set of Z images, each X x Y
    :param normalize: boolean, True to normalize data before doing PCA
    :param n_comp: int, number of components to generate
    :return: numpy array with dimensions (Z, n_comp)
             corresponding to the contribution of each component to each original image
    """
    data = np.nan_to_num(data)
    pca = do_pca(data, normalize, n_comp)
    data_r = reshape_image(data)
    return pca.transform(data_r)


def do_ica(data, normalize=False, n_comp=10):
    """Returns ICA object"""
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshape_image(data)
    ica = FastICA(n_components=n_comp, whiten=True)
    ica.fit(data_r)
    return ica


def get_ica_components(data, normalize=False, n_comp=10):
    """Return reshaped ICA components"""
    data = np.nan_to_num(data)
    ica = do_ica(data, normalize, n_comp)
    comp = np.zeros((data.shape[0], data.shape[1], ica.components_.shape[0]))
    for i in range(ica.components_.shape[0]):
        comp[:, :, i] = ica.components_[i].reshape(data.shape[0], data.shape[1])
    return comp


def get_ica_projections(data, normalize=False, n_comp=10):
    """Return coefficients of projections onto new components"""
    data = np.nan_to_num(data)
    ica = do_ica(data, normalize, n_comp)
    data_r = reshape_image(data)
    return ica.transform(data_r)


def do_facta(data, normalize=False, n_comp=10):
    """Returns FactA object"""
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshape_image(data)
    facta = FactorAnalysis(n_components=n_comp)
    facta.fit(data_r)
    return facta


def get_facta_components(data, normalize=False, n_comp=10):
    """Return reshaped FactA components"""
    data = np.nan_to_num(data)
    facta = do_facta(data, normalize, n_comp)
    comp = np.zeros((data.shape[0], data.shape[1], facta.components_.shape[0]))
    for i in range(facta.components_.shape[0]):
        comp[:, :, i] = facta.components_[i].reshape(data.shape[0], data.shape[1])
    return comp


def get_facta_projections(data, normalize=False, n_comp=10):
    """Return coefficients of projections onto new components"""
    data = np.nan_to_num(data)
    facta = do_facta(data, normalize, n_comp)
    data_r = reshape_image(data)
    return facta.transform(data_r)


def reshape_image(data):
    """
    Reshape each 2D image in the stack into 1D
    example: 109 x 109 x 13 original matrix (stack of 13 109x109 images)
        becomes 13 x 11881 (since 11881 = 109 x 109)
    """
    data_r = np.zeros((data.shape[2], data.shape[0] * data.shape[1]))
    for i in range(data.shape[2]):
        data_r[i] = data[:,:,i].ravel()
    return data_r
