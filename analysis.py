import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis


"""Do PCA, ICA, or factor analysis and return components or coefficients."""


def do_analysis(data, normalize=False, n_comp=10, analysis_type='pca'):
    """
    Do component analysis on the input data and return a fit object.
    :param data: numpy array, set of images to be analyzed
                 set of Z images, each X x Y
    :param normalize: boolean, True to normalize data before doing analysis
    :param n_comp: int, number of components to generate
    :param analysis_type: string for the type of analysis to perform
                          'pca' for Principal Component Analysis (default)
                          'ica' for Independent Component Analysis
                          'fa' for Factor Analysis
    :return: PCA, ICA, or factor analysis object
    """
    data = np.nan_to_num(data)
    if normalize:
        for idx in range(data.shape[2]):
                img = data[:, :, idx]
                peak = np.abs(img).max()
                img = img / peak
                data[:, :, idx] = img

    data_r = reshape_image(data)

    if analysis_type == 'pca':
        fit_object = PCA(n_components=n_comp)
    elif analysis_type == 'ica':
        fit_object = FastICA(n_components=n_comp, whiten=True)
    elif analysis_type == 'fa':
        fit_object = FactorAnalysis(n_components=n_comp)
    fit_object.fit(data_r)
    return fit_object


def get_components(data, normalize=False, n_comp=10, analysis_type='pca'):
    """
    Do component analysis on the input data and return set of component images.
    :param data: numpy array, set of images to be analyzed
                 set of Z images, each X x Y
    :param normalize: boolean, True to normalize data before doing analysis
    :param n_comp: int, number of components to generate
    :param analysis_type: string for the type of analysis to perform
                          'pca' for Principal Component Analysis (default)
                          'ica' for Independent Component Analysis
                          'fa' for Factor Analysis
    :return: numpy array with dimensions (X, Y, n_comp)
             set of n_comp images, each X x Y
    """
    data = np.nan_to_num(data)
    fit_object = do_analysis(data, normalize, n_comp, analysis_type)
    comp = np.zeros((data.shape[0], data.shape[1], fit_object.components_.shape[0]))
    for i in range(fit_object.components_.shape[0]):
        comp[:, :, i] = fit_object.components_[i].reshape(data.shape[0], data.shape[1])
    return comp


def get_projections(data, normalize=False, n_comp=10, analysis_type='pca'):
    """
    Do PCA on the input data and return projection of original data onto components.
    :param data: numpy array, set of images to be analyzed
                 set of Z images, each X x Y
    :param normalize: boolean, True to normalize data before doing analysis
    :param n_comp: int, number of components to generate
    :param analysis_type: string for the type of analysis to perform
                          'pca' for Principal Component Analysis (default)
                          'ica' for Independent Component Analysis
                          'fa' for Factor Analysis
    :return: numpy array with dimensions (Z, n_comp)
             corresponding to the contribution of each component to each original image
    """
    data = np.nan_to_num(data)
    fit_object = do_analysis(data, normalize, n_comp, analysis_type)
    data_r = reshape_image(data)
    return fit_object.transform(data_r)


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
