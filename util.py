import numpy as np
from scipy.io import loadmat

def loadData(filename):
    '''
    Loads a .mat (MATLAB) file and returns
    the vector/matrix as a numpy array.

    Parameters
    ----------
    filename: string containing path of file

    Returns
    -------
    out: a numpy array

    '''
    raw = loadmat(filename)
    for key in raw.keys():
        if (key[:2]=='__'):
            pass
        else:
            out = raw[key]

    return out


def matchDims(main,obj_list):
    '''
    Truncates and matches dimensions of multidimensional arrays.
    
    Parameters
    ----------
    main: the Numpy multidimensional data array
    obj_list: list containing the Numpy vectors 
        describing each dimension (in order).

    Returns
    -------
    t_main: truncated version of input array
    t_obj_list: list with truncated versions of input vectors

    '''
    dims = main.shape
    t_main = main[...]
    t_obj_list = []
    for i in range(len(dims)):
        idim = len(obj_list[i])
        if(idim < dims[i]):
            t_main = np.delete(t_main,[idim:],i)
            t_obj_list.append(obj_list[i])
        else:
            t_obj_list.append(obj_list[i][:dims[i]])

    return t_main, t_obj_list 
