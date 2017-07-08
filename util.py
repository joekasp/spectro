import numpy as np
from scipy.io import loadmat


def loadSolvent(name):
    '''
    Loads all the necessary data arrays for a given solvent.
    
    Parameters
    ----------
    name : string of the solvent name

    Returns
    -------
    out : list containing the data arrays

    '''
    PREFIX = '2D-IR-Data_SNP_NO_Brookes_JPCB_2013/'

    if name == 'D2O':
        W1_NAME = PREFIX + 'D2O/w1.mat'
        W3_NAME = PREFIX + 'D2O/w3_corrected_D2O.mat'
        TAU2_NAME = PREFIX + 'D2O/Tau_2_exp.mat'
        DATA_ARRAY_NAME = PREFIX + 'D2O/Matexp_D2O.mat'
    elif name == 'DMSO': 
        W1_NAME = PREFIX + 'DMSO/w1.mat'
        W3_NAME = PREFIX + 'DMSO/w3_corrected_DMSO.mat'
        TAU2_NAME = PREFIX + 'DMSO/Tau_2_DMSO.mat'
        DATA_ARRAY_NAME = PREFIX + 'DMSO/Matexp_DMSO.mat'
    elif name == 'EG':
        W1_NAME = PREFIX + 'EG/w1.mat'
        W3_NAME = PREFIX + 'EG/w3_corrected_EG.mat'
        TAU2_NAME = PREFIX + 'EG/Tau_2_exp.mat'
        DATA_ARRAY_NAME = PREFIX + 'EG/Matexp_EG.mat'
    elif name == 'EtOH':
        W1_NAME = PREFIX + 'EtOH/w1.mat'
        W3_NAME = PREFIX + 'EtOH/w3_corrected_EtOH.mat'
        TAU2_NAME = PREFIX + 'EtOH/Tau_2_EtOH.mat'
        DATA_ARRAY_NAME = PREFIX + 'EtOH/Matexp_EtOH.mat'
    elif name == 'FA':
        W1_NAME = PREFIX + 'FA/w1.mat'
        W3_NAME = PREFIX + 'FA/w3_corrected_FA.mat'
        TAU2_NAME = PREFIX + 'FA/Tau_2_FA.mat'
        DATA_ARRAY_NAME = PREFIX + 'FA/Matexp_FA.mat'
    elif name == 'H2O':
        W1_NAME = PREFIX + 'H2O/w1.mat'
        W3_NAME = PREFIX + 'H2O/w3_corrected_H2O.mat'
        TAU2_NAME = PREFIX + 'H2O/Tau_2_H2O.mat'
        DATA_ARRAY_NAME = PREFIX + 'H2O/Matexp_H2O.mat'
    elif name == 'MeOH':
        W1_NAME = PREFIX + 'MeOH/w1.mat'
        W3_NAME = PREFIX + 'MeOH/w3_corrected_MeOH.mat'
        TAU2_NAME = PREFIX + 'MeOH/Tau_2_MeOH.mat'
        DATA_ARRAY_NAME = PREFIX + 'MeOH/Matexp_MeOH.mat'
    else:
       raise ValueError('Solvent name not found')

    xw1 = loadData(W1_NAME)
    xw3 = loadData(W3_NAME)
    xt2 = loadData(TAU2_NAME)
    xdata = loadData(DATA_ARRAY_NAME)

    data, vec_list = matchDims(xdata,[xw1,xw3,xt2])
    w1 = vec_list[0]
    w3 = vec_list[1]
    tau2 = vec_list[2]

    return data,w1,w3,tau2


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
            t_main = np.delete(t_main,np.s_[idim:],i)
            t_obj_list.append(obj_list[i])
        else:
            t_obj_list.append(obj_list[i][:dims[i]])

    return t_main, t_obj_list 
