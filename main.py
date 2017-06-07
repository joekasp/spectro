from util import *
from pca import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    W1_NAME = '2D-IR-Data_SNP_NO_Brookes_JPCB_2013/EtOH/w1.mat'
    W3_NAME = '2D-IR-Data_SNP_NO_Brookes_JPCB_2013/EtOH/w3_corrected_EtOH.mat'
    TAU2_NAME = '2D-IR-Data_SNP_NO_Brookes_JPCB_2013/EtOH/Tau_2_EtOH.mat'
    DATA_ARRAY_NAME = '2D-IR-Data_SNP_NO_Brookes_JPCB_2013/EtOH/Matexp_EtOH.mat'

    xw1 = loadData(W1_NAME)
    xw3 = loadData(W3_NAME)
    xt2 = loadData(TAU2_NAME)
    xdata = loadData(DATA_ARRAY_NAME)

    data, vec_list = matchDims(xdata,[xw1,xw3,xt2])
    w1 = vec_list[0]
    w3 = vec_list[1]
    tau2 = vec_list[2]

    doPCA(data,w1,w3,tau2)
    
#    # create the plot
#    X,Y = np.meshgrid(w3,w1)
#    levels = np.linspace(-1,1,25)
#    C = plt.contourf(X,Y,data[:,:,1],levels)
#    plt.colorbar(C,shrink=0.8,extend='both')
#    plt.show()

