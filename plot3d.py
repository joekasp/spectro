import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def surf3d(x,y,Z):
    '''
    Performs 3d surface plot for x,y,Z data
    '''
    X,Y = np.meshgrid(y,x)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=0,cmap=cm.rainbow)
    ax.view_init(elev=30., azim=-35)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('$\omega_{1}$')
    ax.set_ylabel('$\omega_{3}$')
    plt.show()

    return

