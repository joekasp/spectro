import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def surf3d(x,y,Z):
    '''
    Performs 3d surface plot for x,y,Z data
    '''
    X,Y = np.meshgrid(y,x)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    TICK_LIMIT = 5

    surf = ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=0,cmap=cm.rainbow)
    ax.view_init(elev=25., azim=-50)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('$\omega_{1} (cm^{-1})$')
    ax.set_ylabel('$\omega_{3} (cm^{-1})$')
    ax.xaxis.labelpad=15
    ax.yaxis.labelpad=15
    xticks = ax.get_xticks()
    nskip = round(len(xticks)/TICK_LIMIT)
    ax.set_xticks(xticks[::nskip])
    yticks = ax.get_yticks()
    nskip = round(len(yticks)/TICK_LIMIT)
    ax.set_yticks(yticks[::nskip])

    return ax

