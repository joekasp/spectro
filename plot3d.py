import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def surf3d(x,y,Z,window_title='Figure',ax_title='',fig='None',azim=-50,elev=25):
    '''
    Performs 3d surface plot for x,y,Z data
    '''
    X,Y = np.meshgrid(y,x)
    if(fig=='None'):
        fig = plt.figure(figsize=(12,6))
    fig.canvas.set_window_title(window_title)
    ax = fig.gca(projection='3d')

    TICK_LIMIT = 5

    surf = ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=0,cmap=cm.rainbow)
    ax.set_zlim3d(-1.0,1.0)
    ax.view_init(elev=elev, azim=azim)
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
    ax.set_title(ax_title)

    return ax

