import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def loadMat(matPath):
    # read MATLAB processed 5D lightfield
    # L(u,v,s,t,c)

    mat = sio.loadmat(matPath)
    L = mat['LF']

    return L

def reconImg(L):
    # reconstruct light field image from 5D focal stack

    [U, V, S, T, C] = L.shape
    imgRecon = np.zeros((U*S,V*T,C))

    for s in range(0,S):
        for t in range(0,T):
            imgRecon[s*U:(s+1)*U,t*V:(t+1)*V,:] = L[:,:,s,t,:]

    return imgRecon

#code for setting axes to be equal in matplotlib3D
#taken from https://stackoverflow.com/questions/13685386/63625222#63625222

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

