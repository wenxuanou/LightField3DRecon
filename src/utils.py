import numpy as np
import scipy.io as sio

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

