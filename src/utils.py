import os
import numpy as np
from skimage import (
    io,color,filters,transform
)
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

def edgeConfidence(EPI_gray,edgeThresh):
    [H,W] = EPI_gray.shape

    Ce = np.zeros_like(EPI_gray)  # H*W, edge confidence with fix v and t
    for h in range(H):
        for w in range(W):
            # loop 9 pixel neightborhood
            # scanline always alone the horizontal axis for both EPI_h and EPI_v
            for j in range(-3, 3):
                if w+j > 0 and w+j < W:
                    Ce[h, w] = Ce[h, w] + np.square(EPI_gray[h, w] - EPI_gray[h, w+j])
    # compute mask
    Me = Ce > edgeThresh  # H*W
    return Ce,Me

