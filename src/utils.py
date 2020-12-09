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

def edgeConfidence(EPI,edgeThresh):
    # EPI is with color, E: H,W,C
    [H,W,C] = EPI.shape

    Ce = np.zeros((H,W))  # H*W, edge confidence with fix v and t
    for h in range(H):
        for w in range(W):
            # loop 9 pixel neightborhood
            # scanline always alone the horizontal axis for both EPI_h and EPI_v
            for j in range(-1, 1):
                if w+j > 0 and w+j < W:
                    # compute color intensity difference
                    Ce[h, w] = Ce[h, w] + np.square(np.linalg.norm(EPI[h, w,:] - EPI[h, w+j,:]))
    # compute mask
    Me = Ce > edgeThresh  # H*W
    return Ce,Me

def getR_Horizontal(s,d,uHat,EPI_h):
    # using only Horizontal EPI, not 4D lightfield
    # EPI_h: U*S*C

    [U,S,C] = EPI_h.shape
    R_sd = np.zeros((U,C))
    for u in range(U):
        R_sd[u,:] = EPI_h[u,s+(uHat - u)*d,:]

    return R_sd