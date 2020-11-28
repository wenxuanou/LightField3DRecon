import os
import numpy as np
from skimage import (
    io,color,filters,transform
)
import scipy.io as sio


def loadImg(imgPath):
    # read lytro raw image, return L(u,v,s,t,c)
    img = io.imread(imgPath)
    [H,W,C] = img.shape
    print(img.shape)
    # lenslet span: 12*12
    lenSize = 12
    S = int(np.floor(H / lenSize))
    T = int(np.floor(W / lenSize))

    # img = img[3:,:,:]
    # subImg = img[0:lenSize*1,
    #          0:lenSize*4+int(lenSize/2),
    #          :]
    subImg = img[0:lenSize * 1,
             lenSize * (T-4):W,
             :]

    # # rearrange to L(u,v,s,t,c), each lenslet is 16*16
    # # L: 16*16*400*700*3
    # L = np.zeros((16, 16, S, T, C))
    # for s in range(0,S):
    #     for t in range(0,T):
    #         L[:,:,s,t,:] = img[s*16:(s+1)*16,t*16:(t+1)*16,:]
    # print("generated L")
    # return L      # L(u,v,s,t)

    return subImg

def loadMat(matPath):
    # read MATLAB processed 5D lightfield
    # L(u,v,s,t,c)

    mat = sio.loadmat(matPath)
    L = mat['LF']

    return L
