import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    io,color,filters,transform
)

from utils import (
    loadMat,reconImg
)
from func import (
    edgeConfidence,getR_Horizontal,K,r_bar_noiseFree,refinedConfidence,getDepthScore
)

if __name__ == "__main__":

    # light field image captured by Lytro ILLUM
    # demosaiced light field image, processed by MATLAB file
    # matPath = "../data/lightfield2.mat"
    matPath = "../data/tower.mat"

    # read 5D focal stack, processed by MATLAB
    L = loadMat(matPath)      # L(u,v,s,t,c)
    [U,V,S,T,C] = L.shape
    print("L shape: ", L.shape)

    # reconstruct image from focal stack
    # imgRecon = reconImg(L)
    # io.imshow(imgRecon)
    # plt.show()

    #######################################
    # extract EPI

    # vertical EPI, fix v and t
    v0 = int(np.floor(V/2))  # scan vertical center line
    t0 = int(np.floor(T/2))
    EPI_v = L[:,v0,:,t0,:]
    EPI_v = np.reshape(EPI_v,(U,S,C))   # U*S*C
    # io.imshow(EPI_v)
    # plt.show()

    # horizontal EPI, fix u and s
    u0 = int(np.floor(U/2))  # scan horizontal center line
    s0 = int(np.floor(S/2))
    EPI_h = L[u0,:,s0,:,:]
    EPI_h = np.reshape(EPI_h,(V,T,C))   # V*T*C
    # io.imshow(EPI_h)
    # plt.show()

    #######################################
    # EPI edge confidence
    edgeThresh = 0.02

    # horizontal
    # plt.figure(1)
    # io.imshow(EPI_h)

    Ce_h,Me_h = edgeConfidence(EPI_h,edgeThresh)   # Ce_H: U*S; Me_h: U*S
    # plt.figure(2)
    # io.imshow(Me_h)
    # plt.show()

    # vertical
    # plt.figure(1)
    # io.imshow(EPI_v)

    Ce_v, Me_v = edgeConfidence(EPI_v, edgeThresh) # Ce_H: U*S; Me_h: U*S
    # plt.figure(2)
    # io.imshow(Me_v)
    # plt.show()

    #######################################
    # depth computation
    # only compute depth at where Me = 1

    # range of disparity
    D = 20      # 0 to 20
    print("Computing depth score")

    # initialize
    uHat = int(np.floor(U/2))     # make uHat the horizontal centerline
    vHat = int(np.floor(V/2))     # make vHat the horizontal centerline

    # using only horizontal EPI for now
    # EPI_h = L(u0,:,s0,:,:), fixed u and s, V*T*C
    # compute depth score along vHat
    depthScore_vHat = getDepthScore(vHat,T,D,EPI_h) # T*D

    # pixel depth estimate
    D_vHat = np.argmax(depthScore_vHat,axis=1)   # D(vHat,t): T*1
    # print(D_vHat)

    # compute refined confidence, only on horizontal, fixed uHat, Cd(vHat, t)
    Cd_vHat = refinedConfidence(vHat,Ce_h,depthScore_vHat)  # T*1
    # print(Cd_vHat.shape)

    # bilateral mediant filter on Depth estimate
    # TODO: need to build this median filter
    epsilon = 10 ** (-5)  # confident threshold, original epsilon = 0.1

    # depth propagation
    # propagation only confident depth
    temp = Cd_vHat * (10 ** 6) > epsilon
    D_vHat = np.multiply(D_vHat, Cd_vHat > epsilon)    # confidence all too small, scaled down epsilon
    # assign depth alone slope
    depthEPI = np.zeros((V,T))  # V*T
    depthEPI[vHat,:] = D_vHat
