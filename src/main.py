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
    matPath = "../data/guide.mat"

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

    # vertical EPI, fix v and t, EPI: U*S*C
    v0 = int(np.floor(V/2))  # scan vertical center line
    t0 = int(np.floor(T/2))
    EPI_v = L[:,v0,:,t0,:]
    EPI_v = np.reshape(EPI_v,(U,S,C))
    # io.imshow(EPI_v)
    # plt.show()

    # horizontal EPI, fix u and s, EPI: V*T*C
    u0 = int(np.floor(U/2))  # scan horizontal center line
    s0 = int(np.floor(S/2))
    EPI_h = L[u0,:,s0,:,:]
    EPI_h = np.reshape(EPI_h,(V,T,C))
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
    # compute depth score
    depthScore_vHat = getDepthScore(vHat,T,D,EPI_h)

    # pixel depth estimate
    D_vHat = np.argmax(depthScore_vHat,axis=1)   # D(vHat,t): T*1
    # print(D_vHat)

    # compute refined confidence, only on horizontal, fixed uHat, Cd(vHat, t)
    Cd_vHat = refinedConfidence(vHat,Ce_h,depthScore_vHat)  # T*1
    # print(Cd_vHat.shape)

    # bilateral mediant filter on Depth estimate
    # TODO: need to build this median filter


    # Store confident depth estimation as 3D rays
    Gamma = np.zeros((D,V,T,3))         # Gamma(d,vHat,t,r_bar.T): D,u,s,3
    epsilon = 0.01                       # same as the epsilon used in bilateral filter, original epsilon = 0.1
    processed = Cd_vHat > epsilon       # masks
