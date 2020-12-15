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

    depthMap = np.zeros((S,T))  # depthMap, S*T
    edgeThresh = 0.02           # edge confidence threshold
    epsilon = 10 ** (-5)        # refined confident threshold, original epsilon = 0.1

    #######################################
    # extract EPI
    # horizontal EPI, fix u, specify s
    u0 = int(np.floor(U/2))  # scan horizontal center line

    # iterate every row
    for s0 in range(S):
        EPI_h = L[u0,:,s0,:,:]
        EPI_h = np.reshape(EPI_h,(V,T,C))   # V*T*C
        # io.imshow(EPI_h)
        # plt.show()

        #######################################
        # EPI edge confidence
        Ce_h,Me_h = edgeConfidence(EPI_h,edgeThresh)   # Ce_H: V*T; Me_h: V*T

        #######################################
        # depth computation
        # only compute depth at where Me = 1

        # range of disparity
        D = 20      # 0 to 20
        print("Computing depth score")

        # initialize
        vHat = int(np.floor(V/2))     # make vHat the horizontal centerline

        # using only horizontal EPI for now
        # EPI_h = L(u0,:,s0,:,:), fixed u and s, V*T*C
        # compute depth score along vHat
        depthScore_vHat = getDepthScore(vHat,T,D,EPI_h,Me_h) # T*D

        # pixel depth estimate
        print("Estimate depth")
        D_vHat = np.argmax(depthScore_vHat,axis=1)   # D(vHat,t): T*1
        # compute refined confidence, only on horizontal, fixed uHat, Cd(vHat, t)
        Cd_vHat = refinedConfidence(vHat,Ce_h,depthScore_vHat)  # T*1

        # bilateral mediant filter on Depth estimate
        # TODO: need to build this median filter

        # depth propagation
        # propagation only confident depth
        temp = Cd_vHat * (10 ** 6) > epsilon
        D_vHat = np.multiply(D_vHat, Cd_vHat > epsilon)    # confidence all too small, scaled down epsilon

        # plt.plot(D_vHat)
        # plt.show()

        depthMap[s0,:] = D_vHat

        print("Progress: ", s0/S*100,"% ########################")

    # save depth map image
    io.imsave("depthMapTower.png", depthMap)

    # save depth map as matrix
    ext_out = {"depthMap": depthMap}        # all T*3
    np.savez("depthMap.npz", **ext_out)
    print("Depth Map stored")






