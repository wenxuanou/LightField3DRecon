import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    io,color,filters,transform
)
from utils import (
    loadMat,reconImg
)
from func import (
    edgeConfidence,refinedConfidence,getDepthScore_H,getDepthScore_V
)
import scipy.signal

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

    depthMap_H = np.zeros((S,T))  # depthMap, S*T
    depthMap_V = np.zeros((S,T))
    edgeThresh = 0.02           # edge confidence threshold
    epsilon = 0.1          # refined confident threshold, original epsilon = 0.1, set to 10e-5

    #######################################
    # extract EPI
    # horizontal EPI, fix u, specify s
    u0 = int(np.floor(U/2))  # scan horizontal center line
    v0 = int(np.floor(V/2))

    # iterate every row

    # for s0 in range(S):

    s0 = int(np.floor(S/2))

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
    depthScore_vHat = getDepthScore_H(vHat,T,D,EPI_h,Me_h) # T*D

    # pixel depth estimate
    print("Estimate depth")
    D_vHat = np.argmax(depthScore_vHat,axis=1)   # D(vHat,t): T*1
    # compute refined confidence, only on horizontal, fixed uHat, Cd(vHat, t)
    Cd_vHat = refinedConfidence(vHat,Ce_h,depthScore_vHat)  # T*1

    # depth propagation
    # threshold confident depth
    D_vHat = np.multiply(D_vHat, Cd_vHat > epsilon)    # confidence all too small, scaled down epsilon


    # filling up
    if np.count_nonzero(D_vHat) > 0:
        ids = np.argwhere(D_vHat>0)                    # propagate non-zero depth, confident, D_vHat: T*1
        for id in ids:
            r_hat = EPI_h[vHat,id,:]                   # EPI ray of confident locations, r_bar: 1*3
            if np.linalg.norm(r_hat) > 1:
                EPI_h_slice = EPI_h[vHat,:,:]              # EPI_h_slice: T*3

                cond1 = np.linalg.norm(r_hat - EPI_h_slice,axis=1) < epsilon           # similar radiance/color
                cond2 = D_vHat < D_vHat[id]
                # D_vHat[cond1] = D_vHat[id]              # raise lower depth
                D_vHat[np.logical_and(cond1,cond2)] = D_vHat[id]              # raise lower depth

    if s0 == int(np.floor(S/2)):
        plt.figure(1)
        io.imshow(EPI_h)
        plt.figure(2)
        plt.plot(Cd_vHat)
        plt.figure(3)
        plt.plot(D_vHat)
        plt.figure(4)
        plt.plot(depthScore_vHat[int(T/2)])
        # io.imshow(depthScore_vHat.T)
        plt.show()
        print("here")

    depthMap_H[s0,:] = D_vHat
    print("Progress horizontal: ", s0/S*100,"% ########################")

    for t0 in range(T):
        EPI_v = L[:, v0, :, t0, :]
        EPI_v = np.reshape(EPI_v, (U, S, C))  # U*S*C

        #######################################
        # EPI edge confidence
        Ce_v, Me_v = edgeConfidence(EPI_v, edgeThresh)

        #######################################
        # depth computation
        # only compute depth at where Me = 1

        # range of disparity
        D = 20  # 0 to 20
        print("Computing depth score")

        # initialize
        uHat = int(np.floor(U / 2))

        depthScore_uHat = getDepthScore_V(uHat, S, D, EPI_v, Me_v)

        # pixel depth estimate
        print("Estimate depth")

        D_uHat = np.argmax(depthScore_uHat, axis=1)
        Cd_uHat = refinedConfidence(uHat, Ce_v, depthScore_uHat)

        # depth propagation
        # threshold confident depth
        D_uHat = np.multiply(D_uHat, Cd_uHat > epsilon)

        if np.count_nonzero(D_uHat) > 0:
            ids = np.argwhere(D_uHat > 0)
            for id in ids:
                r_hat = EPI_v[uHat, id, :]
                if np.linalg.norm(r_hat) > 1:
                    EPI_v_slice = EPI_v[uHat, :, :]

                    cond1 = np.linalg.norm(r_hat - EPI_v_slice, axis=1) < epsilon  # similar radiance/color
                    cond2 = D_uHat < D_uHat[id]
                    D_uHat[np.logical_and(cond1, cond2)] = D_uHat[id]  # raise lower depth

        depthMap_V[:, t0] = D_uHat
        print("Progress vertical: ", t0 / T * 100, "% ########################")


    depthMap = (depthMap_H + depthMap_V)/2

    print("Depth estimate finished")


    # # bilateral median
    # print("Bilateral filtering")
    # for s in range(S):
    #     for t in range(T):
    #         winSize = 8    # window size, 5+5+1 -> 11*11
    #         depthBuff = np.array([])
    #         # loop in window
    #         for ss in range(s-winSize,s+winSize):
    #             for tt in range(t-winSize,t+winSize):
    #                 if ss>0 and ss<S and tt>0 and tt<T:
    #                     if depthMap[ss,tt] > 0:
    #                         np.append(depthBuff,depthMap[ss,tt])
    #         if depthBuff.size > 0:
    #             depthMap[s,t] = np.median(depthBuff)
    #     print("Progress: ", s / S * 100, "% ########################")


    # save depth map image
    io.imsave("depthMapGuide.png", depthMap)

    # save depth map as matrix
    ext_out = {"depthMap": depthMap}        # all T*3
    np.savez("depthMapGuide.npz", **ext_out)
    print("Depth Map stored")

    io.imshow(depthMap)
    plt.show()



