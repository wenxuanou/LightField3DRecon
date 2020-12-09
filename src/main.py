import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    io,color,filters,transform
)

from utils import (
    loadMat,reconImg,edgeConfidence,getR_Horizontal
)

if __name__ == "__main__":

    # light field image captured by Lytro ILLUM
    # demosaiced light field image, processed by MATLAB file
    matPath = "../data/lightfield2.mat"

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

    # horizontal EPI, fix v and t, EPI: U*S*C
    v = int(np.floor(V/2))  # scan vertical center line
    t = int(np.floor(T/2))
    EPI_h = L[:,v,:,t,:]
    EPI_h = np.reshape(EPI_h,(U,S,C))
    # plt.show()

    # vertical EPI, fix u and s, EPI: V*T*C
    u = int(np.floor(U/2))  # scan horizontal center line
    s = int(np.floor(S/2))
    EPI_v = L[u,:,s,:,:]
    EPI_v = np.reshape(EPI_v,(V,T,C))
    # io.imshow(EPI_v)
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

    # sHat = int(np.floor(S/2))   # make sHat the horizontal centerline
    # tHat = int(np.floor(T/2))   # make tHat the vertical centerline
    uHat = int(np.floor(U/2))
    vHat = int(np.floor(V/2))

    # extract index of confident from mask
    uId = np.argwhere(Me_h[uHat,:]>0)    # Me_h: U*S; list_h:[u,s]
    vId = np.argwhere(Me_v[vHat,:]>0)    # Me_v: V*T; list_v:[v,t]
    print(len(uId))

    for d in range(D):
        R_sd = getR_Horizontal(s, d, uHat, EPI_h)
        print(R_sd.shape)

    # R = np.zeros((U,V,S,T,D,C))     # randiance, R: len(uId)*len(vId)*S*T*D*C
    # for d in range(D):
    #     # in every disparity estimation
    #     for s in range(S):
    #         for t in range(T):
    #             for u in uId:
    #                 for v in vId:
    #
    #                     if s+(uHat - u)*d > S:
    #                         R[u, v, s, t, d, :] = L[u, v, int(np.floor(s + (uHat - u) * d)),
    #                                               int(np.floor(t + (vHat - v) * d)), :]
    #
    #                     R[u,v,s,t,d,:] = L[u,v,int(np.floor(s+(uHat - u)*d)),int(np.floor(t+(vHat - v)*d)),:]
    #
    # Rd = R[:,:,:,:,10,:]
    # Rd.reshape(U,V,S,T,C)
    # R_recon = reconImg(Rd)
    # io.imshow(R_recon)
    # plt.imshow()

