import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    io,color,filters,transform
)

from utils import (
    loadMat,reconImg
)
from func import (
    edgeConfidence,getR_Horizontal,K,r_bar_noiseFree
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

    uHat = int(np.floor(U/2))     # make uHat the horizontal centerline
    vHat = int(np.floor(V/2))     # make vHat the horizontal centerline

    # using only horizontal EPI for now
    # compute depth score
    depthScore = np.zeros((S,D))  # depthScore: S*D*C
    for d in range(D):
        for s in range(S):
            R_sd = getR_Horizontal(s, d, uHat, EPI_h)   # R: N*C, 0 < N < U
            [N,_] = R_sd.shape

            r_bar = EPI_h[uHat,s,:]     # EPI_h: U*S*C
            r_bar = r_bar_noiseFree(r_bar,R_sd)     # r_bar: 1*3

            sumVal = 0
            for n in range(N):
                sumVal = sumVal + K(R_sd[n,:] - r_bar)

            depthScore[s,d] = sumVal / N              # divided by the size of R, scalar

    print(depthScore.shape)

    # pixel depth estimate
    D_uHat = np.argmax(depthScore,axis=1)   # D(uHat,s): S*1
    print(D_uHat)