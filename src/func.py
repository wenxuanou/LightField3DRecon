import numpy as np
import matplotlib.pyplot as plt
from skimage import(
    io
)

def edgeConfidence(EPI,edgeThresh):
    # EPI is with color, E: H,W,C
    [H,W,C] = EPI.shape

    Ce = np.zeros((H,W))  # H*W, edge confidence with fix v and t
    for h in range(H):
        for w in range(W):
            # loop 9 pixel neightborhood
            # scanline always alone the horizontal axis for both EPI_h and EPI_v
            for j in range(-8, 8):
                if w+j > 0 and w+j < W:
                    # compute color intensity difference
                    Ce[h, w] = Ce[h, w] + np.square(np.linalg.norm(EPI[h, w,:] - EPI[h, w+j,:]))
    # compute mask
    Me = Ce > edgeThresh  # H*W
    return Ce,Me

def getR_Horizontal(t,d,vHat,EPI_h):
    # using only Horizontal EPI, not 4D lightfield
    # EPI_h: V*T*C

    [V,T,C] = EPI_h.shape
    R_sd = np.zeros((V,C))

    v = 0
    while t+(vHat - v)*d < T and v < V:   # make sure not over the boundary
        # save pixel intensity alone epipolar line
        R_sd[v, :] = EPI_h[v, t + (vHat - v) * d, :]
        v += 1

    R_sd = R_sd[0:v-1,:]

    return R_sd

def K(x):
    # kernel
    # x is a 3 channel EPI radiance, N*3

    h = 0.02

    normVal = np.linalg.norm(x/h,axis=1)
    mask = normVal < 1

    return np.multiply(mask, 1 - np.square(normVal))    # 1*N


def r_bar_noiseFree(r_bar0,R_td):
    # for noisy EPI
    # r_bar: 1*3, EPI pixel intensity
    # R_td: N*C, N depend on the length of epipolar line in EPI, 0 < N < U

    iterNum = 10   # number of iteration

    [N,C] = R_td.shape
    r_bar = r_bar0

    for iter in range(iterNum):

        # removed for loop
        # K(R_sd[n,:] - r_bar): 1*N
        # R_sd: N*3
        temp = K(R_td - r_bar)
        val1 = np.sum(np.multiply(R_td,K(R_td - r_bar)[:,np.newaxis]),axis=0)    #
        val2 = np.sum(K(R_td - r_bar))

        r_bar = val1 / (val2 + 0.0000001)     # update, avoid NAN

    return r_bar  # 1*3

def refinedConfidence(vHat,Ce, depthScore):
    # refined confidence at fixed uHat
    # Ce: V*T
    # depthScore: T*D

    maxScore = np.amax(depthScore,axis=1)
    avgScore = np.mean(depthScore,axis=1)
    # at vHat
    Cd_vHat = np.multiply(Ce[vHat,:], np.abs(maxScore - avgScore))  # 1*S

    return Cd_vHat

# def depthBilateralMedian(D_uHat,):
#     # remove outliers in depth estimation, fix
    # TODO: depth bilateral median filter


def getDepthScore(vHat,T,D,EPI_h,Me_h):
    # using only horizontal EPI
    # EPI_h = L(u0,:,s0,:,:), fixed u and s, V*T*C
    # Me_h: V*T, only compute depth at Me_h = 1
    depthScore = np.zeros((T, D))  # depthScore: T*D

    for t in range(T):
        for d in range(D):
            # average radiance/color
            r_bar = EPI_h[vHat, t, :]  # EPI_h: V*T*C

            if Me_h[vHat,t] and np.linalg.norm(r_bar)>0:
                R_td = getR_Horizontal(t, d, vHat, EPI_h)  # R: N*C, 0 < N < V
                [N, _] = R_td.shape

                r_bar = r_bar_noiseFree(r_bar, R_td)  # r_bar: 1*3

                # remove looping
                sumVal = K(R_td - r_bar)
                sumVal = np.sum(sumVal) / N

                depthScore[t, d] = sumVal  # divided by the size of R, scalar

        # print("Depth score progress: ",t/T*100,"%")

    return depthScore

