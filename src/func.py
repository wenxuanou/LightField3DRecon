import numpy as np
from skimage import (
    io,color,filters,transform
)

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

    # if np.linalg.norm(x/h,axis=1) < 1:
    #     return 1 - np.linalg.norm(x/h,axis=1)**2
    # else:
    #     return 0


def r_bar_noiseFree(r_bar0,R_td):
    # for noisy EPI
    # r_bar: 1*3, EPI pixel intensity
    # R_td: N*C, N depend on the length of epipolar line in EPI, 0 < N < U

    iterNum = 10   # number of iteration

    [N,C] = R_td.shape
    r_bar = r_bar0
    val1 = np.zeros_like(r_bar)
    val2 = np.zeros_like(r_bar)

    for iter in range(iterNum):
        # for n in range(N):
        #     val1 = val1 + K(R_sd[n,:] - r_bar) * R_sd[n,:]
        #     val2 = val2 + K(R_sd[n,:] - r_bar)

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

    print(Cd_vHat.shape)

    return Cd_vHat

# def depthBilateralMedian(D_uHat,):
#     # remove outliers in depth estimation, fix
    # TODO: depth bilateral median filter


def getDepthScore(vHat,T,D,EPI_h):
    # using only horizontal EPI
    # EPI_h = L(u0,:,s0,:,:), fixed u and s, V*T*C
    depthScore = np.zeros((T, D))  # depthScore: T*D
    for d in range(D):
        for t in range(T):
            R_td = getR_Horizontal(t, d, vHat, EPI_h)  # R: N*C, 0 < N < V
            [N, _] = R_td.shape

            r_bar = EPI_h[vHat, t, :]  # EPI_h: V*T*C
            r_bar = r_bar_noiseFree(r_bar, R_td)  # r_bar: 1*3

            # sumVal = 0
            # for n in range(N):
            #     sumVal = sumVal + K(R_sd[n,:] - r_bar)

            # remove looping
            sumVal = K(R_td - r_bar)
            sumVal = np.sum(sumVal)

            depthScore[t, d] = sumVal / N  # divided by the size of R, scalar

        print(d, "/", D)

    # plt.matshow(depthScore)
    # plt.show()

    return depthScore
