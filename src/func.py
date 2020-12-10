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

def getR_Horizontal(s,d,uHat,EPI_h):
    # using only Horizontal EPI, not 4D lightfield
    # EPI_h: U*S*C

    [U,S,C] = EPI_h.shape
    R_sd = np.zeros((U,C))

    u = 0
    while s+(uHat - u)*d < S and u < U:   # make sure not over the boundary
        # save pixel intensity alone epipolar line
        R_sd[u, :] = EPI_h[u, s + (uHat - u) * d, :]
        u += 1

    R_sd = R_sd[0:u-1,:]

    return R_sd

def K(x):
    # kernel
    # x is a 3 channel EPI radiance

    h = 0.02
    if np.linalg.norm(x/h) < 1:
        return 1 - np.linalg.norm(x/h)**2
    else:
        return 0

def r_bar_noiseFree(r_bar0,R_sd):
    # for noisy EPI
    # r_bar: 1*3, EPI pixel intensity
    # R_sd: N*C, N depend on the length of epipolar line in EPI, 0 < N < U

    iterNum = 10   # number of iteration

    [N,C] = R_sd.shape
    r_bar = r_bar0
    val1 = np.zeros_like(r_bar)
    val2 = np.zeros_like(r_bar)
    for iter in range(iterNum):
        for n in range(N):
            val1 = val1 + K(R_sd[n,:] - r_bar) * R_sd[n,:]
            val2 = val2 + K(R_sd[n,:] - r_bar)
        r_bar = val1 / (val2 + 0.0000001)     # update, avoid NAN

    return r_bar