import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util,transform
)
from scipy import (
    interpolate,signal
)
from utils import(
    loadMat
)


# generate focal stack
def focalStack(L,d):
    # L: light field image, L(u,v,s,t,c)
    # d: vector of depth parameter, 0 ~ some integer

    # output focal stack: I(s,t,c,d)

    # initialize
    [U,V,S,T,C] = L.shape           # 16*16*400*700*3
    D = len(d)
    I = np.zeros((S,T,C,D))         # 400*700*3*D

    # compute focal stack
    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    u = np.arange(lensletSize) - maxUV
    v = np.arange(lensletSize) - maxUV

    print("generating focal stack")
    for count in range(0,D):
        # intergrate
        sumVal = np.zeros((S,T,C))
        print("countD: ",count," / ",D)
        for uu in u:
            for vv in v:
                imgTemp0 = L[int(uu+maxUV),int(vv+maxUV),:,:,0]     # 400*700, S*T
                imgTemp1 = L[int(uu+maxUV),int(vv+maxUV),:,:,1]
                imgTemp2 = L[int(uu+maxUV),int(vv+maxUV),:,:,2]

                L0 = interpolate.interp2d(np.arange(0,T),np.arange(0,S),imgTemp0,kind='cubic')  # S is height, T is Width
                L1 = interpolate.interp2d(np.arange(0,T),np.arange(0,S),imgTemp1,kind='cubic')
                L2 = interpolate.interp2d(np.arange(0,T),np.arange(0,S),imgTemp2,kind='cubic')

                snew = np.arange(d[count]*uu,S+d[count]*uu,1)
                tnew = np.arange(-d[count]*vv,T-d[count]*vv,1)

                sumVal[:,:,0] = sumVal[:,:,0] + L0(tnew, snew)
                sumVal[:,:,1] = sumVal[:,:,1] + L1(tnew, snew)
                sumVal[:,:,2] = sumVal[:,:,2] + L2(tnew, snew)

        I[:,:,:,count] = sumVal / (len(u)*len(v))
        I[:,:,:,count] = np.clip(I[:,:,:,count],0,255)

    return I

# inverse gamma to get linear image
def invGamma(img):

    mask1 = img <= 0.0404482
    mask2 = img > 0.0404482
    temp1 = img / 12.92

    temp2 = np.power(((img + 0.055) / 1.055), 2.4)

    img_linear = np.multiply(temp1,mask1) + np.multiply(temp2,mask2)

    return img_linear

# obtain luminance from focal stack
def get_luminance(I):
    # I: focal stack, I(s,t,c,d)

    [S,T,C,D] = I.shape

    I_luminance = np.zeros((S,T,D))
    for countD in range(0,D):
        Id_linear = invGamma(I[:,:,:,countD])       # S*T*C
        Id_xyz = color.rgb2xyz(Id_linear)           # S*T*C, second channel is luminance
        I_luminance[:,:,countD] = Id_xyz[:,:,1]     # extract y channel

    return I_luminance

def get_luminance_AFI(I_AFI):
    # I_AFT: focal aperture stack, I_AFT(s,t,c,alpha,f)

    [S,T,C,alpha_Num,f_Num] = I_AFI.shape
    I_AFI_luminance = np.zeros((S,T,alpha_Num,f_Num))
    for countA in range(0,alpha_Num):
        for countF in range(0,f_Num):
            I_AFI_linear = invGamma(I_AFI[:, :, :, countA,countF])  # S*T*C
            I_AFI_xyz = color.rgb2xyz(I_AFI_linear)
            I_AFI_luminance[:,:,countA,countF] = I_AFI_xyz[:,:,1]   # extract y channel

    return I_AFI_luminance

# generate gaussian kernel
def gauss_kernel(sigma,size):
    # sigma: gaussian std
    # size: kernel size, size*size

    f = np.zeros((size, size))
    f[int(np.round(size / 2)), int(np.round(size / 2))] = 1
    f = filters.gaussian(f, sigma=sigma)  # generate gaussian kernel

    return f

# get low and high frequency components
def get_Freq(I_luminance,sigma1):
    # I_luminance: focal stack luminance, I(s,t,d)
    # sigma1: gaussian kernel std

    [S,T,D] = I_luminance.shape
    f1 = gauss_kernel(sigma1,6*sigma1)
    print("separating frequency component")

    I_lowFreq = np.zeros((S,T,D))
    for countD in range(0,D):
        I_lowFreq[:,:,countD] = signal.convolve2d(I_luminance[:,:,countD],f1,mode='same')
        print("countD: ", countD, " / ", D)

    I_highFreq = I_luminance - I_lowFreq

    return I_lowFreq, I_highFreq

def get_Freq_AFI(I_AFI_luminance,sigma1):
    # I_AFI_luminance: focal-aperture stack luminance, S*T*alpha_Num*f_Num

    [S,T,alpha_Num,f_Num] = I_AFI_luminance.shape
    f1 = gauss_kernel(sigma1, 6 * sigma1)
    print("separating frequency component")

    I_AFI_lowFreq = np.zeros((S, T, alpha_Num, f_Num))
    for countA in range(0,alpha_Num):
        for countF in range(0,f_Num):
            I_AFI_lowFreq[:,:,countA,countF] = signal.convolve2d(I_AFI_luminance[:,:,countA,countF],f1,mode='same')
            print("countD: ", countF, " / ", f_Num)
        print("countA: ", countA, " / ", alpha_Num)

    I_AFI_highFreq = I_AFI_luminance - I_AFI_lowFreq
    return I_AFI_lowFreq, I_AFI_highFreq

# sharpness weight
def weight_sharp(I_highFreq,sigma2):
    # I_highFreq: focal stack high frequency component
    # sigma2: gaussian filter std

    [S, T, D] = I_highFreq.shape
    f2 = gauss_kernel(sigma2,6*sigma2)

    w_sharp = np.zeros((S,T,D))
    for countD in range(0,D):
        w_sharp[:,:,countD] = signal.convolve2d(np.square(I_highFreq[:,:,countD]),f2,mode='same')

    return w_sharp

def weight_sharp_AFI(I_AFI_highFreq,sigma2):
    # I_AFI_highFreq: S*T*alpha_Num*f_Num

    [S,T,alpha_Num,f_Num] = I_AFI_highFreq.shape
    f2 = gauss_kernel(sigma2,6*sigma2)

    w_AFI_sharp = np.zeros((S,T,alpha_Num,f_Num))
    for countA in range(0,alpha_Num):
        for countF in range(0,f_Num):
            w_AFI_sharp[:,:,countA,countF] = signal.convolve2d(np.square(I_AFI_highFreq[:,:,countA,countF]),f2,mode='same')

    return w_AFI_sharp



# all-focus image
def allFocus(I,w_sharp):
    # I: focal stack
    # w_sharp: sharpness weight

    [S,T,C,D] = I.shape
    I_allFocus = np.zeros((S,T,C))

    w_sum = np.sum(w_sharp,axis=2)     # S*T

    for chan in range(0,C):
        I_chan = I[:,:,chan,:]          # S*T*D

        I_allFocus[:,:,chan] = np.divide( np.sum( np.multiply(w_sharp,I_chan),axis=2),
                               w_sum)

    return I_allFocus

# depth map
def depthMap(w_sharp,d):
    # w_sharp: sharpness weight, w(s,t,d)

    [S,T,D] = w_sharp.shape
    w_sum = np.sum(w_sharp, axis=2)

    sumVal = np.zeros((S,T))
    for countD in range(0,D):
        sumVal = sumVal + w_sharp[:,:,countD] * d[countD]

    I_depth = sumVal / w_sum
    return I_depth

def depthMap_AFI(w_AFI_sharp,f):
    # w_sharp_AFI: S*T*alpha_Num*f_Num

    [S,T,alpha_Num,f_Num] = w_AFI_sharp.shape
    w_sum = np.sum(w_AFI_sharp, axis=(2,3))

    sumVal = np.zeros((S,T))
    for countA in range(0,alpha_Num):
        for countF in range(0,f_Num):
            sumVal = sumVal + w_AFI_sharp[:,:,countA,countF] * f[countF]

    I_AFI_depth = sumVal / w_sum
    return I_AFI_depth

# gerenate AFI
def afi(L,alpha,f):
    # L: U*V*S*T*C
    # alpha: alpha_Num*1
    # f: f_Num*1

    alpha_Num = len(alpha)
    f_Num = len(f)
    [U, V, S, T, C] = L.shape   # 16*16*400*700*3

    I_AFI = np.zeros((S,T,C,alpha_Num,f_Num))   # S*T*C*alpha_Num*f_Num

    for countF in range(0,f_Num):
        print("countF: ", countF, " / ", f_Num)
        for countA in range(0,alpha_Num):
            aperatureSize = alpha[countA]
            maxUV = (aperatureSize - 1) / 2
            u = np.arange(1,aperatureSize) - maxUV - 1
            v = np.arange(1,aperatureSize) - maxUV - 1

            print("countA: ", countA, " / ", alpha_Num)
            sumVal = np.zeros((S, T, C))
            for uu in u:
                for vv in v:
                    imgTemp0 = L[int(uu + maxUV), int(vv + maxUV), :, :, 0]  # 400*700, S*T
                    imgTemp1 = L[int(uu + maxUV), int(vv + maxUV), :, :, 1]
                    imgTemp2 = L[int(uu + maxUV), int(vv + maxUV), :, :, 2]

                    L0 = interpolate.interp2d(np.arange(0, T), np.arange(0, S), imgTemp0, kind='cubic')  # S is height, T is Width
                    L1 = interpolate.interp2d(np.arange(0, T), np.arange(0, S), imgTemp1, kind='cubic')
                    L2 = interpolate.interp2d(np.arange(0, T), np.arange(0, S), imgTemp2, kind='cubic')

                    snew = np.arange(f[countF] * uu, S + f[countF] * uu, 1)
                    tnew = np.arange(-f[countF] * vv, T - f[countF] * vv, 1)

                    sumVal[:, :, 0] = sumVal[:, :, 0] + L0(tnew, snew)
                    sumVal[:, :, 1] = sumVal[:, :, 1] + L1(tnew, snew)
                    sumVal[:, :, 2] = sumVal[:, :, 2] + L2(tnew, snew)

            # I_AFI[:,:,:,countA,countF] = sumVal / (len(u)*len(v))
            # test1 = I_AFI[:, :, :, countA, countF]
            # I_AFI[:, :, :, countA, countF] = I_AFI[:,:,:,countA,countF] / (aperatureSize)            # normalize by size of aperature
            I_AFI[:,:,:,countA,countF] = sumVal / (aperatureSize**2)
            # test2 = I_AFI[:, :, :, countA, countF]
            I_AFI[:,:,:,countA,countF] = np.clip(I_AFI[:,:,:,countA,countF],0,255)

    return I_AFI


if __name__ == "__main__":
    # load light field image
    matPath = "../data/lightfield2.mat"
    L = loadMat(matPath)

    # generate AFI
    f =  np.arange(-1,1,0.25)       # different focal setting
    alpha = [2,4,6,8,10,12,16]     # aperture size, sigma for kernel
    I_AFI = afi(L,alpha,f)         # I_AFI(s,t,c,alpha,f)

    alpha_Num = len(alpha)
    f_Num = len(f)

    # generate AFT depth map
    # initialize Gaussian kernels
    sigma1 = 3
    sigma2 = 3

    # get luminance of focal-aperture stack
    print("getting luminance")
    I_AFI_luminance = get_luminance_AFI(I_AFI)
    # separate frequency component
    I_AFI_lowFreq, I_AFI_highFreq = get_Freq_AFI(I_AFI_luminance,sigma1)
    # sharpness weight
    w_AFI_sharp = weight_sharp_AFI(I_AFI_highFreq,sigma2)

    # generate depth map
    print("generating depth map")
    I_AFI_depth = depthMap_AFI(w_AFI_sharp,f)
    io.imsave("depth_map_AFI.png",I_AFI_depth)