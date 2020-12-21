from skimage import(
    io
)
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

if __name__ == "__main__":

    dataName = "depthMapGuide.npz"
    Data = np.load(dataName, allow_pickle=False)
    depthMap = Data['depthMap']

    [S,T] = depthMap.shape
    # depthMap = scipy.ndimage.gaussian_filter(depthMap,sigma=,mode="mirror")
    depthMapNew = np.zeros_like(depthMap)
    # depthMapNew = depthMap


    # bilateral median
    print("Bilateral filtering")
    for s in range(S):
        for t in range(T):
            winSize = 1  # window size, 5+5+1 -> 11*11
            depthBuff = np.array([])
            # loop in window

            neighbourMask = depthMap[s-winSize:s+winSize,t-winSize:t+winSize] > 0
            nonZeroNum = np.count_nonzero(neighbourMask)
            if nonZeroNum > 1:
                # print("found")
                neighbourVal = np.multiply(neighbourMask, depthMap[s-winSize:s+winSize,t-winSize:t+winSize])
                depthMapNew[s, t] = np.sum(neighbourVal) / nonZeroNum
                # depthMapNew[s, t] = np.median(neighbourVal[neighbourVal>0])


            # for ss in range(s - winSize, s + winSize):
            #     for tt in range(t - winSize, t + winSize):
            #         if ss > 0 and ss < S and tt > 0 and tt < T:
            #             if depthMap[ss, tt] > 0:
            #                 np.append(depthBuff, depthMap[ss, tt])
            # if depthBuff.size > 0:
            #     depthMap[s, t] = np.mean(depthBuff)
            #     print(np.mean(depthBuff))
        print("Progress: ", s / S * 100, "% ########################")

    id = np.argwhere(depthMapNew>0)    # only display depth above 0
    y = id[:,0]
    x = id[:,1]
    io.imsave("depthMapGuideProc.png",depthMapNew)
    plt.figure(1)
    io.imshow(depthMapNew)
    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.scatter(x, depthMapNew[y,x], -y, s=0.01)
    plt.show()