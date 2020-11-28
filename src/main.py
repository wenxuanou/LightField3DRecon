import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    io,color,filters,transform
)

from utils import (
    loadImg,loadMat
)

if __name__ == "__main__":

    # light field image captured by Lytro ILLUM
    # demosaiced light field image, processed by MATLAB file
    imgPath = "../data/lightfield.png"
    # imgPath = "../data/stack.lfp"
    matPath = "../data/lightfield.mat"

    # convert to 4D focal stack

    # subImg = loadImg(imgPath)
    # io.imshow(subImg)
    # plt.show()

    L= loadMat(matPath)      # L(u,v,s,t,c)
    print(L.shape)

