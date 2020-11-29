import numpy as np
import matplotlib.pyplot as plt
from skimage import (
    io,color,filters,transform
)

from utils import (
    loadMat,reconImg
)

if __name__ == "__main__":

    # light field image captured by Lytro ILLUM
    # demosaiced light field image, processed by MATLAB file
    matPath = "../data/lightfield.mat"

    # read 5D focal stack, processed by MATLAB
    L = loadMat(matPath)      # L(u,v,s,t,c)
    [U,V,S,T,C] = L.shape
    print(L.shape)

    imgRecon = reconImg(L)
    io.imshow(imgRecon)
    plt.show()


