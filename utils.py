#!/usr/bin/env python3

import numpy as np
import os
import sys
import time
import shutil
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.misc import imresize


def cropCenter(img, cropx, cropy):
    sh = img.shape
    x = sh[0]
    y = sh[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[startx:startx+cropx, starty:starty+cropy, :]


def createBatch(images, path):
    root = path
    resizeSize = 64

    imgList = os.listdir(root)
    numImg = len(imgList)

    dataX = np.empty((len(images), 64, 64, 3))
    dataY = np.empty((len(images), 64, 64, 3))

    areaX = 32  # num of occluder pixels (x-axis)
    areaY = 32  # num of occluder pixels (y-axis)
    spX = 16
    spY = 16
    n = 0
    for k in images:
        img = plt.imread(root + imgList[k])
        new = cropCenter(img, 96, 96)
        new = imresize(new, (resizeSize, resizeSize), interp='bicubic')
        new = new.astype(float)
        newImg = np.copy(new)
        dataY[n] = newImg
        for i in range(areaX):
            for j in range(areaY):
                new[spX+i, spY+j, :] = 0.
        newImg = np.copy(new)
        dataX[n] = newImg
        n = n + 1
    return dataX, dataY


def createBatchSpec(path):
    root = path
    resizeSize = 64

    imgList = os.listdir(root)
    numImg = len(imgList)
    images = np.arange(numImg)

    dataX = np.empty((len(images), 64, 64, 3))
    dataY = np.empty((len(images), 64, 64, 3))

    areaX = 32  # num of occluder pixels (x-axis)
    areaY = 32  # num of occluder pixels (y-axis)
    spX = 16
    spY = 16
    n = 0
    for k in images:
        img = plt.imread(root + imgList[k])
        new = img.astype(float)
        newImg = np.copy(new)
        dataY[n] = newImg
        for i in range(areaX):
            for j in range(areaY):
                new[spX+i, spY+j, :] = 0.
        newImg = np.copy(new)
        dataX[n] = newImg
        n = n + 1
    return dataX, dataY, numImg


def saveImgs(xs, ys, trueY, save, colWidth=10):
    nImgs = xs.shape[0]
    assert(nImgs == ys.shape[0])

    if not os.path.exists(save):
        os.makedirs(save)

    fnames = []
    for i in range(nImgs):
        xy = np.clip(np.squeeze(np.concatenate([trueY[i], xs[i], ys[i]], axis=1)), 0., 255.)
        fname = "{}/{:04d}.jpg".format(save, i)
        plt.imsave(fname, xy/255.)
        fnames.append(fname)

    os.system('montage -geometry +0+0 -tile {}x {} {}.png'.format(
        colWidth, ' '.join(fnames), save))
    shutil.rmtree(save)
