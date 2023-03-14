# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:44:51 2022

@author: Raiyan
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Raiyan/Desktop/input.jpeg"

img = cv.imread(path)

plt.hist(img.ravel(),256,(0,256))

plt.show()

plt.imshow(img)

plt.show()


op = img

for h in range(img.shape[0]):
    freq = np.zeros(256,np.int32)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            pix = img[h][i][j]
            freq[pix]+=1
    pdf = np.zeros(256,np.float32)
    for i in range(256):
        pdf[i] = freq[i]/(img.shape[1]*img.shape[2])
    cdf = np.zeros(256,np.float32)
    cdf[0]=pdf[0]
    for i in range(1,256):
        cdf[i] = cdf[i-1]+pdf[i]
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            pix = img[h][i][j]
            m = cdf[pix]
            op[h][i][j] = 255*m
            
plt.hist(op.ravel(),256,(0,256))
plt.show()

plt.imshow(op)
plt.show()

