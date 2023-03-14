# -*- coding: utf-8 -*-
"""
Created on Mon May 23 01:40:42 2022

@author: Raiyan
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Raiyan/Desktop/myout/lena.png"

img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

rows = img.shape[0]
cols = img.shape[1]
MN = rows * cols
L = 256

def show_img(image):
     plt.imshow(image, 'gray')
     plt.title('input img')
     plt.show()

def round_off(x):
    return round( x * (float(L)-1) )   

# flatten, , range
def show_hist(image):
     plt.hist(image.ravel(), 256, [0, 256])
     plt.title('input histogram')
     plt.show()

# x = np.array([[0, 2, 1, 3, 4],
#               [1, 3, 4, 3, 3],
#               [0, 1, 3, 1, 4],
#               [3, 1, 4, 2, 0],
#               [0, 4, 2, 4, 4]], np.float32)
# x = np.asarray( img.copy() )

show_img(img)
show_hist(img)

freq = cdf = pdf = np.zeros(256,np.float32)

for i in range(rows):
    for j in range(cols):
        freq[ int(img[i][j]) ] += 1

# for i in range( L ):
    # pdf[i] = freq[i] / MN
    
pdf = freq / MN

# cdf[0] = pdf[0]
# for i in range(1, L ):
    # cdf[i] = cdf[i-1]+pdf[i]
# cdf

# rounding
# for i in range( L ):
#     cdf[i] = round_off( cdf[i] )

cdf = pdf.cumsum()
cdf = cdf * ( float(L)-1 )
f_cdf = np.rint(cdf)

for i in range(rows):
    for j in range(cols):
        img[i][j] = f_cdf[ int( round(img[i][j]) ) ]

show_img(img)
show_hist(img)
