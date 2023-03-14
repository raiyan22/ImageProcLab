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

def show_img(image, title):
     plt.imshow(image, 'gray')
     plt.title(f'{title} image')
     plt.show()

def round_off(x):
    return round( x * (float(L)-1) )   

# flatten, , range
def show_hist(image, title):
     plt.hist(image.ravel(), L, [0, L])
     plt.title(f'{title} histogram')
     plt.show()

show_img(img, "input")
show_hist(img, "input img")

freq = cdf = pdf = np.zeros(L,np.float32)

for i in range(rows):
    for j in range(cols):
        freq[ int(img[i][j]) ] += 1
    
pdf = freq / MN

cdf = pdf.cumsum()
cdf = cdf * ( float(L)-1 )

# rounding off each cdf vals
f_cdf = np.rint(cdf)

for i in range(rows):
    for j in range(cols):
        img[i][j] = f_cdf[ int( round(img[i][j]) ) ]

show_img(img, "output")
show_hist(img, "output img")






