import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "C:/Users/Raiyan/Desktop/myout/bfly.png"

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

def fq_count_of_img():
    freq_op = np.zeros(256)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pix = int(img[i][j])
            freq_op[pix]+=1
    
    plt.plot(freq_op)
    plt.title("Plot of intensity frequencies of output image:")
    plt.show()
     
# flatten, , range
def show_hist(image, title):
     plt.hist(image.ravel(), 256, [0, 256])
     plt.title(f'{title} histogram')
     plt.show()
     
def make_gauss(mean, sigma):
    
    gauss = np.zeros(256, np.float32)
    for i in range(256):
        temp = math.exp( -1.0 * (i - mean) * (i - mean) / (sigma * sigma ) )
        temp2 =  1.0 / ( math.sqrt( 2 * np.pi) * sigma )
        gauss[i] = temp * temp2
        
    return gauss

show_img(img, "input")
show_hist(img, "input")

gauss_1 = make_gauss(80.0, 20.0)
gauss_2 = make_gauss(150.0, 50.0)
    
plt.plot(gauss_1)
plt.title("Gaussian function 1")
plt.show()
    
plt.plot(gauss_2)
plt.title("Gaussian function 2")
plt.show()

gauss = gauss_1 + gauss_2

plt.plot(gauss) 
plt.title("Final Gaussian function")
plt.show()

freq_gauss = gauss
sum_gauss = gauss.sum()

cdf_gauss = pdf_gauss = np.zeros(256,np.float32)
pdf_gauss = freq_gauss / sum_gauss
cdf_gauss[0] = 0.0

for i in range(1,255):
    cdf_gauss[i] = cdf_gauss[i-1] + pdf_gauss[i]
    

for i in range(256):
    cdf_gauss[i] = round(cdf_gauss[i]*255)

######################################

cdf = pdf = freq = np.zeros(256, np.int32)

for i in range(rows):
    for j in range(cols): 
        freq[ int(img[i][j]) ] += 1

pdf = freq / MN
cdf = pdf.cumsum()
   
for i in range(256):
    cdf[i] = round(cdf[i] * 255.0)
    
###################################

for i in range(rows):
    for j in range(cols):
        
        m_dist = sys.maxsize
        f_px = px = int(img[i][j])
        from_cdf = cdf[px]

        for each_val in range(256):
            x = cdf_gauss[ each_val ] - from_cdf
            x = x * -1 if x < 0 else x
            if x < m_dist:
                m_dist = x
                f_px = each_val
        
        img[i][j] = from_cdf = f_px

show_img(img, "enhanced")
show_hist(img, "output")




