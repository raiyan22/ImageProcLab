'''
# https://github.com/semihhdemirel/Histogram-Equalization
image_directory=path
image=cv2.imread(image_directory,0)
cv2.imshow("Before Histogram Equalization",image)


rows=image.shape[0]
cols=image.shape[1]
frequency=np.zeros((256,1))

for i in range(0,rows):
    for j in range(0,cols):
        frequency[image[i,j],0]=frequency[image[i,j],0]+1
        
cumulative_frequency = np.zeros((256,1))

cumulative_frequency[0,0] = frequency[0,0]

for i in range(1,256):
    cumulative_frequency[i,0] = frequency[i,0] + cumulative_frequency[i-1,0]
    
normalized_frequency = cumulative_frequency / cumulative_frequency[255,0]
print(normalized_frequency )

for i in range(255,0,-1):
    if frequency[i,0] != 0:
        maximum_gray_level = i
        break
    
new_pixel = normalized_frequency * maximum_gray_level

# rounding
for i in range(0,256):
    new_pixel[i,0]=round(new_pixel[i,0])

# replacing pixels in img
for i in range(0,rows):
    for j in range(0,cols):
        image[i,j]=new_pixel[image[i,j],0]

cv2.imshow("After Histogram Equalization",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Raiyan/Desktop/input.jpeg"

img = cv.imread(path)

plt.imshow(img, 'gray')
plt.title('input img')
plt.show()

# flatten, , range
plt.hist(img.ravel(), 256, [0, 256])
plt.title('input histogram')
plt.show()

def histoeq(img):
    n = img.shape[1]
    m = img.shape[2]
    print(m)
    
    output = img.copy()

    for c in range(img.shape[0]):
        freq = np.zeros(256, int)
        for i in range(n):               # caluculating frequencies of each intensity for all 256 values
            for j in range(m):
                freq[img[c].item(i, j)] += 1
        
        pdf = np.zeros(256, float)
        
        for i in range(256):
            pdf[i] = freq[i] / (n * m)
        
        cdf = np.zeros(256, float)
        
        cdf[0] = pdf[0]
        
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + pdf[i] # calculating sum of p_k for the formula
        
        for i in range(n):
            for j in range(m):
                output[c].itemset((i, j), 255 * cdf[img[c].item(i, j)])
    
    return output

out = histoeq(img)

image = img

rows=image.shape[0]
cols=image.shape[1]
frequency=np.zeros((256,1))

for i in range(0,rows):
    for j in range(0,cols):
        frequency[image[i,j],0] = frequency[image[i,j],0]+1
        
cumulative_frequency = np.zeros((256,1))

cumulative_frequency[0,0] = frequency[0,0]

for i in range(1,256):
    cumulative_frequency[i,0] = frequency[i,0] + cumulative_frequency[i-1,0]
    
normalized_frequency = cumulative_frequency / cumulative_frequency[255,0]

for i in range(255,0,-1):
    if frequency[i,0] != 0:
        maximum_gray_level = i
        break
    
new_pixel = normalized_frequency * maximum_gray_level

# rounding
for i in range(0,256):
    new_pixel[i,0] = round(new_pixel[i,0])

# replacing pixels in img
for i in range(0,rows):
    for j in range(0,cols):
        image[i,j] = new_pixel[image[i,j],0]
        
plt.imshow(image, 'gray')
plt.title('output img')
plt.show()

# flatten, , range
plt.hist(image.ravel(), 256, [0, 256])
plt.title('output histogram')
plt.show()

