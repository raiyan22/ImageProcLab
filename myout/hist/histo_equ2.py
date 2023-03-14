import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Raiyan/Desktop/input.jpeg"

img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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

plt.imshow(out, 'gray')
plt.title('output img')
plt.show()

# flatten, , range
plt.hist(out.ravel(), 256, [0, 256])
plt.title('input histogram')
plt.show()

