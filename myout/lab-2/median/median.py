import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

# path to the input img
path = 'C:/Users/Raiyan/Desktop/building.jpg'

# reading img + converting from BGR to GRAY 
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

k_h = int(input("Enter kernel height: "))
k_w = k_h
k_size = (k_h,k_w)

# avg kernel
kernel = np.ones( k_size, np.float32)

# img height
img_h = img.shape[0]
# img width
img_w = img.shape[1]
# kernel height // 2 
a = kernel.shape[0] // 2
# kernel width // 2
b = kernel.shape[1] // 2

# empty op img 
output = np.zeros((img_h,img_w), np.float32)

# sum of the values of the kernel
k_sum = kernel.sum()
# print(f'ksum is {ksum}')

        
# visiting each pixel in the img
# m ta row img e ... for each row ...
for i in range(img_h):
    # n ta coln img e ... for each coln ...
    for j in range(img_w):
        # empty var for storing all the values
        values = []
        # visiting each pixel in the kernel
        # a ta row img e ... for each row ...
        for x in range(-a,a+1):
            # b ta coln img e ... for each coln ...
            for y in range(-b,b+1):
                if 0 <= i+x < img_h and 0 <= j+y < img_w:
                    calculated_val = kernel[a+x][b+y] * img[i+x][j+y]
                    values.append( calculated_val )
                else:
                    values.append(0)
        values.sort()
        
        median_val = len(values) // 2
        
        output[i][j] = values[median_val]
        output[i][j] /= k_sum
        

def show_images(images):
    # displaying multiple images side by side
    # https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    
    # err : was giving weird colormap due to diff in the mechanism of reading img of cv2 & matplotlib 
    # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
    # running this once in the code will ALWAYS give gray op
    plt.gray()
    
    no_of_imgs = len(images)
    f = plt.figure()
    for i in range(no_of_imgs):
        
        # Debug, plot figure
        axes = f.add_subplot(1, no_of_imgs, i + 1)
        # the last img will show y axis on the RHS instead of LHS(which is by default)
        
        if i==no_of_images-1:
            axes.yaxis.tick_right() 
        if i==0 :
            plt.title('input')
        else :
            plt.title('op')
        plt.imshow(images[i])
        # plt.rc('font', size=8)        
    plt.show(block=True)
        
show_images([img,output])




                
        

