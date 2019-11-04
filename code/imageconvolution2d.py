
# coding: utf-8

# In[10]:


from PIL import Image
import numpy as np
import scipy
import math
import time
from scipy.misc import toimage
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("task1.png",0)
# i = Image.open('task1.png')
# width, height = i.size



d = img.shape
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]

b1 = img.shape
print (b1)

im_1 = np.asarray(img)
# im_2 = np.asarray(img)

# im = numpy.array(Image.open('task1.png'))
# im1 = Image.fromarray(im, 'L')
# im1.show()


convx = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])

convy = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])


image = np.zeros(shape=(height + 2, width + 2), dtype=int)
image1 = np.array(image)
image2 = np.array(image)

#im_1 = np.pad(im_1, pad_width=1, mode='constant', constant_values=0)


# im_2 = np.pad(im_2, pad_width=1, mode='constant', constant_values=0)
# b2 = im_1.shape
# print(b2)


for m in range(1, height+1):
    for n in range(1, width+1):
        image1[m,n] = im_1[m-1,n-1]
        image2[m,n] = im_1[m-1,n-1]
        
b = image1.shape
print(b)
b2 = image2.shape
print(b2)

def conv2d(h, w, image, conv):
    image3 = np.zeros(shape=(height, width), dtype=int)
    image3 = np.array(image2)
    for x in range(1, h+1):
        for y in range(1, w+1):
            image3[x-1, y-1] = (image[x - 1, y - 1] * conv[0, 0] + image[x - 1, y] * conv[0, 1] + image[x - 1, y + 1] * conv[0, 2] +
                                image[x, y - 1] * conv[1, 0] + image[x, y] * conv[1, 1] + image[x, y + 1] * conv[1, 2] +
                                image[x + 1, y - 1] * conv[2, 0] + image[x + 1, y] * conv[2, 1] + image[x + 1, y + 1] * conv[2, 2])
            if (image3[x-1, y-1] < 0):
                    image3[x-1, y-1] =-image3[x-1,y-1]
    toimage(image3).show()
   # cv2.imwrite('outputx.png', image3)
    #plt.imshow(image3)
    return image3

G2d = np.zeros(shape=(height, width), dtype=int)
G2d = np.array(G2d)

def flip(matrix):
    new_matrix=np.zeros((matrix.shape))
    matrix=np.array(matrix)
    l=len(matrix)
    for i in range(l):
        for j in range(l):
            new_matrix[i,j]=matrix[l-i-1,l-j-1]
    return(new_matrix)

flipped_convy = flip(convy)

print(convy)
print(flipped_convy)

flipped_convx = flip(convx)

print(convx)
print(flipped_convx)


G2dx=conv2d(height, width, image1, flipped_convx)
G2dy=conv2d(height, width, image2, flipped_convy)

for x in range(0, height):
        for y in range(0, width):
            G2d[x,y]=math.sqrt(G2dx[x,y]*G2dx[x,y]+G2dy[x,y]*G2dy[x,y])
#toimage(G2d).show()
toimage(G2d).show()
cv2.imwrite('outputxy.png', G2d)

