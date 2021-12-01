import cv2
import numpy as np
from copy import copy
import time
def GrayScale(IMG):
    height, width, c = IMG.shape
    image_data = np.asarray(IMG)
    for i in range(height):
        for j in range(width):
            image_data[i][j] = image_data[i][j][1]
    Gray_Image = image_data[0:height,0:width,0:1] # 채널을 1개로 바꿔줌
    return Gray_Image # 반환

def Threshold(IMG,threshold,value): # GV(Gray Value,밝기값)이 임계값(Threshold)보다 큰 경우, Value 값으로 할당
    height,width,c = IMG.shape
    print(c)
    for i in range(height):
        for j in range (width):
            if(IMG[i][j] > threshold):
                IMG[i][j] = value
            else:
                IMG[i][j] = 0
    return IMG

def SobelX(Image):
    sobel_Xfilter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sum = 0
    image_data = np.asarray(Image)
    for height in range(0,len(image_data)-3):
        for width in range(0,len(image_data[0])-3):
            for i in range(0,3):
                for j in range(0,3):
                    sum += image_data[height+i][width+j][0] * sobel_Xfilter[i][j]
            image_data[height][width] = abs(sum)
            sum = 0
    image2 = image_data
    return image2
def SobelY(Image):
    sobel_Yfilter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sum = 0
    image_data = np.asarray(Image)
    for height in range(0,len(image_data)-3):
        for width in range(0,len(image_data[0])-3):
            for i in range(0,3):
                for j in range(0,3):
                    sum += image_data[height+i][width+j][0] * sobel_Yfilter[i][j]
            image_data[height][width] = abs(sum)
            sum = 0
    image2 = image_data
    return image2
img = cv2.imread("C:/Users/lim/Desktop/box.png",1)
cv2.imshow('Original',img)
Grayimg = GrayScale(copy(img))
cv2.imshow('Gray',Grayimg)
Thresimg = Threshold(Grayimg,127,255)
cv2.imshow('threshold',Thresimg)
sobelx = SobelX(copy(Grayimg))
cv2.imshow('sobelx',sobelx)
sobely = SobelY(copy(Grayimg))
cv2.imshow('sobely',sobely)
edge = sobelx+sobely
cv2.imshow('sobelx+y',edge)

img_sobel_x = cv2.Sobel(copy(Grayimg), cv2.CV_64F, 1, 0, ksize=1)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
cv2.imshow('sobel2',img_sobel_x)
cv2.waitKey(0)
cv2.destroyAllWindows()
