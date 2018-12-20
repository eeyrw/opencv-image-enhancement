import cv2
import numpy as np

def addSaltAndPepper(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255          
    return NoiseImg

def addGaussNoise(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        NoiseImg[randX,randY]=np.random.random_integers(0,255)        
    return NoiseImg 


def contraharmonicMeanFilter(src,kernelSize,Q):
    srcF=src.copy().astype(float)  
    filteredImage=src.copy() 
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(1,height-1):
        for j in range(1,weight-1):
            sum1=0
            sum2=0
            for kernel_i in range(-kernelSize//2,kernelSize//2):
                for kernel_j in range(-kernelSize//2,kernelSize//2):
                    sum1=sum1+(srcF[kernel_i+i,kernel_j+j])**(Q+1)
                    sum2=sum2+(srcF[kernel_i+i,kernel_j+j])**(Q)
            filteredImage[i,j]=sum1/sum2
    return filteredImage    

def geometricMeanFilter(src,kernelSize):
    srcF=src.copy().astype(float)  
    filteredImage=src.copy() 
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(1,height-1):
        for j in range(1,weight-1):
            product1=1
            for kernel_i in range(-kernelSize//2,kernelSize//2):
                for kernel_j in range(-kernelSize//2,kernelSize//2):
                    product1=product1*(srcF[kernel_i+i,kernel_j+j])
            filteredImage[i,j]=product1**(1/(kernelSize*kernelSize))
    return filteredImage



rawImage=cv2.imread('view.JPG',flags=cv2.IMREAD_GRAYSCALE)
cv2.imshow('Raw Image',rawImage)
spNoiseImage=addSaltAndPepper(rawImage,0.05)
gaussAndSpNoiseImage=addGaussNoise(spNoiseImage,0.05)
cv2.imshow('Hybrid Noise Image',gaussAndSpNoiseImage)
cv2.imshow('Geometric Mean Filtered Image',geometricMeanFilter(gaussAndSpNoiseImage,3))
cv2.imshow('Arithmetic mean filtered Image',contraharmonicMeanFilter(gaussAndSpNoiseImage,3,0))
cv2.imshow('Harmonic mean filtered Image',contraharmonicMeanFilter(gaussAndSpNoiseImage,3,-1))
cv2.imshow('ContraHarmonic mean filtered Image with Q=1.5',contraharmonicMeanFilter(gaussAndSpNoiseImage,3,1.5))
cv2.waitKey(0)