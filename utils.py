# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:03:33 2019

@author: Lahiru D. Chamain
"""

import numpy as np
import pywt
#import cv2

#----------------------------------RGB2YCbCr
def batchRGB2YCBCR(x_batch):
    alpha_R = 0.299
    alpha_G = 0.587
    alpha_B = 0.114
    x_batchnew = np.zeros((x_batch.shape)).astype('float32')
    for i in range(0,x_batch.shape[0]):
        #Y
        x_batchnew[i,:,:,0] = alpha_R*x_batch[i,:,:,0] + alpha_G*x_batch[i,:,:,1] + alpha_B*x_batch[i,:,:,2]
        #Cb
        x_batchnew[i,:,:,1] = (0.5/(1-alpha_B))*(x_batch[i,:,:,2]-x_batchnew[i,:,:,0])
        #Cr
        x_batchnew[i,:,:,2] = (0.5/(1-alpha_R))*(x_batch[i,:,:,0]-x_batchnew[i,:,:,0])
    return x_batchnew

#------------------- converts images to Level-1 , db1 wavelets
def batchwavelet(x_batch,level=1,db='db1',image_dim=32):

    halfdim = int(image_dim//2)
    x_batchnew = np.zeros((x_batch.shape[0],halfdim,halfdim,12)).astype('float32')
    for i in range(0,x_batch.shape[0]):
        #Y layer
        coeffs= pywt.wavedecn(x_batch[i,:,:,0], level = level, wavelet =db)
        coeff_array,_ = pywt.coeffs_to_array(coeffs)
        x_batchnew[i,:,:,0]=coeff_array[0:halfdim,0:halfdim]
        x_batchnew[i,:,:,1]=coeff_array[0:halfdim,halfdim:halfdim*2]
        x_batchnew[i,:,:,2]=coeff_array[halfdim:halfdim*2,0:halfdim]
        x_batchnew[i,:,:,3]=coeff_array[halfdim:halfdim*2,halfdim:halfdim*2]
        #cb
        
        coeffs= pywt.wavedecn(x_batch[i,:,:,1], level = level, wavelet =db)
        coeff_array,_ = pywt.coeffs_to_array(coeffs)
        x_batchnew[i,:,:,4]=coeff_array[0:halfdim,0:halfdim]
        x_batchnew[i,:,:,5]=coeff_array[0:halfdim,halfdim:halfdim*2]
        x_batchnew[i,:,:,6]=coeff_array[halfdim:halfdim*2,0:halfdim]
        x_batchnew[i,:,:,7]=coeff_array[halfdim:halfdim*2,halfdim:halfdim*2]
        
        #cr
        coeffs= pywt.wavedecn(x_batch[i,:,:,2], level = level, wavelet =db)
        coeff_array,_ = pywt.coeffs_to_array(coeffs)
        x_batchnew[i,:,:,8]=coeff_array[0:halfdim,0:halfdim]
        x_batchnew[i,:,:,9]=coeff_array[0:halfdim,halfdim:halfdim*2]
        x_batchnew[i,:,:,10]=coeff_array[halfdim:halfdim*2,0:halfdim]
        x_batchnew[i,:,:,11]=coeff_array[halfdim:halfdim*2,halfdim:halfdim*2]

    return x_batchnew

