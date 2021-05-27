# -*- coding: utf-8 -*-
"""
Created on Mon May 24 01:49:36 2021

@author: Ishan
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys


def calculate_snr(image,count):
    print("SNR")
    if count==1:
        noise_patch = image[:40,240:280,2,29]
        signal_patch=image[20:60,120:160,2,29]
    elif count==2:
        noise_patch = image[120:160,0:40,0,499]
        signal_patch=image[20:60,60:100,0,499]
    elif count==3:
        noise_patch = image[:40,240:280,238]
        signal_patch=image[20:60,120:160,238]
    elif count==4:
        noise_patch = image[:40,120:160,18,152]
        signal_patch=image[20:60,60:100,18,152]
    elif count==5:
        noise_patch = image[:40,140:180,206]
        signal_patch=image[60:100,110:150,206]
    elif count==6:
        noise_patch = image[20:60,400:440,499]
        signal_patch=image[40:80,250:290,499]
    elif count==7:
        noise_patch = image[:40,110:150,255]
        signal_patch=image[40:80,50:90,255]
    elif count==8:
        noise_patch = image[:40,420:460,149]
        signal_patch=image[20:60,250:290,149]
        
    SNR = np.mean(signal_patch)/np.std(noise_patch)
    print(SNR)

    
def contrast_image(image):
    min_i = np.min(image)
    max_i = np.max(image)
    contrast = (max_i-min_i)/(max_i+min_i)
    print("michelson")
    print(contrast)
    print("numpy rms")
    mean_i = np.mean(image)
    new_a = image - mean_i
    rms = np.sqrt(np.mean(new_a**2))
    print(rms)
    print("entropy")

    hist = np.histogram(image,256,[0,256], density=True)
    h_new = hist[0]
    x = np.ma.log2(h_new)    
    e = -(h_new*x).sum()
    print(e)


def plot_image(image,xrow,yrow,count,name):
    plt.subplot(xrow,yrow,count)
    plt.axis('off')
    plt.gca().set_title(name,fontsize=8)
    if len(image.shape)==4:
        #hist = np.histogram(image,256,[0,256], density=True)
        #plt.tight_layout()
        #plt.fill_between(hist[1][1:],hist[0],color='black')

        plt.imshow(image[:,:,int(image.shape[2]/2),1],cmap='jet')
    else:
        #hist = np.histogram(image,256,[0,256], density=True)
        #plt.tight_layout()
        #plt.fill_between(hist[1][1:],hist[0],color='black')
            
        plt.imshow(image[:,:,int(image.shape[2]/2)],cmap='jet')
    
def applyFilter(image,sigma):
    img_freqs = np.fft.fftshift(np.fft.fftn(image))
    new_x = image.shape[0]
    new_y = image.shape[1]
    new_z = image.shape[2]
    if len(image.shape) == 4:
        new_k = image.shape[3]
        [X,Y,Z,K] = np.mgrid[0:new_x,0:new_y,0:new_z,0:new_k]
    else:
        [X,Y,Z] = np.mgrid[0:new_x,0:new_y,0:new_z]
    xpr = X-int(new_x)//2
    ypr = Y-int(new_y)//2
    zpr = Z-int(new_z)//2
    if len(image.shape) == 4:
        kpr = K-int(new_k)//2
        value = xpr**2+ypr**2+zpr**2+kpr**2
    else:
        value = xpr**2+ypr**2+zpr**2

    gaussfilt = np.exp(-(value/(2*sigma**2)))/(2*np.pi*sigma**2)
    gaussfilt = gaussfilt/np.max(gaussfilt)
    filtered_freqs = img_freqs*gaussfilt
    filtered = np.abs(np.fft.ifftn(np.fft.fftshift(filtered_freqs)))
    return filtered
        

    
img = 0
xrow = 3
yrow = 3 
path= "C:\\Users\\Ishan\\Downloads\\Summer Sem\\Medical Image Analysis\\modalities\\"

    
for directory in os.listdir(path):
    
    for file in os.listdir(path+directory):
        t_path = path+directory+"\\"+file
        n_load = nib.load(t_path)
        n_load = n_load.get_data()
        #contrast_image(n_load)
        img = img+1;
        calculate_snr(n_load,img)
        title = file.split(".")[0]
        filtered = applyFilter(n_load,2)
        plot_image(filtered,xrow,yrow,img,title)
        
        
            
plt.show()



