import numpy as np
import cv2
import nrrd#pip install pynrrd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import gc
gc.enable()

import os
import sys
import glob

label_list = [42,43,44,45,38,39,40,41,31,32,35,36,37,47,48,51,52,55,56,57,58,59,60,61,62]
label_name = ['R Cerebral Gray Matter','L Cerebral Gray Matter','R Cerebellar Gray Matter','L Cerebellar Gray Matter','R Cerebral White Matter',
              'L Cerebral White Matter','R Cerebellar White Matter','L Cerebellar White Matter','R Amygdala','L Amygdala',
              'Brain Stem','R Caudate','L Caudate','R Hippocampus','L Hippocampus',
              'R Lateral ventricle','L Lateral ventricle','R Pallidum','L Pallidum','R Putamen',
              'L Putamen','R Thalamus Proper','L Thalamus Proper','R Ventral DC','L Ventral DC']
color_list = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(1.0,1.0,0.0),(1.0,0.0,1.0),
              (0.0,1.0,1.0),(0.0,0.0,0.5),(0.0,0.39,0.0),(0.8,0.36,0.36),(0.69,0.13,0.13),
              (1.0,0.07,0.57),(0.58,0,0.82),(0.54,0.0,0.0),(1.0,0.20,0.70),(1.0,0.30,0),
              (1.0,0.49,0),(1.0,0.18,0.18),(0.93,0.38,0.38),(0.54,0.39,0.03),(0,0.80,0.80),
              (1.0,0.87,0.67),(0.18,0.30,0.30),(0.46,0.53,0.6),(0.14,0.69,0.66),(0.80,0.36,0.36)]

if __name__=='__main__':
    data_name = sys.argv[1]
    save_name = sys.argv[2]
    slice_dims = sys.argv[3]
    
    data,option = nrrd.read(data_name)
    data[:,:,:] = data[::-1,:,:]
    data[:,:,:] = data[:,::-1,:]
    
    h,w,c = data.shape
    slice_h = int(slice_dims.split(',')[0])
    slice_w = int(slice_dims.split(',')[1])
    slice_c = int(slice_dims.split(',')[2])
    
    assert slice_h<=h or slice_w<=w or slice_c<=c
    print('slice dims:{}h {}w {}c'.format(slice_h,slice_w,slice_c))
    
    png_h = data[slice_h,:,:]
    png_w = data[:,slice_w,:]
    png_c = data[:,:,slice_c]
    
    png_list = [png_h,png_w,png_c]
    for i,png in enumerate(png_list):
        png = png.T[::-1,:]
        _h,_w = png.shape[:2]
        color_png = np.ones((_h,_w,3))
        png = png.reshape((_h*_w,1))
        color_png = color_png.reshape((_h*_w,3))
        for index,value in enumerate(png):
            if value in label_list:
                color_png[index] = [color*255.0 for color in color_list[label_list.index(value)] ]
            else:
                color_png[index] = (0,0,0)
        color_png = color_png.reshape((_h,_w,3))
        
        fig  = plt.figure()
        ax = fig.add_subplot(1,1,1)
        gci = ax.imshow((color_png).astype(np.uint8))
        patches = [ mpatches.Patch(color=color_list[i], label="{}".format(label_name[i])) for i in range(len(color_list)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,ncol=2, borderaxespad=0. )
        plt.savefig(save_name+'_{}.png'.format(i),bbox_inches='tight',dpi=200)
        print('save image {}'.format(save_name+'_{}.png'.format(i)))
        
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()