import numpy as np
import cv2
import nrrd#pip install pynrrd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc
gc.enable()

import os
import sys
import glob

label_list = [[42],
              [43],
              [38],
              [39],
              [44],
              [45],
              [40],
              [41],
              [31],
              [32],
              [35],
              [36],
              [37],
              [47],
              [48],
              [51],
              [52],
              [55],
              [56],
              [57],
              [58],
              [59],
              [60],
              [61],
              [62]
              ]

if __name__=='__main__':
    input_name = sys.argv[1]
    
    root = os.getcwd()

    for nrrd_data_path in glob.glob(os.path.join(root,input_name)):
        original_datas,original_options = nrrd.read(nrrd_data_path)
        
        nrrd_name = nrrd_data_path.split('/')[-1][:-5]
        
        save_root_path = os.path.join(root,'result_pngs',nrrd_name)
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
        
        for i in range(3):
            save_path = os.path.join(save_root_path,str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for png_count in range(original_datas.shape[i]):
                if i==0:
                    png = original_datas[png_count,:,:]
                elif i==1:
                    png = original_datas[:,png_count,:]
                else:
                    png = original_datas[:,:,png_count]
                
                png = png.T[::-1,:].astype(np.float32)
                '''
                png = (png-png[0,0])/(png.max()-png[0,0])
                png[png<0] = 0
                png = (256*png).astype(np.uint8)
                '''
                png = (png).astype(np.uint8)
                one_png = np.ones_like(png)
                
                for i,_label_list in enumerate(label_list):
                    for label in _label_list:
                        png[png==label*one_png]=i
                print(np.unique(png))
                
                #cv2.imwrite(os.path.join(save_path,'{}.png'.format(png_count)),png)
                print('save {}.png from {} of {}'.format(png_count,nrrd_name,i))
                
                fig  = plt.figure()
                ax = fig.add_subplot(1,1,1)
                gci = ax.imshow((png).astype(np.uint8),cmap='jet')
                cbar = plt.colorbar(gci)
                cbar.set_ticks(np.linspace(0,24,25))
                plt.savefig(os.path.join(save_path,'{}.png'.format(png_count)),bbox_inches='tight',dpi=200)
                
                plt.cla()
                plt.clf()
                plt.close('all')
                gc.collect()