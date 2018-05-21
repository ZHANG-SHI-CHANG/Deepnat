import numpy as np
import nrrd

import os
import glob
import random

root = os.getcwd()

def match_data_label(path):
    datas = glob.glob( os.path.join(path,'datas','*.nrrd') )
    labels = []
    unneed = []
    for i,datas_path in enumerate(datas):
        datas_name = datas_path.split('\\')[-1][:-5]
        if os.path.exists( os.path.join(path,'labels',datas_name+'_glm.nrrd') ):
            print('success match data:{} with label:{}'.format(datas_name+'.nrrd',datas_name+'_glm.nrrd'))
            labels.append(os.path.join(path,'labels',datas_name+'_glm.nrrd'))
        else:
            print('failure match datas:{}'.format(datas_name+'.nrrd'))
            unneed.append(i)
    if not unneed:
        for i in unneed[::-1]:
            del datas[i]
    
    dataset = list(zip(datas,labels))
    return dataset
def Padding(data):
    row,col,ch = data.shape
    padd = np.zeros((12,col,ch))
    data = np.concatenate([padd,data,padd],axis=0)
    
    row,col,ch = data.shape
    padd = np.zeros((row,12,ch))
    data = np.concatenate([padd,data,padd],axis=1)
    
    row,col,ch = data.shape
    padd = np.zeros((row,col,12))
    data = np.concatenate([padd,data,padd],axis=2)
    return data
    
if __name__=='__main__':
    if not os.path.exists( os.path.join(root,'processed_dataset') ):
        os.mkdir(os.path.join(root,'processed_dataset'))
    if not os.path.exists( os.path.join(root,'processed_dataset','datas') ):
        os.mkdir(os.path.join(root,'processed_dataset','datas'))
    if not os.path.exists( os.path.join(root,'processed_dataset','labels') ):
        os.mkdir(os.path.join(root,'processed_dataset','labels'))
    
    dataset = match_data_label(path=os.path.join(root,'original_dataset'))
    
    for i,(data_path,label_path) in enumerate(dataset):
        data_name = data_path.split('\\')[-1][:-5]
        
        original_data,original_data_option = nrrd.read(data_path)
        original_label,original_label_option = nrrd.read(label_path)
        print('read {} complete'.format(data_name))
        
        #前后左右上下分别补10
        original_data = Padding(original_data)
        original_label = Padding(original_label)
        print('padding complete')
        
        count = 0
        save_count = 0
        
        for ch in range(12,original_data.shape[2]-12,3):
            for row in range(12,original_data.shape[2]-12,3):
                for col in range(12,original_data.shape[2]-12,3):
                    new_data = original_data[row-10:row+13,col-10:col+13,ch-10:ch+13]
                    new_label = original_label[row-10:row+13,col-10:col+13,ch-10:ch+13]
                    
                    if np.where((new_data>20))[0].shape[0] >= (23*23*23/2):
                        original_data_option['sizes'] = [new_data.shape[1],new_data.shape[0],new_data.shape[2]]
                        original_data_option['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
                        original_data_option['space origin'] = ['0','0','0']
                        original_label_option['sizes'] = [new_label.shape[1],new_label.shape[0],new_label.shape[2]]
                        original_label_option['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
                        original_label_option['space origin'] = ['0','0','0']
                        
                        random_save = random.randint(20,100)
                        if count%random_save==0:
                            if save_count<500:
                                print('save data {} {},save label {} {}'.format(data_name+'.nrrd',save_count,data_name+'_glm.nrrd',save_count))
                                nrrd.write(os.path.join(root,'processed_dataset','datas',data_name+'_{}.nrrd'.format(save_count)),
                                           new_data,options=original_data_option)
                                nrrd.write(os.path.join(root,'processed_dataset','labels',data_name+'_{}_glm.nrrd'.format(save_count)),
                                           new_label,options=original_label_option)
                                save_count += 1
                            else:
                                continue
                        
                        count += 1