import numpy as np
import nrrd

import os
import glob

root = os.getcwd()

#R Cerebral Gray Matter(右脑灰质),L Cerebral Gray Matter(左脑灰质),R Cerebral White Matter(右脑白质),L Cerebral White Matter(左脑白质)
#R Cerebellar Gray Matter(右小脑灰质),L Cerebellar Gray Matter(左小脑灰质),R Cerebellar White Matter(右小脑白质),L Cerebellar White Matter(左小脑白质)
#R Amygdala(右脑杏仁核),L Amygdala(左脑杏仁核),Brain Stem(脑干),R Caudate(右脑尾状核),L Caudate(左脑尾状核),R Hippocampus(右脑海马体),L Hippocampus(左脑海马体)
#R Lateral ventricle(右侧脑室),L Lateral ventricle(左侧脑室),R Pallidum(右旋转体),L Pallidum(左旋转体),R Putamen(右脑壳核),L Putamen(左脑壳核)
#R Thalamus Proper(右丘脑),L Thalamus Proper(左丘脑),R Ventral DC(右脑侧),L Ventral DC(左脑侧)
label_list = [42,43,44,45,38,39,40,41,31,32,35,36,37,47,48,51,52,55,56,57,58,59,60,61,62]

class BatchGenerator():
    def __init__(self,DatasetPath,batch_size=2,shuffle=True):
        self.DatasetPath = DatasetPath
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.__prepare()
        
        self.epoch_batch_count = 0
    
    def next(self):
        while True:
            input_data,input_label = self.__getitem__(self.epoch_batch_count)
            self.epoch_batch_count += 1
            if self.epoch_batch_count*self.batch_size>self.__len__():
                self.epoch_batch_count = 0
                if self.shuffle:
                    np.random.shuffle(self.dataset)
                print('------------------------------next epoch------------------------------')
            yield input_data,input_label
    
    def __getitem__(self,idx):
        l_round = idx*self.batch_size
        r_round = (idx+1)*self.batch_size
        
        if r_round>self.__len__():
            r_round = self.__len__()
            l_round = r_round - self.batch_size
        
        input_data = np.zeros((self.batch_size,23,23,23,1),dtype=np.float32)
        input_label = np.zeros((self.batch_size,27),dtype=np.float32)
        
        count = 0
        
        for data_path,label_path in self.dataset[l_round:r_round]:
            data,data_option = nrrd.read(data_path)
            label,label_option = nrrd.read(label_path)
            label = label[10:13,10:13,10:13].reshape((27,))
            for index,value in enumerate(label):
                if value in label_list:
                    label[index] = label_list.index(value) + 1
                else:
                    label[index] = 0
            
            input_data[count,:,:,:,0] = data.astype(np.float32)
            input_label[count,:] = label.astype(np.int32)
            
            count += 1
        
        return input_data,input_label
        
    def __prepare(self):
        datas = glob.glob( os.path.join(self.DatasetPath,'datas','*.nrrd') )
        labels = []
        unneed = []
        for i,datas_path in enumerate(datas):
            datas_name = datas_path.split('\\')[-1][:-5]
            if os.path.exists( os.path.join(self.DatasetPath,'labels',datas_name+'_glm.nrrd') ):
                print('success match data:{} with label:{}'.format(datas_name+'.nrrd',datas_name+'_glm.nrrd'))
                labels.append(os.path.join(self.DatasetPath,'labels',datas_name+'_glm.nrrd'))
            else:
                print('failure match datas:{}'.format(datas_name+'.nrrd'))
                unneed.append(i)
        if not unneed:
            for i in unneed[::-1]:
                del datas[i]
        
        self.dataset = list(zip(datas,labels))
        if self.shuffle:
            np.random.shuffle(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
        
        
if __name__=='__main__':
    train_dataset = BatchGenerator(os.path.join(root,'processed_dataset'))
    
    count = 0
    for input_data,input_label in train_dataset.next():
        print(input_data.shape,input_label.shape)
        print(count)
        count+=1
    