from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import pickle
import os

class BasicDataset(Dataset):       

    def __init__(self, pkl_dir=None):
        self.pkl_dir = pkl_dir
        self.id   = []
        self.data = []
        self.mask = []
        self.target = []
        self.aspect = []
        self.mean = 0.2766042564796702
        self.std  = 0.14487048932764265
        if pkl_dir==None:
            return
        dataset = pickle.load(open(os.path.join(pkl_dir, "DataCollection.pk"),"rb"))
        

        for i in dataset:
            id = i[2]
            nodule_size=float(i[3][1][1])           
            orig_img   = i[0][:,:,0]/255
            orig_img   = np.expand_dims(orig_img, axis=0)

            data = np.concatenate((orig_img, orig_img, orig_img), axis=0)
            mask_img = i[0][:,:,3]/255
            self.id.append(id)
            self.data.append(data)
            target = self.getTarget(i)
            self.target.append(target)  # Label/Calcification/Composition
            self.mask.append(mask_img)
            self.aspect.append([target[-1]])
            

    def __len__(self):
        return len(self.id)


    def __getitem__(self, index):
        img = self.data[index]
        img = img.astype(np.float)
        
        mask = self.mask[index]
        mask = mask.astype(np.int)

        target = torch.LongTensor(self.target[index])
        aspect = torch.tensor(self.aspect[index]).float()

        #print(img.shape)
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask).unsqueeze(0), 'target':target, 'aspect':aspect}
    
    def split_train_val(self, n_fold, idx):
        length = len(self.id)
        train = BasicDataset()
        test = BasicDataset()
        Set = [[],[]]
        for i in range(length):
            Set[self.target[i][0]].append(i)
        
        for i in range(2):
            for j in range(len(Set[i])):
                if j%n_fold!=idx:
                    #add_element(j, train)
                    train.id.append(self.id[Set[i][j]])
                    train.data.append(self.data[Set[i][j]])
                    train.mask.append(self.mask[Set[i][j]])
                    train.target.append(self.target[Set[i][j]])
                    train.aspect.append(self.aspect[Set[i][j]])
                else:
                    #add_element(j, test)
                    test.id.append(self.id[Set[i][j]])
                    test.data.append(self.data[Set[i][j]])
                    test.mask.append(self.mask[Set[i][j]])
                    test.target.append(self.target[Set[i][j]])
                    test.aspect.append(self.aspect[Set[i][j]])
        return train,test
    def getTarget(self, i):
        target_label = int(i[1])
        # cal label
        target_cal_0 = int(i[3][2][9])
        target_cal_1 = int(i[3][2][10])
        target_cal_2 = int(i[3][2][11])
        target_cal_3 = int(i[3][2][12])
        if target_cal_0 == 1 or target_cal_1 == 1:
            target_cal = 1
        elif target_cal_2 == 1:
            target_cal = 1
        else:
            target_cal = 0
        # comp label
        target_comp = int(i[3][2][3])
        if target_comp == 0:
            return
        else:
            if target_comp == 1:
                target_comp = 0
            else:
                target_comp = 1
        target_margin = int(i[3][2][0])  # 0:清晰 1:模糊
        target_shape = int(i[3][2][1])   # 0:规则 1:不规则
        target_echo = int(i[3][2][2])
        if target_echo <= 2:
            target_echo = 0
        else:
            target_echo = 1
        if target_margin == 0 or target_shape == 0:
            print(i[2])
            return
        else:
            target_margin -= 1
            target_shape -= 1
        #target_margin = target_shape
        aspect = i[3][1][-1]
        if aspect > 1:
            target_aspect = 1
        else:
            target_aspect = 0
        return [target_label, target_cal, target_comp, target_margin, target_echo] 

