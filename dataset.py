import os
import cv2
import numpy as np
from mxnet import gluon
import cv2

class dataset(gluon.data.Dataset):
    def __init__(self, data_img_dir: str, label_img_dir:str, index:str, transform=None, **kwargs):
        super(dataset, self).__init__(**kwargs)

        with open(index) as f:
            self.index = [t.strip() for t in f.readlines()]

        self.data_img_dir = data_img_dir
        self.label_img_dir = label_img_dir

        self.transform = transform
    
    def __getitem__(self, idx):
        idx = self.index[idx]

        data_img_path = os.path.join(self.data_img_dir,idx)
        label_img_path=os.path.join(self.label_img_dir,idx)
       
        data_img = cv2.imread(data_img_path)
        label_img=cv2.imread(label_img_path)
    
        if self.transform is None:
            return data_img,label_img
        else:
            return self.transform(data_img, label_img)
    
    def __len__(self):
        return len(self.index)