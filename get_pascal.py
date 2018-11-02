import os
import tarfile
from mxnet import gluon
import cv2
import random

# data_root = 'data'

# url = ("http://hi.cs.waseda.ac.jp/~iizuka/data/colornet.t7")
# sha1 = "c88fa2bb6dc9f942a492a7dc7009b966"

# fname = gluon.utils.download(url, data_root, sha1_hash=sha1)

# with tarfile.open(fname, 'r') as f:
#     f.extractall(data_root)


def getindex(file:str,index_train:str,index_test:str,ratio):
    with open(file,'r') as f:
        filenames_list=[t.strip() for t in f.readlines()]
        #filenames_list=[x[:-4] for x in filenames_list ]
    
    num_test=int(len(filenames_list)*ratio)

    list_test=random.sample(filenames_list,num_test)
    list_train=list(set(filenames_list).difference(set(list_test)))

    with open(index_train,'w') as f:
        for i in range(len(list_train)):
            f.write(list_train[i]+'\n')
    
    with open(index_test,'w') as f:
        for i in range(len(list_test)):
            f.write(list_test[i]+'\n')


getindex('data/whole_index.txt','data/index_train.txt','data/index_test.txt',0.1)

# path='data/data_color/'
# L=os.listdir(path)

# with open('data/whole_index.txt','w') as f:
#     for name in L:
#         f.write(name+'\n')

# if not os.path.isfile('data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'):
#     with tarfile.open('data/data', 'r') as f:
#         f.extractall('data/')

# path='data/data_color/'
# pathaf='data/data_gray/'

# for fn in os.listdir(path):
#     img=cv2.imread(path+fn,0)
#     cv2.imwrite(pathaf+fn,img)
#     print("done!")


# whole_img=random.sample(L,5000)

# with open('data/whole_index.txt','w') as f:
#     for name in L:
#         f.write(name+'\n')

# if not os.path.isfile('data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'):
#     with tarfile.open('data/data', 'r') as f:
#         f.extractall('data/')

# path='data/data_color/'
# pathaf='data/data_gray/'

# for fn in os.listdir(path):
#     img=cv2.imread(path+fn,0)
#     cv2.imwrite(pathaf+fn,img)
#     print("done!")




print('done!')