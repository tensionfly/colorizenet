import mxnet as mx
from mxnet import nd
from mxnet.image import random_crop,fixed_crop,imresize
import cv2

def rand_crop(data, label, height, width):
    data, rect = random_crop(data, (width, height))
    label = fixed_crop(label, *rect)
    return data, label

def transform(data,label):
    data=cv2.resize(data,(224,224))
    # data = cv2.cvtColor(data,cv2.COLOR_BGR2LAB)
    data=nd.array(data)

    label=cv2.resize(label,(224,224))
    label = cv2.cvtColor(label,cv2.COLOR_BGR2LAB)
    label=label.transpose((2,0,1))
    #data,label=rand_crop(data,label,int(data.shape[0]*0.8),int(data.shape[1]*0.8))

    data=data[:,:,0]
    data=data.expand_dims(axis=0)

    data = data.astype('float32')/255
    label = nd.array(label).astype('float32')/255

    return data,label

def transform_test(data,label):
    data=cv2.resize(data,(224,224))
    # data = cv2.cvtColor(data,cv2.COLOR_BGR2LAB)
    data=nd.array(data)
    
    label=cv2.resize(label,(224,224))
    label = cv2.cvtColor(label,cv2.COLOR_BGR2LAB)
    label=label.transpose((2,0,1))
    #data,label=rand_crop(data,label,int(data.shape[0]*0.8),int(data.shape[1]*0.8))

    data=data[:,:,0]
    data=data.expand_dims(axis=0)

    data = data.astype('float32')/255
    label = nd.array(label).astype('float32')/255

    return data,label

class _config():

    batch_size=128
    path_save_model='model/'
    ctx=mx.cpu()

    data_img_dir='data/data_gray/'
    label_img_dir='data/data_color/'
    index_train='data/index_train.txt'
    index_test='data/index_test.txt'


cfg=_config()