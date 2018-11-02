import mxnet as mx
import cv2
from colorize_net import whole_net,resize
from config import cfg,transform,transform_test
from dataset import dataset
import numpy as np

test_dataset=dataset(cfg.data_img_dir,cfg.label_img_dir,cfg.index_test,transform_test)
test_data=mx.gluon.data.DataLoader(test_dataset,1,shuffle=False)

predict_net=whole_net()
predict_net.collect_params().load('model/loss_0.0374_color.params',ctx=mx.cpu())

for data,label in test_data:
    pred=predict_net(data) #data:1 1 w h, pred: 1 2 w h
    pred=resize(pred,pred.shape[2]*2,pred.shape[3]*2)
    label=label[:,1:,:,:]

    img=mx.nd.concat(data[0],pred[0],dim=0)#mx.nd.abs(pred[0]-label[0])+0.5

    img=img.transpose((1,2,0))
    img=(img.asnumpy()*255).astype('uint8')
    img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
    img1=img.astype('int32')
    print(np.abs(img1[:,:,0]-img1[:,:,1]).mean())
    cv2.imshow('pred',img)
    cv2.waitKey(0)


