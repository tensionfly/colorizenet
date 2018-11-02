import mxnet as mx
import time,math,os
from colorize_net import whole_net
from mxnet import init,gluon,autograd,nd
from config import cfg,transform,transform_test
from dataset import dataset
from hsi_rgb import *
from colorize_net import resize

import matplotlib.pyplot as plt 
import numpy as np


def calcu_loss(test_data,net,ctx):
    loss=0
    for data, label in test_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label=label[:,1:,:,:]

        output = net(data)
        label=resize(label,output.shape[3],output.shape[2])

        loss+=nd.abs(output-label).mean().asscalar()
    
    return loss/len(test_data)


def net_train(train_data,test_data,net,loss_func,trainer,num_epochs,path_save_model:str,ctx=mx.cpu()):
    net.collect_params().reset_ctx(ctx)
    test_acc_max=10

    # lr=[]
    # losses=[]
    # k=0

    for e in range(num_epochs):
        start=time.clock()
        train_loss=0.

        # if e>=5:
        #     trainer.set_learning_rate(math.pow(10,-0.1*(e+20)))

        # LR=trainer.learning_rate*0.7
        # trainer.set_learning_rate(LR)

        for i,(data,label) in enumerate(train_data):
            data = data.as_in_context(ctx) 

            label = label.as_in_context(ctx)
            label=label[:,1:,:,:]

            with autograd.record():
                output=net(data)

                label=resize(label,output.shape[3],output.shape[2])

                v=2*nd.abs(label-0.5)+0.1
                v_sqrt=v**(0.5)

                loss=loss_func(v_sqrt*output,v_sqrt*label)
                loss_print1=nd.abs(output-label).max().asscalar()
                loss_print2=nd.abs(output-label).mean().asscalar()

            loss.backward()
            trainer.step(cfg.batch_size,ignore_stale_grad=True)

            # if k<110:
            #     print(k,trainer.learning_rate,mx.nd.mean(loss).asscalar())
            #     lr.append(trainer.learning_rate)
            #     losses.append(mx.nd.mean(loss).asscalar())
            #     k+=1
            #     p=-10+0.1*k
            #     trainer.set_learning_rate(math.pow(10,p))

            # if k==110:
            #     plt.figure()
            #     plt.xticks(np.log([1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]),(1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10))
            #     plt.xlabel('learning rate')
            #     plt.ylabel('loss')
            #     plt.plot(np.log(lr),losses)
            #     plt.show()

            print('e: %d, iter: %d, lossmax: %.5f, lossmean: %.5f'%(e,i,loss_print1,loss_print2))
            train_loss+=loss_print2 # mx.nd.mean(loss).asscalar()

            # if i==4:
            #      net.collect_params().save(path_save_model+'loss_%.4f_color.params'%(loss_print))

        test_acc=calcu_loss(test_data,net,ctx)

        if test_acc_max>test_acc:
            if os.path.exists(path_save_model+'loss_%.4f_color.params'%(test_acc_max)):
                os.remove(path_save_model+'loss_%.4f_color.params'%(test_acc_max))
            test_acc_max=test_acc
            net.collect_params().save(path_save_model+'loss_%.4f_color.params'%(test_acc_max))

        end=time.clock()
        print("Epoch %d. Train Loss: %f, Test loss: %f,time: %.2f s" % (
        e, train_loss/len(train_data), test_acc,(end-start)))

train_dataset=dataset(cfg.data_img_dir,cfg.label_img_dir,cfg.index_train,transform)
train_data=gluon.data.DataLoader(train_dataset,cfg.batch_size,shuffle=True,last_batch='discard')

test_dataset=dataset(cfg.data_img_dir,cfg.label_img_dir,cfg.index_test,transform_test)
test_data=gluon.data.DataLoader(test_dataset,cfg.batch_size,shuffle=False,last_batch='discard')

color_net=whole_net()
# color_net.collect_params().initialize(init=init.Xavier(),ctx=mx.cpu())
color_net.collect_params().load('model/loss_0.0374_color.params',ctx=mx.cpu())

loss=gluon.loss.L2Loss()
    
# trainer=gluon.Trainer(color_net.collect_params(), 'sgd', {'learning_rate' :math.pow(10,-1),'momentum': 0.9})
trainer=gluon.Trainer(color_net.collect_params(), 'adadelta')

net_train(train_data,test_data,color_net,loss,trainer,50,cfg.path_save_model,ctx=cfg.ctx)

