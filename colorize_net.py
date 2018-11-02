from mxnet.gluon import nn
from mxnet import nd
from mxnet.image import imresize,imread
from hsi_rgb import *

def BReLU(x):
    return nd.minimum(1., nd.maximum(0., x))

def resize(x,wi,hi):
    n,c,h,w=x.shape
    x=x.transpose((0,2,3,1))
    out=nd.zeros(shape=(n,hi,wi,c))
    for i in range(n):
        out[i]=imresize(x[i],wi,hi,0)

    return out.transpose((0,3,1,2))

class low_level_feature_net(nn.Block):
    def __init__(self,**kwargs):
        super(low_level_feature_net, self).__init__(**kwargs)
        with self.name_scope():
            self.bn0=nn.BatchNorm()
            self.conv1=nn.Conv2D(64,3,2,1,activation='relu')
            self.bn1=nn.BatchNorm()

            self.conv2=nn.Conv2D(128,3,padding=1,activation='relu')
            self.bn2=nn.BatchNorm()

            self.conv3=nn.Conv2D(128,3,2,1,activation='relu')
            self.bn3=nn.BatchNorm()

            self.conv4=nn.Conv2D(256,3,padding=1,activation='relu')
            self.bn4=nn.BatchNorm()

            self.conv5=nn.Conv2D(256,3,2,1,activation='relu')
            self.bn5=nn.BatchNorm()

            self.conv6=nn.Conv2D(512,3,padding=1,activation='relu')
            self.bn6=nn.BatchNorm()
        
    def forward(self, x):
        x1=self.conv1(self.bn0(x))
        x2=self.conv2(self.bn1(x1))
        x3=self.conv3(self.bn2(x2))
        x4=self.conv4(self.bn3(x3))
        x5=self.conv5(self.bn4(x4))
        x6=self.conv6(self.bn5(x5))
        x7=self.bn6(x6)

        return x7

class mid_level_feature_net(nn.Block):
    def __init__(self,**kwargs):
        super(mid_level_feature_net, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1=nn.Conv2D(512,3,padding=1,activation='relu')
            self.bn1=nn.BatchNorm()

            self.conv2=nn.Conv2D(256,3,padding=1,activation='relu')

    def forward(self,x):
        x1=self.bn1(self.conv1(x))
        x2=self.conv2(x1)

        return x2

class gloab_feature_net(nn.Block):
    def __init__(self,**kwargs):
        super(gloab_feature_net, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1=nn.Conv2D(512,3,2,1,activation='relu')
            self.bn1=nn.BatchNorm()

            self.conv2=nn.Conv2D(512,3,padding=1,activation='relu')
            self.bn2=nn.BatchNorm()

            self.conv3=nn.Conv2D(512,3,2,1,activation='relu')
            self.bn3=nn.BatchNorm()

            self.conv4=nn.Conv2D(512,3,padding=1,activation='relu')
            self.bn4=nn.BatchNorm()

            self.dens1=nn.Dense(1024,'relu')
            self.dens2=nn.Dense(512,'relu')
            self.dens3=nn.Dense(256,'relu')

    def forward(self,x):
        x1=self.bn1(self.conv1(x))
        x2=self.bn2(self.conv2(x1))
        x3=self.bn3(self.conv3(x2))
        x4=self.bn4(self.conv4(x3))

        x5=self.dens1(x4)
        x6=self.dens2(x5)
        x7=self.dens3(x6)

        return x7

class colorize_net(nn.Block):
    def __init__(self,**kwargs):
        super(colorize_net, self).__init__(**kwargs)
        with self.name_scope():
            self.bn0=nn.BatchNorm()
            self.conv0=nn.Conv2D(256,1,activation='relu')
            self.bn1=nn.BatchNorm()
            self.conv1=nn.Conv2D(128,3,padding=1,activation='relu')
            self.bn2=nn.BatchNorm()
            self.conv2=nn.Conv2D(64,3,padding=1,activation='relu')
            self.bn3=nn.BatchNorm()
            self.conv3=nn.Conv2D(64,3,padding=1,activation='relu')
            self.bn4=nn.BatchNorm()
            self.conv4=nn.Conv2D(32,3,padding=1,activation='relu')
            self.bn5=nn.BatchNorm()
            self.conv5=nn.Conv2D(2,3,padding=1,activation='sigmoid')

    def forward(self,x):
        # n,c,h,w=orig.shape
        # orig=orig.expand_dims(axis=1)
        x=self.conv0(self.bn0(x))
        x=self.conv1(self.bn1(x))
        x=resize(x,2*x.shape[3],2*x.shape[2])

        x=self.conv2(self.bn2(x))
        x=self.conv3(self.bn3(x))
        x=resize(x,2*x.shape[3],2*x.shape[2])

        x=self.conv4(self.bn4(x))
        x=self.conv5(self.bn5(x))
        # x=resize(x,w,h)

        # hsi=nd.concat(x,orig,dim=1) #h,s,i
        # bgr=hsi2bgr(hsi)

        return x

class whole_net(nn.Block):
    def __init__(self,**kwargs):
        super(whole_net, self).__init__(**kwargs)
        with self.name_scope():
            low_fea_net=low_level_feature_net()
            mid_fea_net=mid_level_feature_net()
            gloab_fea_net=gloab_feature_net()
            color_net=colorize_net()

            self.net = nn.Sequential()
            self.net.add(low_fea_net,mid_fea_net,gloab_fea_net,color_net)

    def forward(self,x):

        x2_1=self.net[0](x)
        x2_2=self.net[1](x2_1)

        # x_scale=resize(x,224,224)
        # x1_1=self.net[0](x_scale)

        x1_2=self.net[2](x2_1)
        x1_2=x1_2.expand_dims(axis=2)
        x1_2=x1_2.expand_dims(axis=3)
        x1_2=nd.broadcast_to(x1_2,shape=x2_2.shape)

        x12=nd.concat(x1_2,x2_2,dim=1)

        hs=self.net[3](x12)

        return hs


# x=nd.random.uniform(shape=(2,1,100,300))
# y=resize(x,50,100)
# # img=imread('test.jpg',1)
# # testnet=whole_net()
# # testnet.initialize()
# # y=testnet(x)
# print(y.shape)
