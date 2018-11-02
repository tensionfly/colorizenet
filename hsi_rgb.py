from mxnet import nd
import math
from mxnet.image import imresize,imread
import cv2
eps=math.pow(10,-5)

def bgr2hsi(x):
    """ x:n,c(b,g,r),w,h
        return n,c(h,s,i),w,h
    """
    sum_RGB=nd.sum(x.astype('float32'),axis=1)
    R=x[:,0,:,:].astype('float32')
    G=x[:,1,:,:].astype('float32')
    B=x[:,2,:,:].astype('float32')

    r=(R+eps)/(sum_RGB+3*eps)
    g=(G+eps)/(sum_RGB+3*eps)
    b=(B+eps)/(sum_RGB+3*eps)

    cossita=(2*r-g-b)/(2*((r-g)**2+(r-b)*(g-b))**(1.0/2)+eps)
    cossita_cilp=nd.clip(cossita,-1.0,1.0)
   
    sita=nd.arccos(cossita_cilp)

    h=(nd.where(g>=b,sita,2*math.pi-sita)).expand_dims(axis=1)
    
    s=(1-3*nd.minimum(nd.minimum(r,g),b)).expand_dims(axis=1)
    s=nd.clip(s,0.,1.)

    i=((R+G+B)/3).expand_dims(axis=1)

    return nd.concat(h,s,i,dim=1)

def hsi2bgr(x):
    """ x:n,c(h,s,i),w,h
        return n,c(b,g,r),w,h
    """
    h=x[:,0,:,:]
    s=x[:,1,:,:]
    i=x[:,2,:,:]

    mask1=((h>=0)*(h<2*math.pi/3)).expand_dims(axis=1)
    mask2=((h>=2*math.pi/3)*(h<4*math.pi/3)).expand_dims(axis=1)
    mask3=((h>=4*math.pi/3)*(h<2*math.pi)).expand_dims(axis=1)

    value1=i*(1-s)

    value2_1=i*(1+s*nd.cos(h)/nd.cos(math.pi/3-h))
    value2_2=i*(1+s*nd.cos(h-2*math.pi/3)/nd.cos(math.pi-h))
    value2_3=i*(1+s*nd.cos(h)/nd.cos(5*math.pi/3-h))

    value3_1=3*i-(value1+value2_1)
    value3_2=3*i-(value1+value2_2)
    value3_3=3*i-(value1+value2_3)

    value1=value1.expand_dims(axis=1)

    value2_1=value2_1.expand_dims(axis=1)
    value2_2=value2_2.expand_dims(axis=1)
    value2_3=value2_3.expand_dims(axis=1)

    value3_1=value3_1.expand_dims(axis=1)
    value3_2=value3_2.expand_dims(axis=1)
    value3_3=value3_3.expand_dims(axis=1)

    out1=nd.concat(value1,value3_1,value2_1,dim=1) #b,g,r
    out2=nd.concat(value3_2,value2_2,value1,dim=1)
    out3=nd.concat(value2_3,value1,value3_3,dim=1)

    bgr=out1*mask1+out2*mask2+out3*mask3
    bgr_clip=nd.clip(bgr,0,1)

    return bgr_clip


# img=imread('test.jpg') #b,g,r
# # img[:,:,1]+=200
# # cv2.imshow('test',img.asnumpy())
# # cv2.waitKey(0)

# img=imresize(img,1024,680)
# img=img.transpose((2,0,1)).expand_dims(axis=0)
# y=bgr2hsi(img)
# z=his2bgr(y)
# z=z.transpose((0,2,3,1))
# z=z[0]*255
# img1=imread('test1.jpg').astype('float32') 
# # z=bgr2hsi(z*255)
# # z=his2bgr(z)

# # b=bgr2hsi(b*255)
# # b=his2bgr(b)
# z=z.transpose((0,2,3,1))
# cv2.imshow('test',z[0].asnumpy())
# cv2.waitKey(0)

