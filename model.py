# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:25:14 2019

@author: Lahiru D. Chamain
"""
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten

from keras.regularizers import l2
from keras.models import Model
from keras.engine.topology import Layer
from keras import regularizers
import keras.backend.tensorflow_backend as K
import keras.constraints as constraints

r=100.0 #changes the slope of the sigmoids
lambdaq = 0.001 #BW



def MyCeil(x):
    ceilmy = (K.sigmoid(r*(x-150))+K.sigmoid(r*(x-149))+K.sigmoid(r*(x-148))+K.sigmoid(r*(x-147))+K.sigmoid(r*(x-146))+K.sigmoid(r*(x-145))+K.sigmoid(r*(x-144))+K.sigmoid(r*(x-143))+K.sigmoid(r*(x-142))+K.sigmoid(r*(x-141))+K.sigmoid(r*(x-140))+K.sigmoid(r*(x-139))+K.sigmoid(r*(x-138))+K.sigmoid(r*(x-137))+K.sigmoid(r*(x-136))+K.sigmoid(r*(x-135))+K.sigmoid(r*(x-134))+K.sigmoid(r*(x-133))+K.sigmoid(r*(x-132))+K.sigmoid(r*(x-131))+K.sigmoid(r*(x-130))+K.sigmoid(r*(x-129))+K.sigmoid(r*(x-128))+K.sigmoid(r*(x-127))+K.sigmoid(r*(x-126))+K.sigmoid(r*(x-125))+K.sigmoid(r*(x-124))+K.sigmoid(r*(x-123))+K.sigmoid(r*(x-122))+K.sigmoid(r*(x-121))+K.sigmoid(r*(x-120))+K.sigmoid(r*(x-119))+K.sigmoid(r*(x-118))+K.sigmoid(r*(x-117))+K.sigmoid(r*(x-116))+K.sigmoid(r*(x-115))+K.sigmoid(r*(x-114))+K.sigmoid(r*(x-113))+K.sigmoid(r*(x-112))+K.sigmoid(r*(x-111))+K.sigmoid(r*(x-110))+K.sigmoid(r*(x-109))+K.sigmoid(r*(x-108))+K.sigmoid(r*(x-107))+K.sigmoid(r*(x-106))+K.sigmoid(r*(x-105))+K.sigmoid(r*(x-104))+K.sigmoid(r*(x-103))+K.sigmoid(r*(x-102))+K.sigmoid(r*(x-101))+K.sigmoid(r*(x-100))+K.sigmoid(r*(x-99))+K.sigmoid(r*(x-98))+K.sigmoid(r*(x-97))+K.sigmoid(r*(x-96))+K.sigmoid(r*(x-95))+K.sigmoid(r*(x-94))+K.sigmoid(r*(x-93))+K.sigmoid(r*(x-92))+K.sigmoid(r*(x-91))+K.sigmoid(r*(x-90))+K.sigmoid(r*(x-89))+K.sigmoid(r*(x-88))+K.sigmoid(r*(x-87))+K.sigmoid(r*(x-86))+K.sigmoid(r*(x-85))+K.sigmoid(r*(x-84))+K.sigmoid(r*(x-83))+K.sigmoid(r*(x-82))+K.sigmoid(r*(x-81))+K.sigmoid(r*(x-80))+K.sigmoid(r*(x-79))+K.sigmoid(r*(x-78))+K.sigmoid(r*(x-77))+K.sigmoid(r*(x-76))+K.sigmoid(r*(x-75))+K.sigmoid(r*(x-74))+K.sigmoid(r*(x-73))+K.sigmoid(r*(x-72))+K.sigmoid(r*(x-71))+K.sigmoid(r*(x-70))+K.sigmoid(r*(x-69))+K.sigmoid(r*(x-68))+K.sigmoid(r*(x-67))+K.sigmoid(r*(x-66))+K.sigmoid(r*(x-65))+K.sigmoid(r*(x-64))+K.sigmoid(r*(x-63))+K.sigmoid(r*(x-62))+K.sigmoid(r*(x-61))+K.sigmoid(r*(x-60))+K.sigmoid(r*(x-59))+K.sigmoid(r*(x-58))+K.sigmoid(r*(x-57))+K.sigmoid(r*(x-56))+K.sigmoid(r*(x-55))+K.sigmoid(r*(x-54))+K.sigmoid(r*(x-53))+K.sigmoid(r*(x-52))+K.sigmoid(r*(x-51))+K.sigmoid(r*(x-50))+K.sigmoid(r*(x-49))+K.sigmoid(r*(x-48))+K.sigmoid(r*(x-47))+K.sigmoid(r*(x-46))+K.sigmoid(r*(x-45))+K.sigmoid(r*(x-44))+K.sigmoid(r*(x-43))+K.sigmoid(r*(x-42))+K.sigmoid(r*(x-41))+K.sigmoid(r*(x-40))+K.sigmoid(r*(x-39))+K.sigmoid(r*(x-38))+K.sigmoid(r*(x-37))+K.sigmoid(r*(x-36))+K.sigmoid(r*(x-35))+K.sigmoid(r*(x-34))+K.sigmoid(r*(x-33))+K.sigmoid(r*(x-32))+K.sigmoid(r*(x-31))+K.sigmoid(r*(x-30))+K.sigmoid(r*(x-29))+K.sigmoid(r*(x-28))+K.sigmoid(r*(x-27))+K.sigmoid(r*(x-26))+K.sigmoid(r*(x-25))+K.sigmoid(r*(x-24))+K.sigmoid(r*(x-23))+K.sigmoid(r*(x-22))+K.sigmoid(r*(x-21))+K.sigmoid(r*(x-20))+K.sigmoid(r*(x-19))+K.sigmoid(r*(x-18))+K.sigmoid(r*(x-17))+K.sigmoid(r*(x-16))+K.sigmoid(r*(x-15))+K.sigmoid(r*(x-14))+K.sigmoid(r*(x-13))+K.sigmoid(r*(x-12))+K.sigmoid(r*(x-11))+K.sigmoid(r*(x-10))+K.sigmoid(r*(x-9))+K.sigmoid(r*(x-8))+K.sigmoid(r*(x-7))+K.sigmoid(r*(x-6))+K.sigmoid(r*(x-5))+K.sigmoid(r*(x-4))+K.sigmoid(r*(x-3))+K.sigmoid(r*(x-2))+K.sigmoid(r*(x-1))+
              K.sigmoid(r*(x+150))+K.sigmoid(r*(x+149))+K.sigmoid(r*(x+148))+K.sigmoid(r*(x+147))+K.sigmoid(r*(x+146))+K.sigmoid(r*(x+145))+K.sigmoid(r*(x+144))+K.sigmoid(r*(x+143))+K.sigmoid(r*(x+142))+K.sigmoid(r*(x+141))+K.sigmoid(r*(x+140))+K.sigmoid(r*(x+139))+K.sigmoid(r*(x+138))+K.sigmoid(r*(x+137))+K.sigmoid(r*(x+136))+K.sigmoid(r*(x+135))+K.sigmoid(r*(x+134))+K.sigmoid(r*(x+133))+K.sigmoid(r*(x+132))+K.sigmoid(r*(x+131))+K.sigmoid(r*(x+130))+K.sigmoid(r*(x+129))+K.sigmoid(r*(x+128))+K.sigmoid(r*(x+127))+K.sigmoid(r*(x+126))+K.sigmoid(r*(x+125))+K.sigmoid(r*(x+124))+K.sigmoid(r*(x+123))+K.sigmoid(r*(x+122))+K.sigmoid(r*(x+121))+K.sigmoid(r*(x+120))+K.sigmoid(r*(x+119))+K.sigmoid(r*(x+118))+K.sigmoid(r*(x+117))+K.sigmoid(r*(x+116))+K.sigmoid(r*(x+115))+K.sigmoid(r*(x+114))+K.sigmoid(r*(x+113))+K.sigmoid(r*(x+112))+K.sigmoid(r*(x+111))+K.sigmoid(r*(x+110))+K.sigmoid(r*(x+109))+K.sigmoid(r*(x+108))+K.sigmoid(r*(x+107))+K.sigmoid(r*(x+106))+K.sigmoid(r*(x+105))+K.sigmoid(r*(x+104))+K.sigmoid(r*(x+103))+K.sigmoid(r*(x+102))+K.sigmoid(r*(x+101))+K.sigmoid(r*(x+100))+K.sigmoid(r*(x+99))+K.sigmoid(r*(x+98))+K.sigmoid(r*(x+97))+K.sigmoid(r*(x+96))+K.sigmoid(r*(x+95))+K.sigmoid(r*(x+94))+K.sigmoid(r*(x+93))+K.sigmoid(r*(x+92))+K.sigmoid(r*(x+91))+K.sigmoid(r*(x+90))+K.sigmoid(r*(x+89))+K.sigmoid(r*(x+88))+K.sigmoid(r*(x+87))+K.sigmoid(r*(x+86))+K.sigmoid(r*(x+85))+K.sigmoid(r*(x+84))+K.sigmoid(r*(x+83))+K.sigmoid(r*(x+82))+K.sigmoid(r*(x+81))+K.sigmoid(r*(x+80))+K.sigmoid(r*(x+79))+K.sigmoid(r*(x+78))+K.sigmoid(r*(x+77))+K.sigmoid(r*(x+76))+K.sigmoid(r*(x+75))+K.sigmoid(r*(x+74))+K.sigmoid(r*(x+73))+K.sigmoid(r*(x+72))+K.sigmoid(r*(x+71))+K.sigmoid(r*(x+70))+K.sigmoid(r*(x+69))+K.sigmoid(r*(x+68))+K.sigmoid(r*(x+67))+K.sigmoid(r*(x+66))+K.sigmoid(r*(x+65))+K.sigmoid(r*(x+64))+K.sigmoid(r*(x+63))+K.sigmoid(r*(x+62))+K.sigmoid(r*(x+61))+K.sigmoid(r*(x+60))+K.sigmoid(r*(x+59))+K.sigmoid(r*(x+58))+K.sigmoid(r*(x+57))+K.sigmoid(r*(x+56))+K.sigmoid(r*(x+55))+K.sigmoid(r*(x+54))+K.sigmoid(r*(x+53))+K.sigmoid(r*(x+52))+K.sigmoid(r*(x+51))+K.sigmoid(r*(x+50))+K.sigmoid(r*(x+49))+K.sigmoid(r*(x+48))+K.sigmoid(r*(x+47))+K.sigmoid(r*(x+46))+K.sigmoid(r*(x+45))+K.sigmoid(r*(x+44))+K.sigmoid(r*(x+43))+K.sigmoid(r*(x+42))+K.sigmoid(r*(x+41))+K.sigmoid(r*(x+40))+K.sigmoid(r*(x+39))+K.sigmoid(r*(x+38))+K.sigmoid(r*(x+37))+K.sigmoid(r*(x+36))+K.sigmoid(r*(x+35))+K.sigmoid(r*(x+34))+K.sigmoid(r*(x+33))+K.sigmoid(r*(x+32))+K.sigmoid(r*(x+31))+K.sigmoid(r*(x+30))+K.sigmoid(r*(x+29))+K.sigmoid(r*(x+28))+K.sigmoid(r*(x+27))+K.sigmoid(r*(x+26))+K.sigmoid(r*(x+25))+K.sigmoid(r*(x+24))+K.sigmoid(r*(x+23))+K.sigmoid(r*(x+22))+K.sigmoid(r*(x+21))+K.sigmoid(r*(x+20))+K.sigmoid(r*(x+19))+K.sigmoid(r*(x+18))+K.sigmoid(r*(x+17))+K.sigmoid(r*(x+16))+K.sigmoid(r*(x+15))+K.sigmoid(r*(x+14))+K.sigmoid(r*(x+13))+K.sigmoid(r*(x+12))+K.sigmoid(r*(x+11))+K.sigmoid(r*(x+10))+K.sigmoid(r*(x+9))+K.sigmoid(r*(x+8))+K.sigmoid(r*(x+7))+K.sigmoid(r*(x+6))+K.sigmoid(r*(x+5))+K.sigmoid(r*(x+4))+K.sigmoid(r*(x+3))+K.sigmoid(r*(x+2))+K.sigmoid(r*(x+1))-150)
    return ceilmy


def MyReg(weight_matrix):
   # print('lambdaq:............................................................................................................',lambdaq)
    return lambdaq * K.sum(K.square(weight_matrix)-1)

# trainable element-wise multiplication layer
class Quan(Layer):

    def __init__(self, kernel_regularizer=None,trainable = True,kernel_constraint=None, **kwargs):
        super(Quan, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.trainable = trainable
        self.channels=12
        self.kernel_constraint = constraints.get(kernel_constraint)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # use_traditional True: use JPEG quan table; False: use a trainable quan table
        # color_channel = 3 for YCbCr
        # input_shape[2] = 64
        
        self.kernel = self.add_weight(name='Quan', 
                                          shape=(1,1,1,self.channels,),
                                          initializer='ones',
                                          regularizer=self.kernel_regularizer,
                                          trainable=self.trainable,
                                          constraint=self.kernel_constraint)
        super(Quan, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # tile the 1x64 quan table, multiply with our 48*64 input
        return MyCeil(x * self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


class QuanNet(object):
        def __init__(self, input_shape,depth,lambdaq):
            self.input_shape=input_shape
            self.depth = depth
            self.lambdaq = lambdaq
            self._build_model()
            
            
        def resnet_layer(self,inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))
        
            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x
            
            
        
        def resnet_v1(self,input_shape, depth, num_classes=10):
            def ResNet(x):
                
                if (depth - 2) % 6 != 0:
                    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
            
                
                num_filters = 64#changed
                num_res_blocks = int((depth - 2) / 6)
                
                x = self.resnet_layer(x)
                # Instantiate the stack of residual units
                for stack in range(3):
                    for res_block in range(num_res_blocks):
                        strides = 1
                        if stack > 0 and res_block == 0:  # first layer but not first stack
                            strides = 2  # downsample
                        y = self.resnet_layer(inputs=x,
                                         num_filters=num_filters,
                                         strides=strides,batch_normalization=True)
                        y = self.resnet_layer(inputs=y,
                                         num_filters=num_filters,
                                         activation=None,batch_normalization=True)
                        if stack > 0 and res_block == 0:  # first layer but not first stack
                            # linear projection residual shortcut connection to match
                            # changed dims
                            x = self.resnet_layer(inputs=x,
                                             num_filters=num_filters,
                                             kernel_size=1,
                                             strides=strides,
                                             activation=None,
                                             batch_normalization=True)
                        x = keras.layers.add([x, y])
                        x = Activation('relu')(x)
                    num_filters = int(num_filters*1.5)
            
                # Add classifier on top.
                # v1 does not use BN after last shortcut connection-ReLU
                x = AveragePooling2D(pool_size=4)(x)
                y = Flatten()(x)
                outputs = Dense(num_classes,
                                activation='softmax',
                                kernel_initializer='he_normal')(y)
                
                return outputs
            return ResNet

        
        
        def _build_model(self):
            
        #--------------------Quan Block---------    
            inp = Input(shape=(self.input_shape))
            i = inp
            global lambdaq
            lambdaq= self.lambdaq
            quan = Quan(kernel_regularizer=MyReg,trainable = True,kernel_constraint=constraints.NonNeg())
            i = quan(i)
        #-------------------ResNet-20
            outp = self.resnet_v1(self.input_shape,self.depth)(i)
        
            

            model = Model(inputs=inp, outputs=outp)

            self.model=model