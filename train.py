# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:25:14 2019

@author: Lahiru D. Chamain
"""
import keras

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
import os
import utils

from model import QuanNet


# Training parameters
batch_size = 32  
epochs = 200
data_augmentation = True
num_classes = 10

n = 2
depth = n * 6 + 2

lambdaq = 0.001

# Model name and depth
model_type = 'ResNet%d' % (depth)

# Load the CIFAR10 data.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#-----------Level offset---------------
X_train = X_train.astype('float32') - 128.0
X_test = X_test.astype('float32') - 128.0


#----------convert frm RGB to YCRCB----------
X_train = utils.batchRGB2YCBCR(X_train)
X_test = utils.batchRGB2YCBCR(X_test)

#-----------convert to Level 1 DB1 wavelets
x_test = utils.batchwavelet(X_test,image_dim=X_test.shape[1])
x_test = x_test/5


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#custom data generator
def creategen(X,Y,batch_size):
    while True:
    
        datagen = ImageDataGenerator(
                
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)

        batches= datagen.flow( X, Y, batch_size=batch_size,shuffle=True)
       
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            temp = utils.batchwavelet(batch[0].astype('float32'),image_dim=32)
            
            yield temp/5 , batch[1]

            idx0 = idx1
            if idx1 >= X.shape[0]:
                break
counter =0            
def lr_schedule(epoch):
    global counter
    epoch=counter
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    counter = counter+1
    return lr



# Input image dimensions.
input_shape = x_test.shape[1:]


QNet = QuanNet(input_shape=input_shape, depth=depth,lambdaq=lambdaq)
model = QNet.model

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)


#Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)


lr_scheduler = LearningRateScheduler(lr_schedule)


callbacks = [checkpoint, lr_scheduler]




print('Using real-time data augmentation.')
for epoch in range(0,epochs):
        print('Epoch:',epoch)
        for layer in model.layers:
            if layer.name.startswith('quan_'):
                print('weight ',layer.get_weights(),'\n')
                
        
        hist=model.fit_generator(creategen(X_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=int(np.ceil(X_train.shape[0]/batch_size)),validation_data=(x_test,y_test),epochs=1, verbose=1,callbacks=callbacks)



# Score trained model.
import time
start = time.time()
    
scores = model.evaluate(x_test, y_test, verbose=1)

end = time.time()
print(end - start)    
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])




