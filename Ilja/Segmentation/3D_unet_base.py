# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import random
import glob
import os
import sys
import cv2

import tensorflow as tf
import scipy.ndimage
import scipy.misc

from SimpleITK import GetArrayFromImage, ReadImage
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import zoom

from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from unet_model3d import unet
from unet_3D_v2 import unet_model_3d

#Should turn off training on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def dice(img1, img2):
    #Compute and return the dice overlap score of two binary images
    return (2.0 * np.sum(np.logical_and(img1, img2).astype(np.double))) / (np.sum(img1.astype(np.double)) + np.sum(img2.astype(np.double)))

def get_data_array(data_dir, new_shape, shuffle=False):
    path_array = glob.glob(f"{data_dir}/*")
    if shuffle:
        np.random.shuffle(path_array)
    nr_imgs = len(path_array)
    old_shape = (86,333,271)
    factor = (new_shape[0]/old_shape[0],new_shape[1]/old_shape[1],new_shape[2]/old_shape[2])
    
    img_dim = (nr_imgs,round(factor[0]*old_shape[0]),round(factor[1]*old_shape[1]),
               round(factor[2]*old_shape[2]))
    data_array = np.zeros(img_dim)
    label_array = np.zeros(img_dim)
    for i, path in enumerate(path_array):
        mr_img = GetArrayFromImage(ReadImage(f"{path}/mr_bffe.mhd"))
        seg = GetArrayFromImage(ReadImage(f"{path}/prostaat.mhd"))
        data_array[i] = zoom(mr_img,zoom=factor,order=1)
        label_array[i] = zoom(seg,zoom=factor,order=1)
        
    data_array = np.expand_dims(data_array,1)
    label_array = np.expand_dims(label_array,1)
    return data_array, label_array

data_dir = r"TrainingData" #Change to your data directory

#Image shape to which the original images will be subsampled. For now, each
#dimension must be divisible by pool_size^depth (2^4 = 16 by default)
sample_shape = [16,80,64]
data, labels = get_data_array(data_dir,new_shape=sample_shape)

#slice_id = 40
#x_slice = data[1,0,slice_id,:,:]
#y_slice = labels[1,0,slice_id,:,:]
#plt.imshow(data,cmap='gray')


# hyperparameters
depth = 4
channels = 32
use_batchnorm = True
batch_size = 64
epochs = 100#250
input_shape=tuple([1]+sample_shape)
#steps_per_epoch = int(np.ceil((patches_per_im * len(train_images)) / batch_size))

# initialize model
#model = unet(input_shape=(None,None,None,1), depth=depth, channels=channels, batchnorm=use_batchnorm)
#Note that the model is both defined and compiled in this function
model = unet_model_3d(input_shape=input_shape,depth=4,n_base_filters=16)

# compile the model
#model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#model.compile(optimizer=Adam())

# stop the training if the validation loss does not increase for 15 consecutive epochs
#early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

x_train,y_train = data[0:4],labels[0:4]
x_val,y_val = data[4:5],labels[4:5]

# train the model with the data generator, and save the training history
#history = model.fit(data,labels,epochs=epochs)
history = model.fit(x_train,y_train,epochs=epochs,validation_data=(x_val, y_val))


#Evaluate trained model
x_test,y_test = data[5:6],labels[5:6]
slice_id = 10

# predict test samples
y_pred = model.predict(x_test)#, batch_size=4)

#Binarize predictions
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1

#Compute dice overlap
dice_sc = dice(y_test,y_pred)

# visualize the test result
plt.figure(figsize=(12, 10))
plt.suptitle(f"Dice = {dice_sc}")

plt.subplot(1, 3, 1)
plt.title("Image of the retina")
plt.axis('off')
plt.imshow(x_test[0,0,slice_id,:,:],cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Ground truth vessel segmentation")
plt.axis('off')
plt.imshow(y_test[0,0,slice_id,:,:],cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Predicted vessel segmentation")
plt.axis('off')
plt.imshow(y_pred[0,0,slice_id,:,:],cmap='gray')

plt.show()

print(f"Dice = {dice_sc}")

