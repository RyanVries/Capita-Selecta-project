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
from sklearn.model_selection import train_test_split, KFold
from scipy.ndimage.interpolation import zoom

from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from unet_model3d import unet
from unet_3D_v2 import unet_model_3d

#Should turn off training on GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

np.random.seed(12345)

def dice(img1, img2):
    #Compute and return the dice overlap score of two binary images
    return (2.0 * np.sum(np.logical_and(img1, img2).astype(np.double))) / (np.sum(img1.astype(np.double)) + np.sum(img2.astype(np.double)))

def get_data_array(data_dir, new_shape, shuffle=False,unlabeled=False):
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
        data_array[i] = zoom(mr_img,zoom=factor,order=1)
        if unlabeled==False:
            seg = GetArrayFromImage(ReadImage(f"{path}/prostaat.mhd"))
            label_array[i] = zoom(seg,zoom=factor,order=1)
            
    data_array = np.expand_dims(data_array,1)
    if unlabeled==False:
        label_array = np.expand_dims(label_array,1)
        return data_array, label_array
    else:
        return data_array

data_dir = r"TrainingData" #Change to your data directory
data_dir_test=r"TestData"
data_dir_unlab=r"UnlabeledData"

#Image shape to which the original images will be subsampled. For now, each
#dimension must be divisible by pool_size^depth (2^4 = 16 by default)
sample_shape = [16,80,64]
#sample_shape=[80,320,256]
data, labels = get_data_array(data_dir,new_shape=sample_shape)
data_unlab=get_data_array(data_dir_unlab,new_shape=sample_shape,unlabeled=True)
test_data,test_labels=get_data_array(data_dir_test,new_shape=sample_shape)
#slice_id = 40
#x_slice = data[1,0,slice_id,:,:]
#y_slice = labels[1,0,slice_id,:,:]
#plt.imshow(data,cmap='gray')


# hyperparameters
max_it=10
conf=0.8  #what is a confident prediction
min_conf_rat=0.8  #minimal amount of confident predictions needed to pass unlabaled image
cv=len(data)
depth = 4
channels = 32
use_batchnorm = True
batch_size = 5
epochs = 100#250
input_shape=tuple([1]+sample_shape)
val_img=1 #number of validation images
#steps_per_epoch = int(np.ceil((patches_per_im * len(train_images)) / batch_size))

# initialize model
#model = unet(input_shape=(None,None,None,1), depth=depth, channels=channels, batchnorm=use_batchnorm)
#Note that the model is both defined and compiled in this function

# compile the model
#model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#model.compile(optimizer=Adam())

# stop the training if the validation loss does not increase for 15 consecutive epochs
#early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
kf = KFold(n_splits=cv) #leave-one-out-approach


val_dice=np.zeros([cv,val_img])
models=[]
print('Pre-Processing finished'+'\n'+'Model training Starting'+'\n')
for i,(train_index, val_index) in enumerate(kf.split(data)):
    model = unet_model_3d(input_shape=input_shape,initial_learning_rate=1e-1,depth=4,n_base_filters=16,batch_normalization=use_batchnorm)

    print(f'Cross-Validation step {i+1} of {cv}'+'\n')
    
    x_unlab=data_unlab
    x_lab=data[train_index]
    y_lab=labels[train_index]
    x_val=data[val_index]
    y_val=labels[val_index]
    
    it=0
    while len(x_unlab)!=0 and it<max_it:
        it=it+1
        print(f'Start of Iteration {it} of {max_it} with {len(x_lab)} labeled and {len(x_unlab)} unlabeled images'+'\n')
        steps_per_epoch=np.ceil(len(x_lab)/batch_size)
        #model.set_weights(pre-weights)
        history=model.fit(x_lab,    #nu dus geen re-initialisatie van weights!!
                          y_lab,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(x_val,y_val))   
                          
        y_pred_unlab=model.predict(x_unlab)
        for u in range(len(y_pred_unlab)):
            y_pred=y_pred_unlab[u]
            
            pros=y_pred[y_pred>=0.5]
            back=y_pred[y_pred<0.5]
            
            if (np.sum(pros>conf)/np.size(pros))>min_conf_rat and (np.sum(back<(1-conf))/np.size(back))>min_conf_rat:
                x_lab=np.concatenate((x_lab,np.expand_dims(x_unlab[u],axis=1)),axis=0)
                y_lab=np.concatenate((y_lab,np.expand_dims(y_pred>=0.5,axis=1)),axis=0)
                x_unlab=np.delete(x_unlab,u,axis=0)
                
                
    if len(x_unlab)!=0:
        print(f'Self-Training has failed: {len(x_unlab)} unlabeled images remaining'+'\n')
    else:
        print(f'Self-Traning has succeeded'+'\n')
        
    y_pred_val=model.predict(x_val)   
    for v in range(val_img):
        val_dice[i,v]=dice(y_pred_val[v],y_val[v])
    print(f'Mean Validation Dice overlap of {np.mean(val_dice[i,:])}'+'\n')
    models.append(model)
    
print('Process has finished')
        
    
            
 