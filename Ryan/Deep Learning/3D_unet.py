# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import random
import glob
import os
import sys
import cv2
import time
import pandas as pd

import tensorflow as tf
import scipy.ndimage
import scipy.misc

from SimpleITK import GetArrayFromImage, ReadImage
from sklearn.model_selection import train_test_split, KFold
from scipy.ndimage.interpolation import zoom
from tweaked_ImageGenerator import customImageDataGenerator

from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from unet_model3d import unet
from unet_3D_v2 import unet_model_3d
from keras import backend as K

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
    
def get_pretrain_data(data_dir, new_shape):
    nr_imgs = 50
    
    #old_shape = (86,333,271)
    
#    img_dim = (nr_imgs,round(factor[0]*old_shape[0]),round(factor[1]*old_shape[1]),
#               round(factor[2]*old_shape[2]))
    img_dim = (nr_imgs,new_shape[0],new_shape[1],new_shape[2])
    
    data_array = np.zeros(img_dim)
    label_array = np.zeros(img_dim) 
    
    for i in range(nr_imgs):
        subj_str = f"{data_dir}/Case{i:02d}"
        mr_img = GetArrayFromImage(ReadImage(subj_str+".mhd"))
        old_shape = mr_img.shape
        factor = (new_shape[0]/old_shape[0],new_shape[1]/old_shape[1],new_shape[2]/old_shape[2])
        data_array[i] = zoom(mr_img,zoom=factor,order=1)
        seg = GetArrayFromImage(ReadImage(subj_str+"_segmentation.mhd"))
        label_array[i] = zoom(seg,zoom=factor,order=1)
        
    data_array = np.expand_dims(data_array,1)
    label_array = np.expand_dims(label_array,1)   
    
    return data_array, label_array

#data_dir=r"PretrainingData"
data_dir = r"TrainingData" #Change to your data directory
data_dir_test=r"TestData"
data_dir_unlab=r"UnlabeledData"
pretrained_weights='pretrained_weights_Ryan.hdf5'
results_dir=r"results"
exp_name='test10'
exp='Baseline'  #'Baseline','Simple','Full_pre', 'Full_it'

#Image shape to which the original images will be subsampled. For now, each
#dimension must be divisible by pool_size^depth (2^4 = 16 by default)
sample_shape = [16,80,64]
#sample_shape=[80,320,256]

#data, labels = get_pretrain_data(data_dir,new_shape=sample_shape)
data, labels = get_data_array(data_dir,new_shape=sample_shape)
data_unlab=get_data_array(data_dir_unlab,new_shape=sample_shape,unlabeled=True)
test_data,test_labels=get_data_array(data_dir_test,new_shape=sample_shape)



# hyperparameters
if exp=='Baseline':
    max_it=0
elif exp=='Simple':
    max_it=2
elif exp=='Full_pre' or exp=='Full_it':
    max_it=10
conf=0.9  #what is a confident prediction
min_conf_rat=0.5  #minimal fraction of confident predictions needed to pass unlabaled image
cv=5
depth = 4
learning_rate=5e-4
n_base_filters=32
use_batchnorm = True
batch_size = 3
epochs = 100
input_shape=tuple([1]+sample_shape)
val_img=int(len(data)/cv) #number of validation images
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

if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
results_dir=os.path.join(results_dir,exp_name)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
elif len(os.listdir(results_dir)) != 0:
    inp = input("OK to overwrite? (y/n) ")
    if inp != 'y':
        sys.exit()
        

print('Pre-Processing finished'+'\n'+'Model training starting'+'\n')
for i,(train_index, val_index) in enumerate(kf.split(data)):
    if not os.path.exists(os.path.join(results_dir,f'cv{i}')):
        os.mkdir(os.path.join(results_dir,f'cv{i}'))
    
    begin_time=time.time()
    
    with tf.device('/cpu:0'):
        model = unet_model_3d(input_shape=input_shape,initial_learning_rate=learning_rate,depth=depth,n_base_filters=n_base_filters,batch_normalization=use_batchnorm)
        model.load_weights(pretrained_weights)
    print(f'Cross-Validation step {i+1} of {cv}'+'\n')
    
    x_unlab=data_unlab
    x_lab=data[train_index]
    y_lab=labels[train_index]
    x_val=data[val_index]
    y_val=labels[val_index]
    
    if exp=='Baseline':
        train_datagen = customImageDataGenerator(horizontal_flip=True, 
                                                 rotation_range=15,
                                                 zoom_range=0.2,
                                                 brightness_range=[0.8,1.2])
        train_generator = train_datagen.flow(x_lab, y_lab, batch_size=batch_size, shuffle=True)

        val_datagen = customImageDataGenerator()
        val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=True)

        STEP_SIZE_TRAIN = np.ceil(float(train_generator.n)/train_generator.batch_size)
        STEP_SIZE_VAL = np.ceil(float(val_generator.n)/val_generator.batch_size)
        
        checkpointer = ModelCheckpoint(filepath=os.path.join(results_dir,f'cv{i}','best_weights.hdf5'), save_weights_only=True, mode='max', monitor='val_dice_coefficient', verbose=2, save_best_only=True)
        #history=model.fit(x_lab,    #nu dus geen re-initialisatie van weights!!
                          #y_lab,
                          #epochs=epochs,
                          #batch_size=batch_size,
                          #validation_data=(x_val,y_val))   
        history=model.fit_generator(train_generator, 
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    epochs=epochs,
                                    validation_data=val_generator,
                                    validation_steps=STEP_SIZE_VAL,
                                    shuffle=True,
                                    verbose=2,
                                    callbacks = [checkpointer])



    it=0
    while len(x_unlab)!=0 and it<max_it:
        it=it+1
        print(f'Start of Iteration {it} of {max_it} with {len(x_lab)} labeled and {len(x_unlab)} unlabeled images'+'\n')
        if not os.path.exists(os.path.join(results_dir,f'cv{i}',f'it{it}')):
            os.mkdir(os.path.join(results_dir,f'cv{i}',f'it{it}'))
        
        train_datagen = customImageDataGenerator(horizontal_flip=True, 
                                                 rotation_range=15,
                                                 zoom_range=0.2,
                                                 brightness_range=[0.8,1.2])
        train_generator = train_datagen.flow(x_lab, y_lab, batch_size=batch_size, shuffle=True)

        val_datagen = customImageDataGenerator()
        val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=True)

        STEP_SIZE_TRAIN = np.ceil(float(train_generator.n)/train_generator.batch_size)
        STEP_SIZE_VAL = np.ceil(float(val_generator.n)/val_generator.batch_size)
        #steps_per_epoch=np.ceil(len(x_lab)/batch_size)
        if exp=='Full_pre':
            model.load_weights(pretrained_weights)
        #history=model.fit(x_lab,    #nu dus geen re-initialisatie van weights!!
                          #y_lab,
                          #epochs=epochs,
                          #batch_size=batch_size,
                          #validation_data=(x_val,y_val))  
                          
        checkpointer = ModelCheckpoint(filepath=os.path.join(results_dir,f'cv{i}',f'it{it}','best_weights.hdf5'), save_weights_only=True, mode='max', monitor='val_dice_coefficient', verbose=2, save_best_only=True)
        history=model.fit_generator(train_generator, 
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    epochs=epochs,
                                    validation_data=val_generator,
                                    validation_steps=STEP_SIZE_VAL,
                                    shuffle=True,
                                    callbacks = [checkpointer])
        
        y_pred_unlab=model.predict(x_unlab)
        
        if exp=='Full_pre' or exp=='Full_it':
            dels=[]
            for u in range(len(y_pred_unlab)):
                y_pred=y_pred_unlab[u]
                
                pros=y_pred[y_pred>=0.5]
                back=y_pred[y_pred<0.5]
                
                
                if np.sum(pros)!=0 and np.sum(back)!=0:
                    if (np.sum(pros>conf)/np.size(pros))>min_conf_rat and (np.sum(back<(1-conf))/np.size(back))>min_conf_rat:
                        x_lab=np.concatenate((x_lab,np.expand_dims(x_unlab[u],axis=1)),axis=0)
                        y_lab=np.concatenate((y_lab,np.expand_dims(y_pred>=0.5,axis=1)),axis=0)
                        dels.append(u)
            x_unlab=np.delete(x_unlab,dels,axis=0)
        elif exp=='Simple' and it==1:
            x_lab=np.concatenate((x_lab,x_unlab),axis=0)
            y_lab=np.concatenate((y_lab,y_pred_unlab>=0.5),axis=0)
            
        model.save_weights(os.path.join(results_dir,f'cv{i}',f'it{it}','weights.hdf5'))
        
    if exp=='Full_pre' or exp=='Full_it':             
        if len(x_unlab)!=0:
            print(f'Self-Training has failed: {len(x_unlab)} unlabeled images remaining'+'\n')
        else:
            print(f'Self-Training has succeeded'+'\n')
        
    model.save_weights(os.path.join(results_dir,f'cv{i}','end_weights.hdf5'))
    
    if exp=='Baseline':
        model.load_weights(os.path.join(results_dir,f'cv{i}','best_weights.hdf5'))
    elif exp=='Simple' or exp=='Full_pre' or exp=='Full_it':
        model.load_weights(os.path.join(results_dir,f'cv{i}',f'it{it}','best_weights.hdf5'))

    y_pred_val=model.predict(x_val)   
    for v in range(val_img):
        val_dice[i,v]=dice(y_pred_val[v]>=0.5,y_val[v])
    print(f'Mean Validation Dice overlap of {np.mean(val_dice[i,:])}'+'\n')
    models.append(model)
    
    K.clear_session()
    print(f'Time expired for cross-validation step {i+1}: {int(time.time() - begin_time)} sec.')

valco=[f'Validation image {v}' for v in range(val_img)]
columns=['Cross-Validation step']+valco+['Mean','Std']
frame=pd.DataFrame(np.column_stack((list(range(0,cv)),val_dice,np.mean(val_dice,axis=1),np.std(val_dice,axis=1))),columns=columns)
frame.to_excel(os.path.join(results_dir,'results.xlsx'),index=False)
print('Process has finished')
        
    
            
 
