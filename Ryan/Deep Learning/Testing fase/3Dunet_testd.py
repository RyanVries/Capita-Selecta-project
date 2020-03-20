# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import glob
import os
import cv2
import sys
import pandas as pd

from SimpleITK import GetArrayFromImage, ReadImage, WriteImage, GetImageFromArray
from scipy.ndimage.interpolation import zoom


from keras.models import model_from_json


np.random.seed(12345)

def dice(img1, img2):
    #Compute and return the dice overlap score of two binary images
    return (2.0 * np.sum(np.logical_and(img1, img2).astype(np.double))) / (np.sum(img1.astype(np.double)) + np.sum(img2.astype(np.double)))

def get_data_array(data_dir, new_shape):
    path_array = glob.glob(f"{data_dir}/*")
    nr_imgs = len(path_array)
    old_shape = (86,333,271)
    factor = (new_shape[0]/old_shape[0],new_shape[1]/old_shape[1],new_shape[2]/old_shape[2])
    
    img_dim = (nr_imgs,round(factor[0]*old_shape[0]),round(factor[1]*old_shape[1]),
               round(factor[2]*old_shape[2]))
    data_array = np.zeros(img_dim)
    if ranking==False:
        label_array = np.zeros((nr_imgs,old_shape[0],old_shape[1],old_shape[2])) 
    for i, path in enumerate(path_array):
        mr_img = GetArrayFromImage(ReadImage(f"{path}/mr_bffe.mhd"))
        data_array[i] = zoom(mr_img,zoom=factor,order=1)
        if ranking==False:
            label_array[i] = GetArrayFromImage(ReadImage(f"{path}/prostaat.mhd"))
            
    data_array = np.expand_dims(data_array,1)
    if ranking==False:
        return data_array, label_array
    elif ranking==True:
        return data_array
    
    
data_dir_test = r"TestData"   #location of test data
results_dir = r"test_results"   #where to write the results
weights_dir = r"results/pretrainw+base1"   #directory name with all cv steps
weights_name='best_weights.hdf5'  #name of the weights
exp = 'Baseline'  #arbitrary name for this experiment 
ranking=False  #False if our test data and True if test data from Veronika

if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
if not os.path.exists(os.path.join(results_dir,exp)):
    os.mkdir(os.path.join(results_dir,exp))
    if not os.path.exists(os.path.join(results_dir,exp,'Estimated Images')):
        os.mkdir(os.path.join(results_dir,exp,'Estimated Images'))
elif len(os.listdir(os.path.join(results_dir,exp))) != 0:
    inp = input("OK to overwrite? (y/n) ")
    if inp != 'y':
        sys.exit()

#Image shape to which the original images will be subsampled. For now, each
#dimension must be divisible by pool_size^depth (2^4 = 16 by default)
sample_shape = [16,80,64]
og_shape = [86,333,271]
cv = 5

if ranking==False:
    test_data, test_labels = get_data_array(data_dir_test,new_shape=sample_shape)
elif ranking==True:
    test_data = get_data_array(data_dir_test,new_shape=sample_shape)
    dirp = [p for p in os.listdir(data_dir_test) if p[0]=='p']
    
    
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


test_preds = np.zeros((test_data.shape[0],cv,og_shape[0],og_shape[1],og_shape[2]))
factor = (og_shape[0]/sample_shape[0],og_shape[1]/sample_shape[1],og_shape[2]/sample_shape[2])
for i in range(cv):
    path = os.path.join(weights_dir,f'cv{i}')
    if len(os.listdir(path))>1:
        path = os.path.join(path,f'it{len(os.listdir(path))-1}')
    model.load_weights(os.path.join(path,weights_name))
    pred_test = (model.predict(test_data)>=0.5)
    pred_test_og = [zoom(pred_test[z,0,:,:,:],zoom=factor,order=0) for z in range(len(test_data))]
    for t in range(len(test_data)):
        test_preds[t,i,:,:,:] = pred_test_og[t]
        
test_dice = np.zeros((len(test_data),1))
for idx in range(len(test_data)):
    test_mv = (np.sum(test_preds[idx,:,:,:,:],axis=0)>(len(test_data)/2)).astype(float)
    if ranking==False:
        test_dice[idx] = dice(test_mv,test_labels[idx])
    
    itk_image = GetImageFromArray(test_mv)
    itk_image.SetSpacing([0.488281, 0.488281, 1])
    if ranking==False:
        WriteImage(itk_image, os.path.join(results_dir,exp,'Estimated Images',f'Image{idx}.mhd'))
    elif ranking==True:
        if not os.path.exists(os.path.join(results_dir,exp,'Estimated Images',f'p{dirp[idx]}')):
            os.mkdir(os.path.join(results_dir,exp,'Estimated Images',f'{dirp[idx]}'))
        WriteImage(itk_image, os.path.join(results_dir,exp,'Estimated Images',f'{dirp[idx]}','prostaat.mhd'))
        
if ranking==False:
    columns = ['exp_name']+[f'Test image {v}' for v in range(len(test_dice))]+['mean','std']
    data = np.column_stack(([exp],np.transpose(test_dice),[np.mean(test_dice)], [np.std(test_dice)]))
    frame = pd.DataFrame(data,columns=columns)
    frame.to_excel(os.path.join(results_dir,exp,'results_test.xlsx'),index=False)