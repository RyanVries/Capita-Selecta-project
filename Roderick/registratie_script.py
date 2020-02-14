# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
Registratie script om paramter file te testen 
eerst anatomy goed krijgen en dan die transformatie die daarvoor nodig was toepassen op de mask.
dat resultaat analyseren om te kijken hoe goed de prestatie is (validatie)
'''

#import lijst 
from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
#later ingevoegd 
from scrollview import ScrollView
#import sys
#from mpl_toolkits.mplot3d import Axes3D

####################################################################################################
#het aangeven van de juiste input
#General path voor elastix 
PATH=r'C:\Users\s164819\OneDrive - TU Eindhoven\Master Jaar 1\Kwartiel 3\8DM20 Capita Selecta\Regestration opdracht indu\myfolder'
ELASTIX_PATH = os.path.join(PATH,'elastix.exe')
TRANSFORMIX_PATH = os.path.join(PATH,'transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')


#juiste paramtere file kiezen 
inp_parameter=input('welke paramter versie? \n:')
if inp_parameter=="":
    parameter_file=r'parameter_test.txt'
else:
    parameter_file=inp_parameter
#parameter_file=r'paramter_test.txt'
 
# maak een results folder aan de hand van versie
inp_versie = input("Welke versie? \n:")     #vragen om welke versie
resultsmap = r'C:\Users\s164819\OneDrive - TU Eindhoven\Master Jaar 1\Kwartiel 3\8DM20 Capita Selecta\Project\results_overzicht\results' + inp_versie
if not os.path.exists(resultsmap):      #kijken of versie al bestaat, anders overschrijven
    os.mkdir(resultsmap)
 
#path voor beide patienten
datamap_path=r'C:\Users\s164819\OneDrive - TU Eindhoven\Master Jaar 1\Kwartiel 3\8DM20 Capita Selecta\Project\TrainingData'
fixed_path = os.path.join(datamap_path,'p120')          
moving_path = os.path.join(datamap_path,'p128')

#Juiste path bepalen en omzetten naar juist format van mhd naar een standaard image
fixed_img_path = os.path.join(fixed_path,'mr_bffe.mhd')
fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path))

fixed_mask_path = os.path.join(fixed_path,'prostaat.mhd')
fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_path))

moving_img_path = os.path.join(moving_path,'mr_bffe.mhd')
moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))

moving_mask_path = os.path.join(moving_path,'prostaat.mhd')
moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(moving_mask_path)) 

#slice pakken van de image 
slice_nr = 50

fixed_img_slice = fixed_img[slice_nr,:,:]
fixed_mask_slice = fixed_mask[slice_nr,:,:]

moving_img_slice = moving_img[slice_nr,:,:]
moving_mask_slice = moving_mask[slice_nr,:,:]

####################################################################################################
#verwerken van de input in elastix 

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

# Execute the registration. Make sure the paths below are correct, and
# that the results folder exists from where you are running this script
el.register(
    fixed_image=fixed_img_path,
    moving_image=moving_img_path,
    parameters=[parameter_file],
    output_dir=resultsmap)
    #used the normalized correlation and a gradient descent with a single resolution to look for best fit

# Find the results
transform_path = os.path.join(resultsmap, 'TransformParameters.0.txt')
result_path = os.path.join(resultsmap, 'result.0.mhd')

''' 
# Load the fixed, moving, and result images
#fixed_image = imageio.imread(fixed_image_path)[:, :, 0]
#moving_image = imageio.imread(moving_image_path)[:, :, 0]
transformed_moving_image = imageio.imread(result_path)
'''

###################################################################################################
#applying the transformation to the mask 

# Make a new transformix object tr with the CORRECT PATH to transformix
tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)

# Transform a new image with the transformation parameters
img_out = os.path.join(resultsmap,'image')
if not os.path.exists(img_out):
    os.mkdir(img_out)
    
seg_out = os.path.join(resultsmap,'segmentation')
if not os.path.exists(seg_out):
    os.mkdir(seg_out)

t_img_path = tr.transform_image(moving_img_path, 
                                output_dir=img_out)
t_seg_path = tr.transform_image(moving_mask_path, 
                                output_dir=seg_out)

t_img = sitk.GetArrayFromImage(sitk.ReadImage(t_img_path))
t_seg_img = sitk.GetArrayFromImage(sitk.ReadImage(t_seg_path))

###################################################################################################
#visualizing of the results 
fig, ax = plt.subplots(2, 3, figsize=(15, 15))
ax[0,0].imshow(fixed_img_slice, cmap='gray')
ax[0,0].set_title('Fixed Image')
ax[1,0].imshow(fixed_mask_slice, cmap='gray')
ax[1,0].set_title('Segmentation of fixed')
ax[0,1].imshow(moving_img_slice, cmap='gray')
ax[0,1].set_title('Moving Image')
ax[1,1].imshow(moving_mask_slice, cmap='gray')
ax[1,1].set_title('Segmentation of moving')
ax[0,2].imshow(t_img[slice_nr,:,:], cmap='gray')
ax[0,2].set_title('Transformed moving image')
ax[1,2].imshow(t_seg_img[slice_nr,:,:], cmap='gray')
ax[1,2].set_title('Transformed segmentation')


pos_fixed = np.where(fixed_mask==1)
pos_moving = np.where(moving_mask==1)
pos_transformed = np.where(t_seg_img==1)
            
fig = plt.figure(figsize=plt.figaspect(0.33))
ax = fig.add_subplot(1,3,1, projection='3d')
ax.scatter(pos_fixed[0], pos_fixed[1], pos_fixed[2], c='black')
ax.set_title('Fixed')

ax = fig.add_subplot(1,3,2, projection='3d')
ax.scatter(pos_moving[0], pos_moving[1], pos_moving[2], c='black')
ax.set_title('Moving')

ax = fig.add_subplot(1,3,3, projection='3d')
ax.scatter(pos_transformed[0], pos_transformed[1], pos_transformed[2], c='black')
ax.set_title('Transformed')

plt.show()

'''
#3d beeld test
fig, ax = plt.subplots()
ScrollView(fixed_img).plot(ax)
#resultaat geeft maar een beeld waarmee niet doorheen te scrollen is met touchpad
plt.show()
'''

###################################################################################################
#quantifying the results

#using dice score
dice2=( (2.0*np.sum(np.logical_and(t_seg_img,fixed_mask))) / (np.sum(t_seg_img)+np.sum(fixed_mask)) )
print('Dice similarity score 2 is {}'.format(dice2))



