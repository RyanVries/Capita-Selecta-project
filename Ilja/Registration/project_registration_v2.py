# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#! /usr/bin/env python
#
# Example script that shows how to perform the registration

from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import sys
from mpl_toolkits.mplot3d import Axes3D

# IMPORTANT: these paths may differ on your system, depending on where
# Elastix has been installed. Please set accordingly.
ELASTIX_PATH = os.path.join(r'C:\Elastix\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Elastix\transformix.exe')
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

image_folder = "TrainingData"
param_file = 'Par0001affine.txt'
result_dir = r'Results/test'

fixed_subject = "p102"
moving_subject = "p107"

fixed_subject_path = os.path.join(image_folder,fixed_subject)
moving_subject_path = os.path.join(image_folder,moving_subject)

fixed_img_path = os.path.join(fixed_subject_path,'mr_bffe.mhd')
fixed_seg_img_path = os.path.join(fixed_subject_path,'prostaat.mhd')

moving_img_path = os.path.join(moving_subject_path,'mr_bffe.mhd')
moving_seg_img_path = os.path.join(moving_subject_path,'prostaat.mhd')
 
fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path))
fixed_seg_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_seg_img_path))
    
moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))
moving_seg_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_seg_img_path))

slice_id = 60

fixed_img_slice = fixed_img[slice_id,:,:]
fixed_seg_img_slice = fixed_seg_img[slice_id,:,:]

moving_img_slice = moving_img[slice_id,:,:]
moving_seg_img_slice = moving_seg_img[slice_id,:,:]

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

# Make a results directory if none exists
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
else:
    if len(os.listdir(result_dir) ) != 0 and result_dir != r'Results/test':
        inp = input("OK to overwrite? (y/n) ")
        if inp != 'y':
            sys.exit()
            
            
#fig, ax = plt.subplots(1, 3, figsize=(10, 10))

# Execute the registration. Make sure the paths below are correct, and
# that the results folder exists from where you are running this script

el.register(
    fixed_image=fixed_img_path,
    moving_image=moving_img_path,
    parameters=[param_file],
    output_dir=result_dir)

# Find the results
transform_path = os.path.join(result_dir, 'TransformParameters.0.txt')
result_path = os.path.join(result_dir, 'result.0.mhd')
#
# Open the logfile into the dictionary log
#fig = plt.figure()
#
#for i in range(4):
#    log_path = os.path.join(result_dir, 'IterationInfo.0.R{}.txt'.format(i))
#    log = elastix.logfile(log_path)
#    # Plot the 'metric' against the iteration number 'itnr'
#    plt.plot(log['itnr'], log['metric'])
#plt.legend(['Resolution {}'.format(i) for i in range(4)])
#
#
#fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path))
#moving_image = sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path))
#moving_subject_img = sitk.GetArrayFromImage(sitk.ReadImage(result_path))

#moving_subject_img_slice = moving_subject_img[40,:,:]
#
## Load the fixed, moving, and result images
##fixed_image = imageio.imread(fixed_image_path)[:, :, 0]
##moving_image = imageio.imread(moving_image_path)[:, :, 0]
##transformed_moving_image = imageio.imread(result_path)
#

#
# Make a new transformix object tr with the CORRECT PATH to transformix
tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)

# Transform a new image with the transformation parameters
img_out = os.path.join(result_dir,'image')
seg_out = os.path.join(result_dir,'segmentation')

t_img_path = tr.transform_image(moving_img_path, output_dir=img_out)
t_seg_path = tr.transform_image(moving_seg_img_path, output_dir=seg_out)

t_img = sitk.GetArrayFromImage(sitk.ReadImage(t_img_path))
t_seg_img = sitk.GetArrayFromImage(sitk.ReadImage(t_seg_path))

t_img_slice = t_img[slice_id,:,:]
t_seg_img_slice = t_seg_img[slice_id,:,:]
#t_img = imageio.imread(t_img_path.replace('dcm', 'tiff'))[slice_id,:,:]


fig, ax = plt.subplots(2, 3, figsize=(10, 15))
ax[0,0].imshow(fixed_img_slice)#, cmap='gray')
ax[0,0].set_title('Fixed Image')
ax[1,0].imshow(fixed_seg_img_slice, cmap='gray')
ax[1,0].set_title('Segmentation')
ax[0,1].imshow(moving_img_slice)#, cmap='gray')
ax[0,1].set_title('Moving Image')
ax[1,1].imshow(moving_seg_img_slice, cmap='gray')
ax[1,1].set_title('Segmentation')
ax[0,2].imshow(t_img_slice)
ax[0,2].set_title('Transformed moving image')
ax[1,2].imshow(t_seg_img_slice, cmap='gray')
ax[1,2].set_title('Transformed segmentation')


pos_fixed = np.where(fixed_seg_img==1)
pos_moving = np.where(moving_seg_img==1)
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

#
## Get the Jacobian matrix
#jacobian_matrix_path = tr.jacobian_matrix(output_dir=result_dir)
#
## Get the Jacobian determinant
#jacobian_determinant_path = tr.jacobian_determinant(output_dir=result_dir)
#
## Get the full deformation field
#deformation_field_path = tr.deformation_field(output_dir=result_dir)
#
#jacobian_image = imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))
#jacobian_binary = jacobian_image>0
#
## Add a plot of the Jacobian determinant (in this case, the file is a tiff file)
#ax[3].imshow(jacobian_binary,cmap='gray')
#ax[3].set_title('Jacobian\ndeterminant')
#
# Show the plots
#[x.set_axis_off() for x in ax]

