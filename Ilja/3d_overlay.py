# -*- coding: utf-8 -*-
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
from mpl_toolkits.mplot3d import axes3d
import cv2 as cv
from scipy import ndimage

# IMPORTANT: these paths may differ on your system, depending on where
# Elastix has been installed. Please set accordingly.
ELASTIX_PATH = os.path.join(r'C:\Elastix\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Elastix\transformix.exe')
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

image_folder = "TrainingData"
param_file = 'Par0001rigid.txt'
result_dir = r'Results/test'

fixed_subject = "p102"
moving_subject = "p108"

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

pos_fixed = np.where(fixed_seg_img==1)
pos_moving = np.where(moving_seg_img==1)

def overlay_segmentations(fixed_seg,moving_seg,sparsity):
    #Overlays the segmentation of the fixed image (red) with the segmentation
    #of the moving image (blue).
    
    fixed_edge_sparse = sparse_coordinates(fixed_seg,sparsity)
    moving_edge_sparse = sparse_coordinates(moving_seg,sparsity)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(fixed_edge_sparse[0], fixed_edge_sparse[1], fixed_edge_sparse[2], c='red', alpha=0.5, linewidths=.1)
    ax.scatter(moving_edge_sparse[0], moving_edge_sparse[1], moving_edge_sparse[2], c='blue', alpha=0.5, linewidths=.1)
    ax.set_title('Overlay')
    
    for angle in range(0, 360):
        ax.view_init(30, 40)

def sparse_coordinates(seg,sparsity):
    #Extracts edge coordinates from a binary 3D segmentation and reduces the size
    #of the list of the coordinates using the sparsity parameter (0-1)
    
    seg_edge = seg - ndimage.binary_erosion(seg,np.ones((3,3,3)))
    edge_coords = np.where(seg_edge==1)
    
    perm = np.random.permutation(len(edge_coords[0]))
    perm_slice = perm[0:int(sparsity*len(perm))]
    edge_sparse = (edge_coords[0][perm_slice],edge_coords[1][perm_slice],edge_coords[2][perm_slice])
    
    return edge_sparse

overlay_segmentations(fixed_seg_img,moving_seg_img,0.01)