# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
from sklearn.metrics.cluster import adjusted_mutual_info_score, normalized_mutual_info_score
from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import elastix
import imageio
import time
import os

# IMPORTANT: these paths may differ on your system, depending on where
# Elastix has been installed. Please set accordingly.
ELASTIX_PATH = os.path.join(r'C:\Elastix\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Elastix\transformix.exe')
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Make a results directory if none exists
results_dir = r"results"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

# Define the paths to the images you want to register and the parameter files
data_dir = r"TrainingData"
param_dir = r"Parameters"
image_dir = r"Images\translation-affine-parameters_test"

def get_param_array(names):
    #Appends a .txt extension to the file names listed in 'names' and returns
    #the resulting list
    return_array = []
    for name in names:
        return_array.append(os.path.join(param_dir, name + ".txt"))
    return return_array

def transform_img(img_path, transformix_object):
    #Transforms the image on 'img_path' using a transformix object and returns
    #transformed image
    transformed_path = transformix_object.transform_image(img_path, output_dir=results_dir, verbose=False)
    transformed_img = GetArrayFromImage(ReadImage(transformed_path))
    return transformed_img

def write_to_file(write_string, file_name):
    #Appends the string 'write_string' to a txt-file 'file_name'
    f = open(f"{results_dir}/{file_name}.txt", 'a')
    f.write(write_string + "\n")
    f.close()

def get_determinant_img(input_transform):
    jacobian_determinant_path = input_transform.jacobian_determinant(output_dir=results_dir, verbose=False)
    jacobian_determinant_img = imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))
    return jacobian_determinant_img

def dice(img1, img2):
    #Compute and return the dice overlap score of two binary images
    return (2.0 * np.sum(np.logical_and(img1, img2).astype(np.double))) / (np.sum(img1.astype(np.double)) + np.sum(img2.astype(np.double)))

def get_image_paths(input_fixed_img_name, input_moving_img_name):
    #Returns an array with the paths of the prostate images and segmentations
    #corresponding to the input fixed image and moving image name
    print(f"{input_fixed_img_name}_{input_moving_img_name}", end=": ", flush=True)
    mr_img_fixed_path = os.path.join(data_dir, input_fixed_img_name, "mr_bffe.mhd")
    mr_img_moving_path = os.path.join(data_dir, input_moving_img_name, "mr_bffe.mhd")
    pros_img_fixed_path = os.path.join(data_dir, input_fixed_img_name, "prostaat.mhd")
    pros_img_moving_path = os.path.join(data_dir, input_moving_img_name, "prostaat.mhd")
    return np.array([mr_img_fixed_path, mr_img_moving_path, pros_img_fixed_path, pros_img_moving_path])

def register_get_transform(input_fixed_img_path, input_moving_img_path, input_param_array):
    #Performs registrations on the images on 'input_fixed_img_path' and
    #'input_moving_img_path' using the parameter files in the array 'input_param_array'
    # Execute the registration.
    print("registration", end=" ", flush=True)
    el.register(fixed_image=input_fixed_img_path, moving_image=input_moving_img_path, parameters=input_param_array, output_dir=results_dir, verbose=False)

    # Make a new transformix object
    print("transformation", end=" ", flush=True)
    transform_path = os.path.join(results_dir, f"TransformParameters.{len(input_param_array) - 1}.txt")
    return_transform = elastix.TransformixInterface(parameters=transform_path, transformix_path=TRANSFORMIX_PATH)
    return return_transform

def ncc(im1, im2):
    #Computes the normalized cross-correlation coefficient of two images
    im1_flat = im1.flatten().astype(np.double)
    im2_flat = im2.flatten().astype(np.double)
    im1_av = np.sum(im1_flat) / len(im1_flat)
    im2_av = np.sum(im1_flat) / len(im2_flat)

    num = np.sum((im1_flat - im1_av) * (im2_flat - im2_av))
    den = (np.sum((im1_flat - im1_av) ** 2) * np.sum((im2_flat - im2_av) ** 2)) ** .5
    return num / den

def nmi(im1, im2, bins=32):
    #Computes the normalized mutual information of two images
    im1_flat = im1.flatten().astype(np.double)
    im2_flat = im2.flatten().astype(np.double)
    minval = np.min([np.min(im1_flat), np.min(im2_flat)])
    maxval = np.max([np.max(im1_flat), np.max(im2_flat)])
    rang = maxval - minval
    im1n = np.divide(im1_flat - minval, rang)
    im2n = np.divide(im2_flat - minval, rang)
    im1bin = np.round(im1n * (bins - 1))
    im2bin = np.round(im2n * (bins - 1))
    p = np.histogram2d(im1bin, im2bin, bins)[0]
    p += 1e-9
    p = p / np.sum(p)
    p_I = np.sum(p, axis=1)
    p_J = np.sum(p, axis=0)
    num = np.sum(p_I * np.log2(p_I)) + np.sum(p_J * np.log2(p_J))
    den = np.sum(np.multiply(p, np.log2(p)))
    return num / den

def ltransform_weights(metric,b0,b1):
    #Linearly transform an array of similarity values to a weight using bias
    #b0 and coefficient b1
    weight = b0+b1*metric
    return weight

def normalize_array(x):
    #Normalize an array to 0-1 range
    return (x-np.min(x))/(np.max(x)-np.min(x))

def transform_images(input_img_paths, input_transform):
    #Transforms images in 'input_img_paths' to images using 'input_transform'
    print("results", end=" ", flush=True)
    fixed_mr_img = GetArrayFromImage(ReadImage(input_img_paths[0]))
    moving_mr_img = GetArrayFromImage(ReadImage(input_img_paths[1]))
    transformed_moving_mr_img = transform_img(input_img_paths[1], input_transform)

    fixed_pros_img = GetArrayFromImage(ReadImage(input_img_paths[2]))
    moving_pros_img = GetArrayFromImage(ReadImage(input_img_paths[3]))
    transformed_moving_pros_img = transform_img(input_img_paths[3], input_transform)
    
    return (fixed_mr_img, moving_mr_img, transformed_moving_mr_img, 
            fixed_pros_img, moving_pros_img, transformed_moving_pros_img)

def compute_metrics(fixed_mr_img, moving_mr_img, transformed_moving_mr_img, 
            fixed_pros_img, moving_pros_img, transformed_moving_pros_img):
    
    dices = []
    dices.append(dice(fixed_pros_img, moving_pros_img))
    dices.append(dice(fixed_pros_img, transformed_moving_pros_img))
    
    nccs = []
    nccs.append(ncc(fixed_mr_img, moving_mr_img))
    nccs.append(ncc(fixed_mr_img, transformed_moving_mr_img))
    
    nmis = []
    nmis.append(nmi(fixed_mr_img, moving_mr_img))
    nmis.append(nmi(fixed_mr_img, transformed_moving_mr_img))
    
    return dices,nccs,nmis
    
def report_metrics(dices,nccs,nmis,duration,fixed_img_name,moving_img_name):
    
    score_string = f"{fixed_img_name}\t{moving_img_name}"
    score_string += f"\t{dices[0]}"
    score_string += f"\t{nccs[0]}"
    score_string += f"\t{nmis[0]}"
    score_string += f"\t{dices[1]}"
    score_string += f"\t{nccs[1]}"
    score_string += f"\t{nmis[1]}"
    score_string += "\t" + str(duration)
    print(score_string, flush=True)
    return score_string

def register_images(fixed_img_name,moving_img_name,param_array):
    #Performs registration of images (using input names of the fixed image and
    #moving image respectively) and returns the transformed segmentation,
    #similarity metric values and duration of registration
    image_paths = get_image_paths(fixed_img_name, moving_img_name)
    begin_time = time.time()
    #transform = register_get_transform(image_paths[0], image_paths[1], param_array)
    duration = time.time() - begin_time
    #fixed_mr_img, moving_mr_img, transformed_moving_mr_img, \
    #fixed_pros_img, moving_pros_img, transformed_moving_pros_img = transform_images(image_paths, transform)
    fixed_mr_img, moving_mr_img, fixed_pros_img, moving_pros_img = retrieve_images(image_paths)
    
    transformed_moving_mr_img,transformed_moving_pros_img = retrieve_transformed_images(fixed_img_name, moving_img_name, len(param_array))
    dices, nccs, nmis = compute_metrics(fixed_mr_img, moving_mr_img, transformed_moving_mr_img,
    fixed_pros_img, moving_pros_img, transformed_moving_pros_img)
    
    return transformed_moving_pros_img,dices,nccs,nmis,duration

def retrieve_images(input_img_paths):
    #Retrieve fixed and moving mr and segmentation images
    fixed_mr_img = GetArrayFromImage(ReadImage(input_img_paths[0]))
    moving_mr_img = GetArrayFromImage(ReadImage(input_img_paths[1]))
    
    fixed_pros_img = GetArrayFromImage(ReadImage(input_img_paths[2]))
    moving_pros_img = GetArrayFromImage(ReadImage(input_img_paths[3]))
    
    return fixed_mr_img,moving_mr_img,fixed_pros_img,moving_pros_img

def retrieve_transformed_images(fixed_img_name,moving_img_name,nr_params):
    #Retrieve transformed images (assuming these already exist)
    t_moving_path = os.path.join(image_dir,f"{fixed_img_name}-{moving_img_name}",f"{nr_params-1}")
    t_moving_pros_path = os.path.join(t_moving_path,"pros","result.mhd")
    t_moving_img_path = os.path.join(t_moving_path,"mr","result.mhd")
    transformed_moving_pros_img = GetArrayFromImage(ReadImage(t_moving_pros_path))
    transformed_moving_mr_img = GetArrayFromImage(ReadImage(t_moving_img_path))
    return transformed_moving_mr_img,transformed_moving_pros_img

def atlas_segmentation(atlas, vote_threshold=0.5, weights=None):
    #Performs atlas-based segmentation based on a set of binary label images
    atlas_size = atlas.shape[3]
    
    #If weights = none, simply add the segmentations
    if weights is None:
        combined = np.sum(atlas, axis = 3)/atlas_size
        
    #If weights are provided, multiply each atlas image with its corresponding weight
    else:
        combined = np.zeros((atlas.shape[0],atlas.shape[1],atlas.shape[2]))
        for atlas_id in range(atlas_size):
            combined += atlas[:,:,:,atlas_id]*weights[atlas_id]
        
        combined = combined/np.sum(weights)
        #print(np.sum(weights))
    
    segm = (combined > vote_threshold).astype(int)
    return segm

def report_results(img_name,dice_score):
    result_string = f"{img_name}"
    #print(''.join(map(str, dice_score)))
    print(dice_score)
    result_string += "\t" + str(dice_score)
    #result_string += "\t" + ''.join(map(str, dice_score))
    return result_string
       

def leave_one_out(img_names,data_dir,param_array,result_file,all_results_file,b0,b1):
    n_img = len(img_names)
    #Get shape of the 3d images
    sh = np.shape(GetArrayFromImage(ReadImage(os.path.join(data_dir,img_names[0],'mr_bffe.mhd'))))
    prostate_estm = np.zeros((sh[0],sh[1],sh[2],len(img_names)))
    seg_dices = np.zeros(n_img)
    
    #Loop over each patient
    for pat_idx, pat in enumerate(img_names): 
        prostates_estm_pat = np.zeros((sh[0],sh[1],sh[2],len(img_names)-1))
        atlas_idx = 0
        nmis_atlas = np.zeros(n_img-1)
        
        #Loop over each atlas image
        for img_idx in range(n_img):
            if img_names[img_idx] != pat: #do not register patient to itself
                #Perform registration of the atlas image to the current patient image
                pat_pros,dices,nccs,nmis,duration = register_images(pat, img_names[img_idx], param_array)
                nmis_atlas[atlas_idx] = nmis[1]
                output_string = report_metrics(dices, nccs, nmis, duration, pat, img_names[img_idx])
                write_to_file(output_string, all_results_file)
                prostates_estm_pat[:,:,:,atlas_idx] = pat_pros
                atlas_idx += 1
        
        #Assign weights based on image similarity
        #Hier wordt bepaald hoe de nmi gebruikt wordt om de weights te gebruiken!
        #In dit geval alleen genormaliseerd             
        rel_nmis = nmis_atlas/np.max(nmis_atlas)
        weights = normalize_array(rel_nmis)
        #weights = ltransform_weights(rel_nmis,b0,b1)
        #weights = np.ones(14)
       # weights = weights1**2
        print('weights',weights)
        
        prostate_estm[:,:,:,pat_idx] = atlas_segmentation(prostates_estm_pat,weights=weights)
        ground_truth = GetArrayFromImage(ReadImage(os.path.join(data_dir,pat,'prostaat.mhd')))
        seg_dices[pat_idx] = dice(ground_truth, prostate_estm[:,:,:,pat_idx])
        score_string = report_results(pat,seg_dices[pat_idx])
        write_to_file(score_string,result_file)
        
        print(f"Registration of patient: {pat} has been completed\n")
        print(score_string)
    
    print(f"Registration has been completed\n")
    dice_av = np.mean(seg_dices)
    print(f"Average dice: {dice_av}")
    return
    
all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]
#total_begin_time = time.time()
########################################################################################################################
param_file_names = ["translation", "affine", "parameters_test"]
param_array = get_param_array(param_file_names)

all_results_file = "test"  #Resultaten voor alle vergelijkingen afbeeldingen
result_file = "test2"      #Alleen totale dice scores

#Parameters voor lineaire transformatie maar is nu niet echt meer nodig maar ok
#b0 = -4.9024
#b1 = 5.3462
b0 = -5.7409
b1 = 6.1845

leave_one_out(all_image_names,data_dir,param_array,result_file,all_results_file,b0,b1)

