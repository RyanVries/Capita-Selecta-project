from __future__ import print_function, absolute_import
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import elastix
import imageio
import time
import view
import os
import cv2
import pandas as pd
import sys

# IMPORTANT: these paths may differ on your system, depending on where
# Elastix has been installed. Please set accordingly.
ELASTIX_PATH = os.path.join(r'elastix\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'elastix\transformix.exe')
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

def get_param_array(names):
    return_array = []
    for name in names:
        return_array.append(os.path.join(param_dir,name + ".txt"))
    return return_array

def transform_img(img_path, transformix_object):
    transformed_path = transformix_object.transform_image(img_path, output_dir=results_dir, verbose=False)
    transformed_img = GetArrayFromImage(ReadImage(transformed_path))
    return transformed_img

def write_to_file(fixed_name, moving_name, write_string, file_name):
    f = open(f"{results_dir}/{file_name}.txt", 'a')
    f.write(f"{fixed_name}\t{moving_name}\t{write_string}\t{time.time()}\n")
    f.close()

def get_determinant_img(input_transform):
    jacobian_determinant_path = input_transform.jacobian_determinant(output_dir=results_dir, verbose=False)
    jacobian_determinant_img = imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))
    return jacobian_determinant_img

def get_score(img1, img2):
    return (2.0 * np.sum(np.logical_and(img1, img2))) / (np.sum(img1) + np.sum(img2))

def get_image_paths(input_fixed_img_name, input_moving_img_name):
    print(f"{input_fixed_img_name}_{input_moving_img_name}", end=": ", flush=True)
    mr_img_fixed_path = os.path.join(data_dir, input_fixed_img_name, "mr_bffe.mhd")
    mr_img_moving_path = os.path.join(data_dir, input_moving_img_name, "mr_bffe.mhd")
    pros_img_fixed_path = os.path.join(data_dir, input_fixed_img_name, "prostaat.mhd")
    pros_img_moving_path = os.path.join(data_dir, input_moving_img_name, "prostaat.mhd")
    return np.array([mr_img_fixed_path, mr_img_moving_path, pros_img_fixed_path, pros_img_moving_path])

def register_get_transform(input_fixed_img_path, input_moving_img_path, input_param_array):
    # Execute the registration.
    print("registration", end=" ", flush=True)
    begin_time = time.time()
    el.register(fixed_image=input_fixed_img_path, moving_image=input_moving_img_path, parameters=input_param_array, output_dir=results_dir)
    global duration
    duration = time.gmtime(time.time() - begin_time)

    # Make a new transformix object
    print("transformation", end=" ", flush=True)
    transform_path = os.path.join(results_dir, f"TransformParameters.{len(input_param_array) - 1}.txt")
    return_transform = elastix.TransformixInterface(parameters=transform_path, transformix_path=TRANSFORMIX_PATH)
    return return_transform

def get_images(input_img_paths, input_transform):
    print("results", end=" ", flush=True)
    fixed_mr_img = GetArrayFromImage(ReadImage(input_img_paths[0]))
    moving_mr_img = GetArrayFromImage(ReadImage(input_img_paths[1]))
    fixed_pros_img = GetArrayFromImage(ReadImage(input_img_paths[2]))
    moving_pros_img = GetArrayFromImage(ReadImage(input_img_paths[3]))
    transformed_moving_mr_img = transform_img(input_img_paths[1], input_transform)
    transformed_moving_pros_img = transform_img(input_img_paths[3], input_transform)
    jacobian_determinant_img = get_determinant_img(input_transform)
    return np.array([fixed_mr_img, fixed_pros_img, moving_mr_img, moving_pros_img, transformed_moving_mr_img,
                     transformed_moving_pros_img, jacobian_determinant_img, (jacobian_determinant_img < 0) * 255])

def get_results(input_images, input_fixed_img_name, input_moving_img_name, save=False, animate=True):
    before_score = get_score(input_images[1], input_images[3])
    after_score = get_score(input_images[1], input_images[5])
    if animate:
        print("animation", end=" ", flush=True)
        fig, ax = plt.subplots(2, 4, figsize=(20, 11.25))
        fig.set_facecolor("#888888")
        ax[0, 0].set_title(f"Fixed image: {input_fixed_img_name}")
        ax[0, 1].set_title(f"Moving image: {input_moving_img_name}")
        ax[0, 2].set_title("Transformed moving image")
        ax[0, 3].set_title("Jacobian determinant")
        ax[1, 0].set_title("Fixed prostate image")
        ax[1, 1].set_title(f"Moving prostate image\nDice: {before_score:.3f}")
        ax[1, 2].set_title(f"Transformed moving prostate image\nDice: {after_score:.3f}")
        ax[1, 3].set_title("Jacobian determinant < 0")
        ims = []
        for i in range(len(input_images[0])):
            im00 = ax[0, 0].imshow(input_images[0, i, :, :], animated=True, cmap='gray')
            im10 = ax[1, 0].imshow(input_images[1, i, :, :], animated=True, cmap='gray')
            im01 = ax[0, 1].imshow(input_images[2, i, :, :], animated=True, cmap='gray')
            im11 = ax[1, 1].imshow(input_images[3, i, :, :], animated=True, cmap='gray')
            im02 = ax[0, 2].imshow(input_images[4, i, :, :], animated=True, cmap='gray')
            im12 = ax[1, 2].imshow(input_images[5, i, :, :], animated=True, cmap='gray')
            im03 = ax[0, 3].imshow(input_images[6, i, :, :], animated=True)
            im13 = ax[1, 3].imshow(input_images[7, i, :, :], animated=True, cmap='gray')
            for y in ax:
                for x in y:
                    x.set_axis_off()
            ims.append([im00, im01, im02, im03, im10, im11, im12, im13])
        ani = animation.ArtistAnimation(fig, ims, blit=True, interval=100)
        if save:
            print("saving", end=" ", flush=True)
            ani.save(f"{results_dir}/videos/{after_score:.3f}={before_score:.3f}_{input_fixed_img_name}_{input_moving_img_name}.mp4")
            plt.close()
        else:
            plt.ion()
            plt.show()
            view.show_figure(input_fixed_img_name, input_moving_img_name)  # single_fraction=1, both_fraction=1, scale_factor=1, opacity=0.1, mode="sphere"
    print(f"| dice: {before_score:.3f} => {after_score:.3f} | registration duration: {duration[4]:02}:{duration[5]:02}")
    return f"{before_score:.3f}\t{after_score:.3f}\t{duration[4] * 60 + duration[5]}", before_score, after_score

def run_all(fixed_img_name,moving_img_name,save=False, animate=True):
    image_paths = get_image_paths(fixed_img_name, moving_img_name)
    transform = register_get_transform(image_paths[0], image_paths[1], param_array)
    [fixed_mr_img, fixed_pros_img, moving_mr_img, moving_pros_img, transformed_moving_mr_img,
                     transformed_moving_pros_img, jacobian_determinant_img, jacobian_determinant_img_negs]=get_images(image_paths, transform)
    _,before_dice,dice=get_results([fixed_mr_img, fixed_pros_img, moving_mr_img, moving_pros_img, transformed_moving_mr_img,
                     transformed_moving_pros_img, jacobian_determinant_img, jacobian_determinant_img_negs], fixed_img_name, moving_img_name, save, animate)
    return transformed_moving_pros_img, before_dice,dice

def run_all_quick(fixed_img_name,moving_img_name):
    image_paths = get_image_paths(fixed_img_name, moving_img_name)
    transform = register_get_transform(image_paths[0], image_paths[1], param_array)
    output_string,_,_ = get_results(get_images(image_paths, transform), fixed_img_name, moving_img_name, False, False)
    write_to_file(fixed_img_name, moving_img_name, output_string, "[results]")

# Make a results directory if none exists
results_dir_start = r'results/test8'
if not os.path.exists(results_dir_start):
    os.mkdir(results_dir_start)
elif len(os.listdir(results_dir_start)) != 0:
    inp = input("OK to overwrite? (y/n) ")
    if inp != 'y':
        sys.exit()

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

# Define the paths to the images you want to register and the parameter files
data_dir = r"TrainingData"
param_dir = r"parameter_files"

dirs = [fold for fold in os.listdir(data_dir) if 'p' in fold]
# param_file_names = ["fast"]
param_file_names = ["translation", "affine", "parameters_test"]
param_array = get_param_array(param_file_names)

########################################################################################################################

regs=['registration Dice '+str(i) for i in range(len(dirs))]
regs.append('average registration Dice')
ress=['resulting Dice '+str(i) for i in range(len(dirs))]
ress.append('average resulting Dice')
columns=(regs+ress)

sh=np.shape(GetArrayFromImage(ReadImage(os.path.join(data_dir,dirs[0],'mr_bffe.mhd'))))
prostate_estm=np.zeros((sh[0],sh[1],sh[2],len(dirs)))
for idx,pat in enumerate(dirs): #over patient
    prostates_estm_pat=np.zeros((sh[0],sh[1],sh[2],len(dirs)-1))
    dices=np.zeros((2*len(dirs)+2,1))
    frame=pd.DataFrame(columns=columns)
    j=0
    mask = np.ones((len(dirs),1),dtype=bool)
    for i in range(len(dirs)): #leave one out over the atlases
        if dirs[i]!=pat: #not register patient to itself
            results_dir=os.path.join(results_dir_start,pat+'_'+dirs[i])
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)
            pat_pros,before_dice,dice=run_all(pat,dirs[i],False,False)
            prostates_estm_pat[:,:,:,j]=pat_pros
            dices[i]=before_dice
            dices[i+len(dirs)+1]=dice
            j=j+1
        else:
            dices[i]=-1
            dices[i+len(dirs)+1]=-1
            mask[i]=0
    dices[len(dirs)]=np.ma.average(np.reshape(dices[0:len(dirs)],(len(dirs),1)),weights=mask)
    dices[2*len(dirs)+1]=np.ma.average(np.reshape(dices[len(dirs)+1:2*len(dirs)+1],(len(dirs),1)),weights=mask)
    frame=frame.append(pd.Series(dices[:,0],index=columns), ignore_index=True)
    prostate_estm[:,:,:,idx]=(np.sum(prostates_estm_pat,axis=3)>len(dirs)).astype(int)
    frame.to_excel(os.path.join(results_dir_start,pat)+'.xlsx')
    
#run_all(results_dir,True, True)

# for i in range(20):
#     fixed_img_name, moving_img_name = np.random.choice(dirs, 2, replace=False)
#     run_all(result_dir=results_dir,save=True)

# for fixed_img_name in all_image_names:
#     for moving_img_name in dirs:
#         if moving_img_name is not fixed_img_name:
#             run_all_quick(result_dir=results_dir)

# for fixed_img_name in all_image_names:
#     for moving_img_name in dirs:
#         if moving_img_name is not fixed_img_name:
#             run_all(result_dir=results_dir,save=True)
