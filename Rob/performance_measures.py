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
ELASTIX_PATH = os.path.join(r"C:\Users\s167917\Documents\_Python_\elastix\elastix.exe")
TRANSFORMIX_PATH = os.path.join(r"C:\Users\s167917\Documents\_Python_\elastix\transformix.exe")
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Make a results directory if none exists
results_dir = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project\results"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

# Define the paths to the images you want to register and the parameter files
data_dir = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\TrainingData"
param_dir = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project\parameter_files"

def get_param_array(names):
    return_array = []
    for name in names:
        return_array.append(os.path.join(param_dir, name + ".txt"))
    return return_array

def transform_img(img_path, transformix_object):
    transformed_path = transformix_object.transform_image(img_path, output_dir=results_dir, verbose=False)
    transformed_img = GetArrayFromImage(ReadImage(transformed_path))
    return transformed_img

def write_to_file(write_string, file_name):
    f = open(f"{results_dir}/{file_name}.txt", 'a')
    f.write(write_string + "\n")
    f.close()

def get_determinant_img(input_transform):
    jacobian_determinant_path = input_transform.jacobian_determinant(output_dir=results_dir, verbose=False)
    jacobian_determinant_img = imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))
    return jacobian_determinant_img

def dice(img1, img2):
    return (2.0 * np.sum(np.logical_and(img1, img2).astype(np.double))) / (np.sum(img1.astype(np.double)) + np.sum(img2.astype(np.double)))

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
    # begin_time = time.time()
    el.register(fixed_image=input_fixed_img_path, moving_image=input_moving_img_path, parameters=input_param_array, output_dir=results_dir, verbose=False)
    # global duration
    # duration = time.gmtime(time.time() - begin_time)

    # Make a new transformix object
    print("transformation", end=" ", flush=True)
    transform_path = os.path.join(results_dir, f"TransformParameters.{len(input_param_array) - 1}.txt")
    return_transform = elastix.TransformixInterface(parameters=transform_path, transformix_path=TRANSFORMIX_PATH)
    return return_transform

def ncc(im1, im2):
    im1_flat = im1.flatten().astype(np.double)
    im2_flat = im2.flatten().astype(np.double)
    im1_av = np.sum(im1_flat) / len(im1_flat)
    im2_av = np.sum(im1_flat) / len(im2_flat)

    num = np.sum((im1_flat - im1_av) * (im2_flat - im2_av))
    den = (np.sum((im1_flat - im1_av) ** 2) * np.sum((im2_flat - im2_av) ** 2)) ** .5
    return num / den

def mi(img1, img2, nr_bins=32):
    img1 = img1.flatten().astype(np.double)
    img2 = img2.flatten().astype(np.double)
    hist1 = np.histogram(img1, nr_bins)[0]
    hist2 = np.histogram(img2, nr_bins)[0]
    p = np.outer(hist1, hist2)
    p = p / np.sum(p)
    p_i = np.sum(p, 0)
    p_j = np.sum(p, 1)
    nonzero = p > 1e-9
    return np.sum(p[nonzero] * np.log(p[nonzero] / np.outer(p_i, p_j)[nonzero]))

def nmi(im1, im2, bins=32):
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

def msd(img1, img2):
    diff = np.square((img1 - img2).astype(np.double))
    tot_sum = np.sum(diff)
    return_metric = tot_sum / diff.size
    return return_metric

def get_results(input_img_paths, input_transform):
    print("results", end=" ", flush=True)
    fixed_mr_img = GetArrayFromImage(ReadImage(input_img_paths[0]))
    moving_mr_img = GetArrayFromImage(ReadImage(input_img_paths[1]))
    transformed_moving_mr_img = transform_img(input_img_paths[1], input_transform)

    fixed_pros_img = GetArrayFromImage(ReadImage(input_img_paths[2]))
    moving_pros_img = GetArrayFromImage(ReadImage(input_img_paths[3]))
    transformed_moving_pros_img = transform_img(input_img_paths[3], input_transform)

    before_dice = dice(fixed_pros_img, moving_pros_img)
    after_dice = dice(fixed_pros_img, transformed_moving_pros_img)
    score_string = f"{fixed_img_name}\t{moving_img_name}\t{before_dice}\t"
    for func in [nmi, ncc, msd]:
        score_string += str(func(fixed_mr_img, moving_mr_img)) + "\t"
    score_string += f"\t{after_dice}\t"
    for func in [nmi, ncc, msd]:
        score_string += str(func(fixed_mr_img, transformed_moving_mr_img)) + "\t"
    score_string += "\t" + str(time.time() - total_begin_time)
    print(score_string, end=" ", flush=True)
    return score_string

def run_all_quick():
    image_paths = get_image_paths(fixed_img_name, moving_img_name)
    transform = register_get_transform(image_paths[0], image_paths[1], param_array)
    output_string = get_results(image_paths, transform)
    write_to_file(output_string, "dice_vs_performance_measures")

all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]
total_begin_time = time.time()
########################################################################################################################
param_file_names = ["translation", "affine", "parameters_test"]
param_array = get_param_array(param_file_names)
img_nr = 0
for fixed_img_name in all_image_names:
    for moving_img_name in all_image_names:
        if moving_img_name is not fixed_img_name:
            img_nr += 1
            print(f"{img_nr:03}/210", end=" | ", flush=True)
            run_all_quick()
            print("")
