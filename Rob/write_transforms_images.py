from __future__ import print_function, absolute_import
from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import elastix
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

# Define project folder
PROJECT_PATH = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project"
IMAGES_PATH = r"F:\_images_"

# Define the paths
DATA_PATH = os.path.join(PROJECT_PATH, "training_data")
PARAM_PATH = os.path.join(PROJECT_PATH, "parameter_files")

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
img_nr = 0

def get_param_array(names):
    global IMAGES_PATH
    IMAGES_PATH += f"/{'-'.join(names)}"
    if not os.path.exists(IMAGES_PATH):
        os.mkdir(IMAGES_PATH)
    return_array = []
    for name in names:
        return_array.append(os.path.join(PARAM_PATH, name + ".txt"))
    return return_array

def get_image_paths(input_fixed_img_name, input_moving_img_name):
    mr_img_fixed_path = os.path.join(DATA_PATH, input_fixed_img_name, "mr_bffe.mhd")
    mr_img_moving_path = os.path.join(DATA_PATH, input_moving_img_name, "mr_bffe.mhd")
    pros_img_fixed_path = os.path.join(DATA_PATH, input_fixed_img_name, "prostaat.mhd")
    pros_img_moving_path = os.path.join(DATA_PATH, input_moving_img_name, "prostaat.mhd")
    return np.array([mr_img_fixed_path, mr_img_moving_path, pros_img_fixed_path, pros_img_moving_path])

def transform_img(img_path, output_dir, transformix_object):
    transformed_path = transformix_object.transform_image(img_path, output_dir=output_dir, verbose=False)
    transformed_img = GetArrayFromImage(ReadImage(transformed_path))
    return transformed_img

def make_dir(input_dir):
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

def write_transform_images():
    image_paths = get_image_paths(fixed_img_name, moving_img_name)
    results_dir = f"{IMAGES_PATH}/{fixed_img_name}-{moving_img_name}"
    make_dir(results_dir)

    print("register", end="", flush=True)
    el.register(fixed_image=image_paths[0], moving_image=image_paths[1], parameters=param_array, output_dir=results_dir, verbose=False)
    print(" | transform ", end="", flush=True)
    nr_params = len(param_array)
    for i in range(nr_params):
        print(".", end="", flush=True)
        temp_dir = f"{results_dir}/{i}"
        temp_mr_dir = f"{temp_dir}/mr"
        temp_pros_dir = f"{temp_dir}/pros"
        make_dir(temp_dir)
        make_dir(temp_mr_dir)
        make_dir(temp_pros_dir)
        transform_path = os.path.join(results_dir, f"TransformParameters.{i}.txt")
        transform = elastix.TransformixInterface(parameters=transform_path, transformix_path=TRANSFORMIX_PATH)
        transform_img(image_paths[1], temp_mr_dir, transform)
        transform_img(image_paths[3], temp_pros_dir, transform)
    print(" | ", end="", flush=True)

def counter():
    global img_nr
    img_nr += 1
    print(f"{img_nr:03}/210 | {fixed_img_name}-{moving_img_name} | ", end="", flush=True)
    global begin_time
    begin_time = time.time()

def timer():
    duration = time.gmtime(time.time() - begin_time)
    print(f"{duration[4]:02}:{duration[5]:02}")

all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]
param_file_names = ["translation", "affine", "parameters_test"]
param_array = get_param_array(param_file_names)

for fixed_img_name in all_image_names:
    for moving_img_name in all_image_names:
        if fixed_img_name is not moving_img_name:
            counter()
            write_transform_images()
            timer()
