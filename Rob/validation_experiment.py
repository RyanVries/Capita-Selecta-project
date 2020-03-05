from SimpleITK import GetArrayFromImage, ReadImage, WriteImage, GetImageFromArray
import numpy as np
import metrics
import elastix
import time
import os

# Define paths
ELASTIX_PATH = os.path.join(r"C:\Users\s167917\Documents\_Python_\elastix\elastix.exe")
TRANSFORMIX_PATH = os.path.join(r"C:\Users\s167917\Documents\_Python_\elastix\transformix.exe")

PROJECT_PATH = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project"
FOLDER_NAME = "translation-affine-parameters_test"

TRAINING_DATA_PATH = os.path.join(PROJECT_PATH, "training_data")
VALIDATION_DATA_PATH = os.path.join(PROJECT_PATH, "validation_data")
PARAM_PATH = os.path.join(PROJECT_PATH, "parameter_files")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")
TEMP_RESULTS_PATH = os.path.join(RESULTS_PATH, "temp")
PREDICTIONS_PATH = os.path.join(RESULTS_PATH, "predictions")
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
if not os.path.exists(PREDICTIONS_PATH):
    os.mkdir(PREDICTIONS_PATH)
if not os.path.exists(TEMP_RESULTS_PATH):
    os.mkdir(TEMP_RESULTS_PATH)

el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

def get_param_array(names):
    return_array = []
    for name in names:
        return_array.append(os.path.join(PARAM_PATH, name + ".txt"))
    return return_array

def get_transform(fixed_img_path, moving_img_path):
    el.register(fixed_image=fixed_img_path, moving_image=moving_img_path, parameters=param_array, output_dir=TEMP_RESULTS_PATH, verbose=False)
    transform_path = os.path.join(TEMP_RESULTS_PATH, f"TransformParameters.{len(param_array) - 1}.txt")
    return elastix.TransformixInterface(parameters=transform_path, transformix_path=TRANSFORMIX_PATH)

def get_transformed_image(input_img_path, input_transform):
    transformed_path = input_transform.transform_image(input_img_path, output_dir=TEMP_RESULTS_PATH, verbose=False)
    return GetArrayFromImage(ReadImage(transformed_path))

def write_mhd(input_name, input_img):
    itk_image = GetImageFromArray(input_img)
    itk_image.SetSpacing([0.488281, 0.488281, 1])
    file_path = f"{PREDICTIONS_PATH}/{input_name}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    WriteImage(itk_image, f"{file_path}/prostaat_prediction.mhd")

all_training_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]
all_validation_image_names = ["p137", "p141", "p143", "p144", "p147"]
param_file_names = ["translation", "affine", "parameters_test"]
param_array = get_param_array(param_file_names)

nr_atlas_images = len(all_training_image_names)
for valid_img_name in all_validation_image_names:
    begin_time = time.time()
    print(f"{valid_img_name} |", end="\t", flush=True)
    valid_img_path = f"{VALIDATION_DATA_PATH}/{valid_img_name}/mr_bffe.mhd"
    valid_img = GetArrayFromImage(ReadImage(valid_img_path))
    weights = np.zeros(nr_atlas_images)
    predictions = np.zeros((nr_atlas_images, 86, 333, 271))

    for i, atlas_img_name in enumerate(all_training_image_names):
        print(f"{atlas_img_name}", end="\t", flush=True)
        atlas_mr_img_path = f"{TRAINING_DATA_PATH}/{atlas_img_name}/mr_bffe.mhd"
        atlas_pros_img_path = f"{TRAINING_DATA_PATH}/{atlas_img_name}/prostaat.mhd"
        transform = get_transform(valid_img_path, atlas_mr_img_path)
        transformed_atlas_mr_img = get_transformed_image(atlas_mr_img_path, transform)
        predictions[i] = get_transformed_image(atlas_pros_img_path, transform)
        weights[i] = metrics.nmi(valid_img, transformed_atlas_mr_img)
    weights = (weights - np.min(weights)) ** 2
    prediction = np.zeros((86, 333, 271))
    for i in range(nr_atlas_images):
        prediction += predictions[i] * weights[i]
    write_mhd(valid_img_name, prediction)
    print(time.time() - begin_time)
