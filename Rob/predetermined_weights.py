from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import metrics
import os

# Define paths
PROJECT_PATH = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project"
FOLDER_NAME = "translation-affine-parameters_test"
IMAGES_PATH = os.path.join(r"F:\_images_", FOLDER_NAME)

DATA_PATH = os.path.join(PROJECT_PATH, "training_data")

max_iteration = FOLDER_NAME.count("-")

def get_transformed_image(input_fixed_img, input_moving_img, img_type="pros", iteration=max_iteration):
    temp_dir = f"{IMAGES_PATH}/{input_fixed_img}-{input_moving_img}"
    image_path = f"{temp_dir}/{iteration}/{img_type}/result.mhd"
    return GetArrayFromImage(ReadImage(image_path))

def get_original_image(input_image, img_type="prostaat"):
    image_path = f"{DATA_PATH}/{input_image}/{img_type}.mhd"
    return GetArrayFromImage(ReadImage(image_path))

def get_predetermined_weights(input_img_names):
    return_weights = np.zeros(len(input_img_names))
    for fixed_img_name in input_img_names:
        fixed_img = get_original_image(fixed_img_name)
        for index, moving_img_name in enumerate(input_img_names):
            if fixed_img_name is not moving_img_name:
                return_weights[index] += metrics.dice(fixed_img, get_transformed_image(fixed_img_name, moving_img_name)) / 14
    print_string = np.array2string(return_weights, separator=",\t").replace("\n", "\t")
    print(print_string)
    return return_weights

np.set_printoptions(precision=3)
all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]

# get_predetermined_weights(all_image_names)
#
# for test_img in all_image_names:
#     temp_img_list = all_image_names.copy()
#     temp_img_list.remove(test_img)
#     get_predetermined_weights(temp_img_list)

####################################################################################################
final_predetermined_weights = np.array([0.789, 0.647, 0.65, 0.718, 0.578, 0.804, 0.52, 0.697, 0.773, 0.692, 0.713, 0.742, 0.797, 0.637, 0.704])

####################################################################################################
predetermined_weights_matrix = np.array([[0.591, 0.6, 0.665, 0.548, 0.748, 0.477, 0.643, 0.718, 0.631, 0.658, 0.694, 0.743, 0.581, 0.655],
                                         [0.733, 0.611, 0.668, 0.535, 0.749, 0.467, 0.647, 0.718, 0.641, 0.656, 0.691, 0.746, 0.59, 0.653],
                                         [0.731, 0.592, 0.664, 0.515, 0.74, 0.484, 0.634, 0.708, 0.692, 0.661, 0.684, 0.733, 0.583, 0.653],
                                         [0.738, 0.598, 0.595, 0.546, 0.745, 0.483, 0.649, 0.741, 0.645, 0.664, 0.699, 0.745, 0.583, 0.656],
                                         [0.735, 0.647, 0.585, 0.685, 0.742, 0.52, 0.683, 0.709, 0.655, 0.684, 0.681, 0.733, 0.618, 0.656],
                                         [0.727, 0.588, 0.585, 0.658, 0.52, 0.475, 0.634, 0.709, 0.629, 0.652, 0.682, 0.733, 0.575, 0.642],
                                         [0.738, 0.595, 0.622, 0.678, 0.555, 0.772, 0.679, 0.728, 0.651, 0.662, 0.701, 0.752, 0.596, 0.678],
                                         [0.732, 0.592, 0.595, 0.668, 0.539, 0.743, 0.477, 0.712, 0.636, 0.663, 0.681, 0.733, 0.589, 0.645],
                                         [0.734, 0.647, 0.589, 0.669, 0.534, 0.744, 0.483, 0.636, 0.632, 0.668, 0.678, 0.733, 0.632, 0.661],
                                         [0.729, 0.593, 0.596, 0.664, 0.543, 0.743, 0.477, 0.638, 0.714, 0.657, 0.688, 0.738, 0.581, 0.647],
                                         [0.736, 0.586, 0.65, 0.665, 0.545, 0.746, 0.495, 0.644, 0.722, 0.635, 0.689, 0.743, 0.583, 0.645],
                                         [0.73, 0.588, 0.588, 0.656, 0.531, 0.742, 0.468, 0.635, 0.71, 0.632, 0.656, 0.737, 0.579, 0.642],
                                         [0.73, 0.591, 0.586, 0.656, 0.517, 0.741, 0.483, 0.633, 0.708, 0.632, 0.659, 0.68, 0.598, 0.648],
                                         [0.732, 0.604, 0.626, 0.669, 0.54, 0.748, 0.489, 0.667, 0.72, 0.639, 0.66, 0.715, 0.747, 0.668],
                                         [0.737, 0.6, 0.62, 0.674, 0.552, 0.754, 0.482, 0.642, 0.728, 0.641, 0.662, 0.685, 0.75, 0.587]])
