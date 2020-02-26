from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import metrics
import os

# Define paths
PROJECT_PATH = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project"
FOLDER_NAME = "translation-affine-parameters_test"
IMAGES_PATH = os.path.join(r"F:\_images_", FOLDER_NAME)

DATA_PATH = os.path.join(PROJECT_PATH, "training_data")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
max_iteration = FOLDER_NAME.count("-")

def get_transformed_image(input_fixed_img, input_moving_img, img_type="pros", iteration=max_iteration):
    temp_dir = f"{IMAGES_PATH}/{input_fixed_img}-{input_moving_img}"
    image_path = f"{temp_dir}/{iteration}/{img_type}/result.mhd"
    return GetArrayFromImage(ReadImage(image_path))

def get_original_image(input_image, img_type="prostaat"):
    image_path = f"{DATA_PATH}/{input_image}/{img_type}.mhd"
    return GetArrayFromImage(ReadImage(image_path))

def get_normalized_nmi_weight(input_fixed_img, input_moving_img):
    nmi = metrics.nmi(input_fixed_img, input_moving_img)
    return nmi - 1.006

def get_normalized_ncc_weight(input_fixed_img, input_moving_img):
    ncc = metrics.ncc(input_fixed_img, input_moving_img)
    return ncc + 0.025

def write_to_txt(write_string):
    f = open(f"{RESULTS_PATH}/{FOLDER_NAME}_.txt", 'a')
    f.write(f"{write_string}\n")
    f.close()

def get_predetermined_weights(input_leave_out_img):
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_leave_out_img)
    return_weights = np.zeros(14)
    for fixed_img_name in temp_img_names:
        fixed_img = get_original_image(fixed_img_name)
        for i, moving_img_name in enumerate(temp_img_names):
            if fixed_img_name is not moving_img_name:
                return_weights[i] += metrics.dice(fixed_img, get_transformed_image(fixed_img_name, moving_img_name)) / 14
    print_string = np.array2string(return_weights, separator="\t")[1:-1].replace("\n", "\t")
    print(print_string, end="\t| ", flush=True)
    write_to_txt(print_string)
    return return_weights

def get_live_weights(input_fixed_img_name, weight_func):
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_fixed_img_name)
    return_weights = np.zeros(14)
    fixed_img = get_original_image(input_fixed_img_name, img_type="mr_bffe")
    for i, moving_img_name in enumerate(temp_img_names):
        return_weights[i] = weight_func(fixed_img, get_transformed_image(input_fixed_img_name, moving_img_name, img_type="mr"))
    print(np.array2string(return_weights, separator="   ")[1:-1].replace("\n", "   "), end="\t| ", flush=True)

    return return_weights

def get_dice_for_img_weights(input_test_img, input_weights):
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_test_img)
    prediction = np.zeros((86, 333, 271))
    for i, moving_img_name in enumerate(temp_img_names):
        prediction += get_transformed_image(input_test_img, moving_img_name) * input_weights[i]
    original_img = get_original_image(input_test_img)
    # for threshold in np.arange(0.0, 1.0, 0.025):
    threshold = 0.5
    final_dice = metrics.dice(original_img, prediction > (np.sum(input_weights) * threshold))
    print(f"\t{final_dice:.3f}", end="", flush=True)
    print("")
    return final_dice

np.set_printoptions(precision=3)
all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]

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
predetermined_weights_matrix[predetermined_weights_matrix < 0.5] = 0
###############################################################################################
# total_dice = 0
# for test_img in all_image_names:
#     print(test_img, end=" | ", flush=True)
#     dice = get_dice_for_img_weights(test_img, np.ones(14))
#     total_dice += dice
# print(f"final dice: {total_dice / 15}\n")

###############################################################################################
total_dice = 0
for i, test_img in enumerate(all_image_names):
    print(test_img, end=" | ", flush=True)
    # weights = get_predetermined_weights(test_img)
    weights = predetermined_weights_matrix[i]
    dice = get_dice_for_img_weights(test_img, weights)
    total_dice += dice
print(f"final dice: {total_dice / 15}\n")

# ###############################################################################################
# total_dice = 0
# for test_img in all_image_names:
#     print(test_img, end=" | ", flush=True)
#     weights = get_live_weights(test_img, get_normalized_nmi_weight)
#     dice = get_dice_for_img_weights(test_img, weights)
#     total_dice += dice
# print(f"final dice: {total_dice / 15}\n")
#
# ###############################################################################################
# total_dice = 0
# for test_img in all_image_names:
#     print(test_img, end=" | ", flush=True)
#     weights = get_live_weights(test_img, get_normalized_ncc_weight)
#     dice = get_dice_for_img_weights(test_img, weights)
#     total_dice += dice
# print(f"final dice: {total_dice / 15}\n")
