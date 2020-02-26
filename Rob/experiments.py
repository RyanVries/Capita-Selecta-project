from SimpleITK import GetArrayFromImage, ReadImage
import performance_metrics as pm
import numpy as np
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

def get_predetermined_weights(input_leave_out_img):
    print(input_leave_out_img)
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_leave_out_img)
    return_weights = np.zeros(14)
    for fixed_img_name in temp_img_names:
        fixed_img = get_original_image(fixed_img_name)
        for i, moving_img_name in enumerate(temp_img_names):
            if fixed_img_name is not moving_img_name:
                return_weights[i] += pm.dice(fixed_img, get_transformed_image(fixed_img_name, moving_img_name)) / 14
        print(np.array2string(return_weights, separator="   ")[1:-1].replace("\n", "   "))
    return return_weights

def get_normalized_nmi_weight(input_fixed_img, input_moving_img):
    nmi = pm.nmi(input_fixed_img, input_moving_img)
    return (nmi - 1.006) / 0.069

def get_normalized_ncc_weight(input_fixed_img, input_moving_img):
    ncc = pm.ncc(input_fixed_img, input_moving_img)
    return (ncc + 0.025) / 0.67

def get_live_weights(input_fixed_img_name, weight_func):
    print(input_fixed_img_name)
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_fixed_img_name)
    return_weights = np.zeros(14)
    fixed_img = get_original_image(input_fixed_img_name, img_type="mr_bffe")
    for i, moving_img_name in enumerate(temp_img_names):
        return_weights[i] = weight_func(fixed_img, get_transformed_image(input_fixed_img_name, moving_img_name, img_type="mr"))
    print(np.array2string(return_weights, separator="   ")[1:-1].replace("\n", "   "))
    return return_weights

def get_dice_for_img_weights(input_test_img, input_weights):
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_test_img)
    prediction = np.zeros((86, 333, 271))
    for i, moving_img_name in enumerate(temp_img_names):
        prediction += get_transformed_image(input_test_img, moving_img_name) * input_weights[i]
    final_dice = pm.dice(get_original_image(input_test_img), prediction > np.max(prediction) * 0.5)
    print(f"dice: {final_dice}\n")
    return final_dice

np.set_printoptions(precision=3)
all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]

# ###############################################################################################
# total_dice = 0
# for test_img in all_image_names:
#     dice = get_dice_for_img_weights(test_img, np.ones(14))
#     total_dice += dice
# print(f"final dice: {total_dice / 15}")
#
# ###############################################################################################
# total_dice = 0
# for test_img in all_image_names:
#     weights = get_predetermined_weights(test_img)
#     dice = get_dice_for_img_weights(test_img, weights)
#     total_dice += dice
# print(f"final dice: {total_dice / 15}")
#
# ###############################################################################################
# total_dice = 0
# for test_img in all_image_names:
#     weights = get_live_weights(test_img, get_normalized_nmi_weight)
#     dice = get_dice_for_img_weights(test_img, weights)
#     total_dice += dice
# print(f"final dice: {total_dice / 15}")

###############################################################################################
total_dice = 0
for test_img in all_image_names:
    weights = get_live_weights(test_img, get_normalized_ncc_weight)
    dice = get_dice_for_img_weights(test_img, weights)
    total_dice += dice
print(f"final dice: {total_dice / 15}")
