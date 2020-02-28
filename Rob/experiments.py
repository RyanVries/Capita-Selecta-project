from SimpleITK import GetArrayFromImage, ReadImage
from predetermined_weights import predetermined_weights_matrix
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
    return nmi - 1.00591886842159

def write_to_txt(write_string):
    f = open(f"{RESULTS_PATH}/{FOLDER_NAME}=experiments.txt", 'a')
    f.write(f"{write_string}\n")
    f.close()

def get_live_weights(input_fixed_img_name, weight_func):
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_fixed_img_name)
    return_weights = np.zeros(14)
    fixed_img = get_original_image(input_fixed_img_name, img_type="mr_bffe")
    for index, moving_img_name in enumerate(temp_img_names):
        return_weights[index] = weight_func(fixed_img, get_transformed_image(input_fixed_img_name, moving_img_name, img_type="mr"))
    return return_weights

def squared(input_values):
    return input_values ** 2

def get_prediction_from_weights(input_test_img, input_weights):
    temp_img_names = all_image_names.copy()
    temp_img_names.remove(input_test_img)
    prediction = np.zeros((86, 333, 271))
    for index, moving_img_name in enumerate(temp_img_names):
        prediction += get_transformed_image(input_test_img, moving_img_name) * input_weights[index]
    return prediction

def get_threshold_dices(input_img, input_prediction, input_weights):
    dice_array = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        dice_array[i] = metrics.dice(input_img, input_prediction > (np.sum(input_weights) * threshold))
    return dice_array

def print_threshold_dice_values(input_dice_matrix):
    average_dice = np.average(input_dice_matrix, 0)
    average_dice_string = np.array2string(average_dice, separator="\t")[1:-1].replace("\n", "\t")
    max_index = np.argmax(average_dice)
    final_string = f"{average_dice_string}\t{thresholds[max_index]}\t{average_dice[max_index]}"
    write_to_txt(final_string)

def print_threshold_dice_value(input_dice_array):
    final_string = np.array2string(input_dice_array, separator="\t")[1:-1].replace("\n", "\t")
    write_to_txt(final_string)

##########################################################
def loop_all_default():
    dice_matrix = np.zeros((len(all_image_names), len(thresholds)))
    weights = np.ones(14)
    for i, test_img_name in enumerate(all_image_names):
        print(".", end="", flush=True)
        test_img = get_original_image(test_img_name)
        prediction = get_prediction_from_weights(test_img_name, weights)
        dice_matrix[i] = get_threshold_dices(test_img, prediction, weights)
    print("")
    print_threshold_dice_values(dice_matrix)

def loop_all_nmi(input_func):
    dice_matrix = np.zeros((len(all_image_names), len(thresholds)))
    for i, test_img_name in enumerate(all_image_names):
        print(".", end="", flush=True)
        test_img = get_original_image(test_img_name)
        weights = get_live_weights(test_img_name, get_normalized_nmi_weight)
        if input_func is not None:
            weights = input_func(weights)
        prediction = get_prediction_from_weights(test_img_name, weights)
        dice_matrix[i] = get_threshold_dices(test_img, prediction, weights)
    print("")
    print_threshold_dice_values(dice_matrix)

def loop_all_predetermined(normalize, input_func):
    dice_matrix = np.zeros((len(all_image_names), len(thresholds)))
    for i, test_img_name in enumerate(all_image_names):
        print(".", end="", flush=True)
        test_img = get_original_image(test_img_name)
        weights = predetermined_weights_matrix[i]
        if normalize:
            weights -= np.min(weights)
        if input_func is not None:
            weights = input_func(weights)
        prediction = get_prediction_from_weights(test_img_name, weights)
        dice_matrix[i] = get_threshold_dices(test_img, prediction, weights)
    print("")
    print_threshold_dice_values(dice_matrix)

##########################################################
def loop_all_default_threshold(threshold):
    dice_array = np.zeros(len(all_image_names))
    weights = np.ones(14)
    for i, test_img_name in enumerate(all_image_names):
        print(".", end="", flush=True)
        test_img = get_original_image(test_img_name)
        prediction = get_prediction_from_weights(test_img_name, weights)
        dice_array[i] = metrics.dice(test_img, prediction > (np.sum(weights) * threshold))
    print("")
    print_threshold_dice_value(dice_array)

def loop_all_nmi_threshold(input_func, threshold):
    dice_array = np.zeros(len(all_image_names))
    for i, test_img_name in enumerate(all_image_names):
        print(".", end="", flush=True)
        test_img = get_original_image(test_img_name)
        weights = get_live_weights(test_img_name, get_normalized_nmi_weight)
        if input_func is not None:
            weights = input_func(weights)
        prediction = get_prediction_from_weights(test_img_name, weights)
        dice_array[i] = metrics.dice(test_img, prediction > (np.sum(weights) * threshold))
    print("")
    print_threshold_dice_value(dice_array)

def loop_all_predetermined_threshold(normalize, input_func, threshold):
    dice_array = np.zeros(len(all_image_names))
    for i, test_img_name in enumerate(all_image_names):
        print(".", end="", flush=True)
        test_img = get_original_image(test_img_name)
        weights = predetermined_weights_matrix[i]
        if normalize:
            weights -= np.min(weights)
        if input_func is not None:
            weights = input_func(weights)
        prediction = get_prediction_from_weights(test_img_name, weights)
        dice_array[i] = metrics.dice(test_img, prediction > (np.sum(weights) * threshold))
    print("")
    print_threshold_dice_value(dice_array)

np.set_printoptions(precision=5)
thresholds = np.arange(0.0, 1.0, 0.01)
write_to_txt("------------------------------\n")  # + np.array2string(thresholds, separator="\t")[1:-1].replace("\n", "\t"))
all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]

# loop_all_default()
# loop_all_nmi(None)
# loop_all_nmi(squared)
# loop_all_predetermined(False, None)
# loop_all_predetermined(True, None)
# loop_all_predetermined(True, squared)

loop_all_default_threshold(0.49)
loop_all_nmi_threshold(None, 0.43)
loop_all_nmi_threshold(squared, 0.45)
loop_all_predetermined_threshold(False, None, 0.43)
loop_all_predetermined_threshold(True, None, 0.41)
loop_all_predetermined_threshold(True, squared, 0.44)

# loop_all_default_threshold(0.5)
# loop_all_nmi_threshold(None, 0.5)
# loop_all_nmi_threshold(squared, 0.5)
# loop_all_predetermined_threshold(False, None, 0.5)
# loop_all_predetermined_threshold(True, None, 0.5)
# loop_all_predetermined_threshold(True, squared, 0.5)
