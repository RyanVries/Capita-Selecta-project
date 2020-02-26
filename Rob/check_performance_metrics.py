from SimpleITK import GetArrayFromImage, ReadImage
import performance_metrics as pm
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

def write_to_txt(write_string):
    f = open(f"{RESULTS_PATH}/{FOLDER_NAME}.txt", 'a')
    f.write(f"{write_string}\n")
    f.close()

def get_transformed_image(input_fixed_img, input_moving_img, img_type="pros", iteration=max_iteration):
    temp_dir = f"{IMAGES_PATH}/{input_fixed_img}-{input_moving_img}"
    image_path = f"{temp_dir}/{iteration}/{img_type}/result.mhd"
    return GetArrayFromImage(ReadImage(image_path))

def get_metric_results(input_fixed_img, input_moving_img):
    og_fixed_img_mr = get_original_image(input_fixed_img, "mr_bffe")
    og_moving_img_mr = get_original_image(input_moving_img, "mr_bffe")
    og_fixed_img_pros = get_original_image(input_fixed_img, "prostaat")
    og_moving_img_pros = get_original_image(input_moving_img, "prostaat")
    return_string = f"{pm.dice(og_fixed_img_pros, og_moving_img_pros)}"
    for func in performance_metric_functions:
        return_string += f"\t{func(og_fixed_img_mr, og_moving_img_mr)}"
    for i in range(max_iteration + 1):
        tf_moving_img_mr = get_transformed_image(input_fixed_img, input_moving_img, img_type="mr", iteration=i)
        tf_moving_img_pros = get_transformed_image(input_fixed_img, input_moving_img, img_type="pros", iteration=i)
        return_string += f"\t{pm.dice(og_fixed_img_pros, tf_moving_img_pros)}"
        for func in performance_metric_functions:
            return_string += f"\t{func(og_fixed_img_mr, tf_moving_img_mr)}"
    print(return_string)
    return return_string

def get_original_image(input_image, img_type="prostaat"):
    image_path = f"{DATA_PATH}/{input_image}/{img_type}.mhd"
    return GetArrayFromImage(ReadImage(image_path))

all_image_names = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]
performance_metric_functions = [pm.nmi, pm.ncc]  # , pm.msd]

for fixed_img_name in all_image_names:
    for moving_img_name in all_image_names:
        if fixed_img_name is not moving_img_name:
            results_string = get_metric_results(fixed_img_name, moving_img_name)
            write_to_txt(results_string)
