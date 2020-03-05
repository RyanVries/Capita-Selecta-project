from SimpleITK import GetArrayFromImage, ReadImage
from scipy import ndimage
from mayavi import mlab
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2
import os

# Define paths
PROJECT_PATH = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project"
FOLDER_NAME = "translation-affine-parameters_test"
IMAGES_PATH = os.path.join(r"F:\_images_", FOLDER_NAME)

DATA_PATH = os.path.join(PROJECT_PATH, "training_data")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")

def get_xyz_from_edges(img, factor):
    img_coord = np.where(img == 1)
    random_pixels = np.random.sample(len(img_coord[0])) <= factor
    x = img_coord[0][random_pixels]
    y = img_coord[2][random_pixels] * 0.488281
    z = img_coord[1][random_pixels] * 0.488281
    return x, y, z

def get_edges(img):
    img -= ndimage.binary_erosion(img, np.ones((3, 3, 3)))
    return img

def show_figure(fixed_img_path, moving_img_path, single_fraction=1, both_fraction=1, scale_factor=1, opacity=0.1, mode="sphere", save_file_name=None):  # point, cube, sphere
    fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path))
    moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))
    both = np.logical_and(fixed_img, moving_img)

    fixed_img_edges = get_edges(fixed_img)
    moving_img_edges = get_edges(moving_img)
    fixed_img_edges -= np.logical_and(fixed_img_edges, both)
    moving_img_edges -= np.logical_and(moving_img_edges, both)
    if save_file_name is not None:
        mlab.figure(figure=f"Comparison of prostate segmentation", size=(1920, 1080), bgcolor=(1, 1, 1))
    else:
        mlab.figure(figure=f"Comparison of prostate segmentation", size=(1600, 900), bgcolor=(0, 0, 0))
    x, y, z = get_xyz_from_edges(fixed_img, single_fraction)
    mlab.points3d(x, y, z, mode=mode, resolution=4, scale_factor=scale_factor, color=(1, 0, 0), opacity=opacity)
    x, y, z = get_xyz_from_edges(moving_img, single_fraction)
    mlab.points3d(x, y, z, mode=mode, resolution=4, scale_factor=scale_factor, color=(0, 1, 0), opacity=opacity)
    x, y, z = get_xyz_from_edges(both, both_fraction)
    mlab.points3d(x, y, z, mode=mode, resolution=4, scale_factor=1, color=(0, 0, 1))
    mlab.view(225, 45, 250)
    if save_file_name is not None:
        mlab.savefig(save_file_name)
        remove_white_borders(save_file_name)
        mlab.close()
    else:
        mlab.show()

def show_image_save(input_img, save_file_name=None):
    plt.figure()
    plt.imshow(input_img, cmap='gray')
    if save_file_name is not None:
        cv2.imwrite(save_file_name, input_img)
    plt.ion()
    plt.show()

def show_slices(input_fixed_img_name, input_moving_img_name, input_slice_nr=40, save=True):
    if save:
        temp_path = f"{RESULTS_PATH}/slices/{input_fixed_img_name}-{input_moving_img_name}"
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        temp_path = f"{temp_path}/{input_slice_nr}"
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
    bits = 11
    fixed_mr_img_slice = GetArrayFromImage(ReadImage(f"{DATA_PATH}/{input_fixed_img_name}/mr_bffe.mhd"))[input_slice_nr, :, :] / 2 ** bits * 255
    fixed_pros_img_slice = GetArrayFromImage(ReadImage(f"{DATA_PATH}/{input_fixed_img_name}/prostaat.mhd"))[input_slice_nr, :, :] * 255
    moving_mr_img_slice = GetArrayFromImage(ReadImage(f"{DATA_PATH}/{input_moving_img_name}/mr_bffe.mhd"))[input_slice_nr, :, :] / 2 ** bits * 255
    moving_pros_img_slice = GetArrayFromImage(ReadImage(f"{DATA_PATH}/{input_moving_img_name}/prostaat.mhd"))[input_slice_nr, :, :] * 255
    # transformed_moving_mr_img_slice = GetArrayFromImage(ReadImage(f"{IMAGES_PATH}/{fixed_img_name}-{moving_img_name}/2/mr/result.mhd"))[input_slice_nr, :, :] / 2 ** bits * 255
    # transformed_moving_pros_img_slice = GetArrayFromImage(ReadImage(f"{IMAGES_PATH}/{fixed_img_name}-{moving_img_name}/2/pros/result.mhd"))[input_slice_nr, :, :] * 255
    for i, img_slice in enumerate([fixed_mr_img_slice, fixed_pros_img_slice, moving_mr_img_slice, moving_pros_img_slice]):#, transformed_moving_mr_img_slice, transformed_moving_pros_img_slice]):
        if save:
            show_image_save(img_slice, f"{temp_path}/{names[i]}_{input_fixed_img_name}_{input_moving_img_name}_{input_slice_nr}.png")
        else:
            show_image_save(img_slice)

def remove_white_borders(input_img_path):
    img = cv2.imread(input_img_path)
    negative = 255 * 3 - np.sum(img, 2)
    ones_x = np.where((np.sum(negative, 0) > 0) == 1)
    ones_y = np.where((np.sum(negative, 1) > 0) == 1)
    cv2.imwrite(input_img_path, img[np.min(ones_y):np.max(ones_y) + 1, np.min(ones_x):np.max(ones_x) + 1, :])

names = ("fixed_mr", "fixed_pros", "moving_mr", "moving_pros", "transformed_mr", "transformed_pros")
imgs = "p128p102"
fixed_img_name = imgs[:4]
moving_img_name = imgs[-4:]
show_slices(fixed_img_name, moving_img_name, 40)
# save_path = f"{RESULTS_PATH}/slices/{fixed_img_name}-{moving_img_name}/{fixed_img_name}_{moving_img_name}_3d_"
# show_figure(f"{DATA_PATH}/{fixed_img_name}/prostaat.mhd", f"{DATA_PATH}/{moving_img_name}/prostaat.mhd", save_file_name=f"{save_path}before.png")
# show_figure(f"{DATA_PATH}/{fixed_img_name}/prostaat.mhd", f"{IMAGES_PATH}/{fixed_img_name}-{moving_img_name}/2/pros/result.mhd", save_file_name=f"{save_path}after.png")
