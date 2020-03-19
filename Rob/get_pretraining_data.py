from SimpleITK import GetArrayFromImage, ReadImage
from scipy.ndimage.interpolation import zoom
import numpy as np

base_dir = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project\deep_learning_data"
pretraining_dir = f"{base_dir}/PretrainingData"

def get_pretraining_data(input_pretraining_dir, input_sample_shape=(16, 80, 64)):
    nr_imgs = 50
    full_shape = tuple([nr_imgs] + list(input_sample_shape))
    resized_mr_imgs = np.zeros(full_shape)
    resized_seg_imgs = np.zeros(full_shape)
    for i in range(nr_imgs):
        full_mr_img = GetArrayFromImage(ReadImage(f"{input_pretraining_dir}/Case{i:02}.mhd"))
        full_seg_img = GetArrayFromImage(ReadImage(f"{input_pretraining_dir}/Case{i:02}_segmentation.mhd"))
        full_img_shape = full_mr_img.shape
        zoom_factor = (input_sample_shape[0] / full_img_shape[0], input_sample_shape[1] / full_img_shape[1], input_sample_shape[2] / full_img_shape[2])
        resized_mr_imgs[i] = zoom(full_mr_img, zoom_factor, order=1)
        resized_seg_imgs[i] = zoom(full_seg_img, zoom_factor, order=1)
    return resized_mr_imgs, resized_seg_imgs

get_pretraining_data(pretraining_dir)
