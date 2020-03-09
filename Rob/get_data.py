from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import glob

def get_data_array(data_dir, shuffle=False):
    path_array = glob.glob(f"{data_dir}/*")
    if shuffle:
        np.random.shuffle(path_array)
    nr_imgs = len(path_array)
    data_array = np.zeros((nr_imgs, 86, 333, 271))
    label_array = np.zeros((nr_imgs, 86, 333, 271))
    for i, path in enumerate(path_array):
        data_array[i] = GetArrayFromImage(ReadImage(f"{path}/mr_bffe.mhd"))
        label_array[i] = GetArrayFromImage(ReadImage(f"{path}/prostaat.mhd"))
    return data_array, label_array

data, label = get_data_array(r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project\training_data")
