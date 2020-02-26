from scipy import ndimage
from mayavi import mlab
import SimpleITK as sitk
import numpy as np
import os

data_dir = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project\training_data"

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

def show_figure(fixed_img_path, moving_img_path, single_fraction=1, both_fraction=1, scale_factor=1, opacity=0.1, mode="sphere"):  # point, cube, sphere
    fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path))
    moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))
    both = np.logical_and(fixed_img, moving_img)

    fixed_img_edges = get_edges(fixed_img)
    moving_img_edges = get_edges(moving_img)
    fixed_img_edges -= np.logical_and(fixed_img_edges, both)
    moving_img_edges -= np.logical_and(moving_img_edges, both)

    mlab.figure(figure=f"Comparison of prostate segmentations", size=(1600, 900))
    x, y, z = get_xyz_from_edges(fixed_img, single_fraction)
    mlab.points3d(x, y, z, mode=mode, resolution=4, scale_factor=scale_factor, color=(1, 0, 0), opacity=opacity)
    x, y, z = get_xyz_from_edges(moving_img, single_fraction)
    mlab.points3d(x, y, z, mode=mode, resolution=4, scale_factor=scale_factor, color=(0, 1, 0), opacity=opacity)
    x, y, z = get_xyz_from_edges(both, both_fraction)
    mlab.points3d(x, y, z, mode=mode, resolution=4, scale_factor=1, color=(1, 1, 1))
    mlab.show()

# all_imgs = ["p102", "p107", "p108", "p109", "p113", "p116", "p117", "p119", "p120", "p123", "p125", "p128", "p129", "p133", "p135"]
# fixed_image_name, moving_image_name = np.random.choice(all_imgs, 2, replace=False)
# show_figure(fixed_image_name, moving_image_name)

# 1 translate/rigid als eerste verschil
# 2 partial predetermined weights + part MI score
# 3 elke registratie: check MI, als lager dan ervoor, skip, pak hoogste MI

# show_figure(f"{data_dir}/p113/prostaat.mhd", f"{data_dir}/p117/prostaat.mhd")
show_figure(f"{data_dir}/p113/prostaat.mhd", r"F:\_images_\translation-affine-parameters_test\p113-p117\2\pros\result.mhd")
