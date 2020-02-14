from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import sampling
import bspline
import multRes
import jacobian
import registration_assignments


def get_el(ELASTIX_PATH):
    if not os.path.exists(ELASTIX_PATH):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    # Define a new elastix object 'el' with the correct path to elastix
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    return el


def get_images(path, p_no, is_mask):
    """
    Gets the image of the patient
    :param path: path to data folder
    :param p_no: patient_number
    :param is_mask: boolean is mask or not
    :return: image: array
    """
    mask = 'mr_bffe.mhd'
    if is_mask:
        mask = 'prostaat.mhd'

    # fixed image
    image_path = os.path.join(path, p_no, mask)
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

    return image_path, image


def visualize(fix_im, fix_im_m, mov_im, mov_im_m, tr_im, tr_im_m):
    slice_id = 50

    fig, ax = plt.subplots(2, 3, figsize=(10, 15))
    ax[0, 0].imshow(fix_im[slice_id, :, :], cmap='gray')
    ax[0, 0].set_title('Fixed Image')
    ax[1, 0].imshow(fix_im_m[slice_id, :, :], cmap='gray')
    ax[1, 0].set_title('Fixed Mask')
    ax[0, 1].imshow(mov_im[slice_id, :, :], cmap='gray')
    ax[0, 1].set_title('Moving Image')
    ax[1, 1].imshow(mov_im_m[slice_id, :, :], cmap='gray')
    ax[1, 1].set_title('Moving Mask')
    ax[0, 2].imshow(tr_im[slice_id, :, :])
    ax[0, 2].set_title('Transformed moving image')
    ax[1, 2].imshow(tr_im_m[slice_id, :, :], cmap='gray')
    ax[1, 2].set_title('Transformed mask')

    pos_fixed = np.where(fix_im_m == 1)
    pos_moving = np.where(mov_im_m == 1)
    pos_transformed = np.where(tr_im_m == 1)

    fig = plt.figure(figsize=plt.figaspect(0.33))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(pos_fixed[0], pos_fixed[1], pos_fixed[2], c='grey')
    ax.set_title('Fixed')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(pos_moving[0], pos_moving[1], pos_moving[2], c='grey')
    ax.set_title('Moving')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(pos_transformed[0], pos_transformed[1], pos_transformed[2], c='grey')
    ax.set_title('Transformed')
    plt.show()

    return


def registration(parameters, fix_im_p, mov_im_p, mov_im_m_p, name):
    ELASTIX_PATH = os.path.join(r'C:\Users\s148534\PycharmProjects\myfolder\elastix.exe')
    TRANSFORMIX_PATH = os.path.join(r'C:\Users\s148534\PycharmProjects\myfolder\transformix.exe')
    if not os.path.exists(ELASTIX_PATH):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    if not os.path.exists(TRANSFORMIX_PATH):
        raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

    # Define a new elastix object 'el' with the correct path to elastix
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

    # Make a results directory if none exists
    if not os.path.exists('results'+name):
        os.mkdir('results'+name)

    # # Execute the registration. Make sure the paths below are correct, and
    # that the results folder exists from where you are running this script
    parm = parameters

    el.register(
        fixed_image=fix_im_p,
        moving_image=mov_im_p,
        parameters=[os.path.join(parm)],
        output_dir='results'+name)

    # Find the results
    transform_path = os.path.join('results'+name, 'TransformParameters.0.txt')
    tr_im_p = os.path.join('results'+name, 'result.0.mhd')

    # Make a new transformix object tr with the CORRECT PATH to transformix
    tr = elastix.TransformixInterface(parameters=transform_path,
                                      transformix_path=TRANSFORMIX_PATH)

    # Make a results directory if none exists
    if not os.path.exists('results_tr'+name):
        os.mkdir('results_tr'+name)

    tr_im_m_p = tr.transform_image(mov_im_m_p, output_dir=r'results_tr'+name)

    return tr_im_p, tr_im_m_p

