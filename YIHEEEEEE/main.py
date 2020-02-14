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
import functions as func

# bspline.bspline()

# Get elastix object
ELASTIX_PATH = os.path.join(r'C:\Users\s148534\PycharmProjects\myfolder\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Users\s148534\PycharmProjects\myfolder\transformix.exe')
el = func.get_el(ELASTIX_PATH)

# get images
path = 'TrainingData'
p_no_fix = 'p102'
p_no_mov = 'p108'

fix_im_p = os.path.join(path, p_no_fix, 'mr_bffe.mhd')
fix_im = sitk.GetArrayFromImage(sitk.ReadImage(fix_im_p))

fix_im_m_p = os.path.join(path, p_no_fix, 'prostaat.mhd')
fix_im_m = sitk.GetArrayFromImage(sitk.ReadImage(fix_im_m_p))

mov_im_p = os.path.join(path, p_no_mov, 'mr_bffe.mhd')
mov_im = sitk.GetArrayFromImage(sitk.ReadImage(mov_im_p))

mov_im_m_p = os.path.join(path, p_no_mov, 'prostaat.mhd')
mov_im_m = sitk.GetArrayFromImage(sitk.ReadImage(mov_im_m_p))

# fix_im_p, fix_im = func.get_images(path, p_no_fix, False)
# fix_im_m_p, fix_im_m = func.get_images(path, p_no_fix, True)
# mov_im_p, mov_im = func.get_images(path, p_no_mov, False)
# mov_im_m_p, mov_im_m = func.get_images(path, p_no_mov, True)



rigid = 'parameters_rigid.txt'
bspline64 = 'parameters_bspline64.txt'
bspline32 = 'parameters_bspline32.txt'
bspline16 = 'parameters_bspline16.txt'
bspline8 = 'parameters_bspline8.txt'
tr_im_p, tr_im_m_p = func.registration(rigid, fix_im_p, mov_im_p, mov_im_m_p, '_rigid')
tr_im_p, tr_im_m_p = func.registration(bspline64, fix_im_p, tr_im_p, tr_im_m_p, '_bspline64')
# tr_im_p, tr_im_m_p = func.registration(bspline32, fix_im_p, tr_im_p, tr_im_m_p, '_bspline32')
# tr_im_p, tr_im_m_p = func.registration(bspline16, fix_im_p, tr_im_p, tr_im_m_p, '_bspline16')
# tr_im_p, tr_im_m_p = func.registration(bspline8, fix_im_p, tr_im_p, tr_im_m_p, '_bspline8')

tr_im = sitk.GetArrayFromImage(sitk.ReadImage(tr_im_p))
tr_im_m = sitk.GetArrayFromImage(sitk.ReadImage(tr_im_m_p))

dice = ((2.0 * np.sum(np.logical_and(mov_im_m, fix_im_m))) / (np.sum(mov_im_m) + np.sum(fix_im_m)))
print('Dice similarity score is {}'.format(dice))

dice = ((2.0 * np.sum(np.logical_and(tr_im_m, fix_im_m))) / (np.sum(tr_im_m) + np.sum(fix_im_m)))
print('Dice similarity score is {}'.format(dice))

func.visualize(fix_im, fix_im_m, mov_im, mov_im_m, tr_im, tr_im_m)

print('finished')

