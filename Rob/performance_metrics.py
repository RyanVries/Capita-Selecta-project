import numpy as np

def dice(img1, img2):
    num = 2.0 * np.sum(np.logical_and(img1, img2).astype(np.double))
    den = np.sum(img1.astype(np.double)) + np.sum(img2.astype(np.double))
    return num / den

def nmi(img1, img2, bins=32):
    img1_flat = img1.flatten().astype(np.double)
    img2_flat = img2.flatten().astype(np.double)
    min_value = np.min([np.min(img1_flat), np.min(img2_flat)])
    max_value = np.max([np.max(img1_flat), np.max(img2_flat)])
    rang = max_value - min_value
    img1n = np.divide(img1_flat - min_value, rang)
    img2n = np.divide(img2_flat - min_value, rang)
    img1bin = np.round(img1n * (bins - 1))
    img2bin = np.round(img2n * (bins - 1))
    p = np.histogram2d(img1bin, img2bin, bins)[0]
    p += 1e-9
    p = p / np.sum(p)
    p_i = np.sum(p, axis=1)
    p_j = np.sum(p, axis=0)
    num = np.sum(p_i * np.log2(p_i)) + np.sum(p_j * np.log2(p_j))
    den = np.sum(np.multiply(p, np.log2(p)))
    return num / den

def ncc(im1, im2):
    im1_flat = im1.flatten().astype(np.double)
    im2_flat = im2.flatten().astype(np.double)
    im1_av = np.sum(im1_flat) / len(im1_flat)
    im2_av = np.sum(im1_flat) / len(im2_flat)

    num = np.sum((im1_flat - im1_av) * (im2_flat - im2_av))
    den = (np.sum((im1_flat - im1_av) ** 2) * np.sum((im2_flat - im2_av) ** 2)) ** .5
    return num / den

def msd(img1, img2):
    diff = np.square((img1 - img2).astype(np.double))
    tot_sum = np.sum(diff)
    return tot_sum / diff.size
