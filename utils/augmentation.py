import numpy as np
import cv2
import math
from numba import prange
from skimage.exposure import equalize_adapthist

def f_rgb2gray(img, need_not=False):
    """
    rgb2gray

    :param img: (CHANGE) np.array
    :param need_not: if need bitwise_not
    :return: np.array
    """
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if need_not:
            img = cv2.bitwise_not(img)
    return img

def f_percentile_threshold(img, percentile=99.9):
    """
    Threshold an image to reduce bright spots

    :param img: (CHANGE) numpy array of image data
    :param percentile: cutoff used to threshold image
    :return: np.array: thresholded version of input image
    """

    non_zero_vals = img[img > 0]

    if len(non_zero_vals) > 0:
        img_max = np.percentile(non_zero_vals, percentile, overwrite_input=True)

        threshold_mask = img > img_max
        img[threshold_mask] = img_max

    return img

def f_histogram_normalization(img):
    """
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :return: numpy.array:image data with dtype float32.
    """

    img = img.astype('float32')
    sample_value = img[(0,) * img.ndim]
    if (img == sample_value).all():
        return np.zeros_like(img)
    img = rescale_intensity_v2(img, out_range=(0.0, 1.0))
    return img

def f_equalize_adapthist(img, kernel_size=None):
    """
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :param kernel_size: (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    :return: numpy.array:Pre-processed image
    """
    if kernel_size is None:
        kernel_size = 128
    clahe = cv2.createCLAHE(clipLimit=2.56, tileGridSize=(math.ceil(img.shape[0] / kernel_size),
                                                          math.ceil(img.shape[1] / kernel_size)))
    img = clahe.apply(img)
    return img

def f_clahe_rgb(img, kernel_size=128):
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit=2.56, tileGridSize=(math.ceil(img.shape[0] / kernel_size),
                                                          math.ceil(img.shape[1] / kernel_size)))
    cl = clahe.apply(l_channel)

    merged_channels = cv2.merge((cl, a_channel, b_channel))

    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

def rescale_intensity_v2(img, out_range):
    imin = np.min(img)
    imax = np.max(img)
    _, omax = out_range

    for i in prange(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = ((img[i][j] - imin) / (imax - imin)) * omax
    return img
