# Supporting functions for sea lion project

import cv2
import numpy as np
from skimage import feature
import sys

def get_blobs(dotted_image:str, clean_image:str):
    """ Given two images - one dotted and its clean match - return blobs """
    r = 0.4     #scale down
    width = 100 #patch size

    # Grab arguments
    dotted_image = cv2.imread(dotted_image)
    clean_image = cv2.imread(clean_image)

    # absolute difference between Train and Train Dotted -- use to isolate dot locations
    dot_diff_image = cv2.absdiff(dotted_image,clean_image)
    mask_1 = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    dots_only_image = cv2.bitwise_or(dot_diff_image, dot_diff_image, mask=mask_1)

    mask_1 = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    dots_only_max_image = np.max(dots_only_image,axis=2)

    # detect blobs
    blobs = feature.blob_log(dots_only_max_image, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h,w,d = clean_image.shape # (3328, 4992, 3)

    res=np.zeros((int((w*r)//width)+1,int((h*r)//width)+1,5), dtype='int16')

    return blobs