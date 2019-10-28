import os
import sys
from glob import glob
import pandas as pd
from slf import slf
import cv2

sys.path.append(os.getcwd()+'/code/')

""" Code to rip the train and train-dotted images down to a smaller size for machine learning training prep."""

# Identify original image locations
train_dotted_path = r'../data/TrainSmall2/TrainDotted/'
train_path = r'../data/TrainSmall2/Train/'

# Identify output locations
small_dotted_path = r'../results/smaller_bbox_chips/TrainDotted/'
small_clean_path = r'../results/smaller_bbox_chips/Train/'



#### Run lob comparisons
#### Record coordinates
####    -- Save coordinates off as one dictionary per image - of centerpoints?
#### Create .xml file for each image with pascal VOC writer module
#### final desired inputs = directory of images and one .xml file for each image

orig_filenames = [file[-6:] for file in glob(train_path+'/*.jpg')]

for filename in orig_filenames:
    image_1 = cv2.imread("../../data/TrainSmall2/TrainDotted/" + filename)
    image_2 = cv2.imread("../../data/TrainSmall2/Train/" + filename)

