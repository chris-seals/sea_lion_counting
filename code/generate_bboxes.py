# Generate bounding boxes around the sea lions in the dotted images
# These boxes are turned into image files and saved off in a folder structure

import cv2
import numpy as np
import pandas as pd
import sea_lion_fun

# Identify image file locations
train_dotted_path = r'../data/TrainSmall2/TrainDotted/'
train_path = r'../data/TrainSmall2/Train/'

# Create image filenames
filenames = [str(x)+'.jpg' for x in range(41,51)]

for file in filenames:
    sea_lion_fun.get_blobs(train_dotted_path+file, train_path+file)


# Extract dot locations

# Create locations of sea lions
# Iterate through the locations, extract bounding box by type of sea lion
# 1. check for directory existence
# 2. if not existing, create it
# 3. save off chip for each bbox, and write chip path
