# Generate bounding boxes around the sea lions in the dotted images
# These boxes are turned into image files and saved off in a folder structure

import cv2
import numpy as np
import pandas as pd
import slf



# Create image filenames
filenames = [str(x)+'.jpg' for x in range(41,51)]

# Create coordinate dataframe
coordinates = slf.create_df(filenames)
# Generate blobs
for file in filenames:
    blobs = slf.get_blobs(*slf.retrieve_image_paths(file))
    slf.count_classes(blobs, file, coordinates)

print(coordinates)
# Extract dot locations

# Create locations of sea lions
# Iterate through the locations, extract bounding box by type of sea lion
# 1. check for directory existence
# 2. if not existing, create it
# 3. save off chip for each bbox, and write chip path
