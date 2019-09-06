# Generate bounding boxes around the sea lions in the dotted images
# These boxes are turned into image files and saved off in a folder structure

import cv2
import numpy as np
import pandas as pd
import slf



# Create image filenames
filenames = [str(x)+'.jpg' for x in range(41,51)]

# Create coordinate dataframe
coordinates = slf.create_coord_df(filenames)
# Iterate through our images
for file in filenames:
    print(f'Processing {file}')
    # Extract the blobs from the marking dots
    blobs = slf.get_blobs(*slf.retrieve_image_paths(file))
    # Tally up our classes/coordinates based on the dot blob colors, append coords to the dataframe
    slf.count_classes(blobs, file, coordinates)

# 1. check for directory existence
# 2. if not existing, create it
slf.create_chip_dir()

slf.create_chips(coordinates)
# Create chips

# 3. save off chip for each bbox, and write chip path

coordinates.to_csv('../results/coordinates.csv')


