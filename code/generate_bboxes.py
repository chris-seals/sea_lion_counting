# Generate bounding boxes around the sea lions in the dotted images
# These boxes are turned into image files and saved off in a folder structure
import os
import sys

sys.path.append(os.getcwd()+'/code/')

from slf import slf

if __name__ == '__main__':

    # Create image filenames
    mismatched = [3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 268, 290, 311, 331, 344, 380, 384, 406, 421,
                  469, 475, 490, 499, 507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 779, 781, 794, 800,
                  811, 839, 840, 869, 882, 901, 903, 905, 909, 913, 927, 946]
    filenames = [str(x)+'.jpg' for x in range(41,51) if x not in mismatched]
    #filenames = [str(x)+'.jpg' for x in range(20,50) if x not in mismatched]

    # Create coordinate dataframe
    coordinates = slf.create_coord_df(filenames)
    # Iterate through our images
    for file in filenames:
        if int(file.split('.')[0]) % 100 == 0:  # dont print every filename, just once every 100 for progress updates
            print(f'Processing {file}')
        # Extract the blobs from the marking dots
        blobs = slf.get_blobs(*slf.retrieve_image_paths(file))
        # Tally up our classes/coordinates based on the dot blob colors, append coords to the dataframe
        slf.count_classes(blobs, file, coordinates)

    # 1. check for directory existence
    # 2. if not existing, create it
    slf.create_chip_dir()


    # Create chips

    # 3. save off chip for each bbox, and write chip path
    slf.create_chips(coordinates, filenames)


    coordinates.to_csv('../results/coordinates.csv')


