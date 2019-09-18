# Supporting functions for sea lion project

# SLF - Sea Lion Functions

import cv2
import numpy as np
import pandas as pd
from skimage import feature
from tqdm import tqdm

# define scaling and width size for initial
r = 0.4     #scale down
width = 100 #patch size

chip_sizes = {
    'adult_females':80,
    'adult_males':120,
    'juveniles':60,
    'pups':40,
    'subadult_males':100
}

# Identify image file locations
train_dotted_path = r'../data/TrainSmall2/TrainDotted/'
train_path = r'../data/TrainSmall2/Train/'

class_names = ['adult_females', 'adult_males', 'juveniles', 'pups', 'subadult_males']

results_dir = r'../results/bbox_chips/'

filenames = [str(x)+'.jpg' for x in range(41,51)]


def get_blobs(dotted_image:str, clean_image:str):

    """ Given two images - one dotted and its clean match - return blobs """

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
    #print(h,w,d)

    return blobs



def create_coord_df(files):

    """ Create a dataframe to hold the coordinates of all marked seals in the training data"""
    global class_names
    df = pd.DataFrame(index=files, columns=class_names)


    return df


def retrieve_image_paths(file):
    dotted = train_dotted_path + file
    undotted = train_path + file
    return dotted, undotted

def count_classes(blobs, file, df):

    """ iterate through blobs and identify what class they belong to. Aggregate the coordinates in lists,
    and append to coordinates dataframe."""

    # retrieve dotted_image path
    dotted_image = cv2.imread(retrieve_image_paths(file)[0])

    # initialize lists - we will append a coordinate pair each time we find a dot of one of these classes of sea lion
    adult_males = []
    subadult_males = []
    pups = []
    juveniles = []
    adult_females = []

    # Loop through blobs - depending on type of sea lion type, mark them accordingly
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob

        # get the color of the pixel from Train Dotted in the center of the blob

        b, g, R = dotted_image[int(y)][int(x)][:]

        # print(b,g,R)
        x1 = int((x * r) // width)
        y1 = int((y * r) // width)

        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        # Adult males
        if R > 200 and b < 50 and g < 50:
            half_size = 60
            adult_males.append((int(x), int(y)))

        # Subadult males
        elif R > 200 and b > 200 and g < 50:  # MAGENTA
            half_size = 40
            subadult_males.append((int(x), int(y)))

        # Pups
        elif R < 100 and b < 100 and 150 < g < 200:  # GREEN
            half_size = 20
            pups.append((int(x), int(y)))

        # Juveniles
        elif R < 100 and 100 < b and g < 100:  # BLUE
            half_size = 30
            juveniles.append((int(x), int(y)))

        # Females
        elif R < 150 and b < 50 and g < 100:  # BROWN
            half_size = 40
            adult_females.append((int(x), int(y)))

    df["adult_males"][file] = adult_males
    df["subadult_males"][file] = subadult_males
    df["adult_females"][file] = adult_females
    df["juveniles"][file] = juveniles
    df["pups"][file] = pups  # Ideal chip sizes:

    return


def create_chip_dir():
    """ Check if image/chip result locations exist. If they don't create them. """
    import os.path
    ##TODO Make this also create the bbox_chips if it doesn't already exist. os.makedirs didn't behave as expected
    for sea_lion_type in class_names:
        if not os.path.exists(results_dir+sea_lion_type):
            print(f'Creating directory for {sea_lion_type}.')
            #os.makedirs(results_dir)
            os.mkdir(results_dir+sea_lion_type)
        else:
            print(f'Directory exists for {sea_lion_type}.')

    return


def create_chips(df):
    """ Create chip images around each sea lion for further segmentation and labeling"""
    import os.path
    for sea_lion_type in class_names:
        chip_num = 0
        for file in tqdm(filenames,total=len(filenames)):
            for pair in tqdm(df[sea_lion_type][file], total=len(df[sea_lion_type][file]),desc=f'{sea_lion_type}-{file}'):
                y, x = pair[0], pair[1]
                width = int(chip_sizes.get(sea_lion_type) * 0.5)
                height = int(chip_sizes.get(sea_lion_type) * 0.5)
                chip = cv2.imread(retrieve_image_paths(file)[1]).copy()
                chip = chip[x-width:x+width, y-height:y+height]
                chip_num += 1
                chip_name = f'{file.split(".")[0]}_{sea_lion_type}_{chip_num}.png'
                cv2.imwrite(os.path.join(results_dir,sea_lion_type,chip_name), chip)

    return















