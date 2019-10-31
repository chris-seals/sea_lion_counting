import json
import os
import sys
from glob import glob
import pandas as pd
import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm

sys.path.append(os.getcwd()+'/code/')


class Dataset(object):

    def __init__(self, target_size=(256, 256), img_color_mode='rgb', config='config.json'):
        """ Initiate class with the data directories needed.
        Args:
        """
        with open(config) as config_file:
            conf = json.load(config_file)

        self._dotted_data_dir = conf['dotted_data_dir']
        self._unmarked_data_dir = conf['unmarked_data_dir']
        self._small_chip_unmarked_dir = conf['small_chip_unmarked_dir']
        self._small_chip_dotted_dir = conf['small_chip_dotted_dir']
        self.class_names = conf['class_names']
        self._chip_size = conf['chip_size']
        self._target_size = target_size
        self._img_color_mode = img_color_mode
        self.r = conf['r']
        # Setting target image and mask sizes as (height, width, number of channels).
        self._img_channels = self.find_numb_channels(self._img_color_mode)
        self._target_img_size = (self._target_size[0], self._target_size[1], self._img_channels)

        self.filenames = glob(self._unmarked_data_dir + '*.jpg')
        self.filenames = [f.split('\\')[1] for f in self.filenames]
        self.dataframe_path = conf['dataframe_path']

    def find_numb_channels(self, color_mode):
        """ Calculate number of channels according to color mode.
        Args:
            color_mode (str): image color mode. One of "grayscale", "rgb".
        Returns (int): number of channels for image. "grayscale" = 1, "rgb" = 3.
        """
        if color_mode == 'rgb':
            numb_channels = 3
        elif color_mode == 'grayscale':
            numb_channels = 1
        else:
            raise KeyError("This is not a valid color mode. Use 'rgb' or 'grayscale'")
        return numb_channels


    def mini_chipper(self):
        """ Rip images into smaller chips.
        """
        img_count = 1
        #self.r = 0.4 # Scaling down

        for file in self.filenames:
            #print(file)
            image_1 = cv2.imread(self._dotted_data_dir + file)
            image_2 = cv2.imread(self._unmarked_data_dir + file)
            h, w, d = image_1.shape

            width = self._chip_size

            ma = cv2.cvtColor((1 * (np.sum(image_1, axis=2) > 20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
            img = cv2.resize(image_2 * ma, (int(w * self.r), int(h * self.r)))
            img_d = cv2.resize(image_1 * ma, (int(w * self.r), int(h * self.r)))

            h1, w1, d = img.shape

            for i in range(int(w1 // width)):
                for j in range(int(h1 // width)):
                    cv2.imwrite(os.path.join(self._small_chip_unmarked_dir, f'chip{img_count}.jpg'),
                                img[j * width:j * width + width, i * width:i * width + width, :])
                    cv2.imwrite(os.path.join(self._small_chip_dotted_dir, f'chip{img_count}.jpg'),
                                img_d[j * width:j * width + width, i * width:i * width + width, :])
                    img_count += 1
                    if img_count % 200 == 0:
                        print(f'Processing chip no.', img_count)

    def reset_small_chips(self):
        """ Clears out the small chip directories in case a re-run is required."""
        confirm = input('Are you sure you want to delete small chips? (y/n)')
        if confirm == 'y':
            import os
            dirs = [self._small_chip_dotted_dir,
                    self._small_chip_unmarked_dir]
            for directory in dirs:
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)

                    except Exception as e:
                        print(e)
            print("Small chips deleted, directories clean.")
        elif confirm == 'n':
            return
        else:
            print("invalid input, please hit 'y' or 'n'")
            confirm = input('Are you sure you want to delete small chips? (y/n)')

    def create_df(self):

        """ Create a dataframe to hold the coordinates of all marked seals in the training data.
        Must be run AFTER mini-chipper... assertion check should catch this"""
        chips = glob(self._small_chip_dotted_dir + '*.jpg')
        assert len(chips) > 0
        self.chip_index = [f.split('\\')[1] for f in chips]
        df = pd.DataFrame(index=self.chip_index, columns=self.class_names)
        df.to_csv(self.dataframe_path)

        return  #self.dataframe, self.chip_index

    def retrieve_coords(self):
        #chip_index = self.chip_index
        self.dataframe = pd.read_csv(self.dataframe_path, dtype=object, index_col=0)
        print(self.dataframe.head())
        width = self._chip_size
        #(width)
        r = self.r
        chips = glob(self._small_chip_dotted_dir + '*.jpg')
        self.chip_index = [f.split('\\')[1] for f in chips]

        for file in tqdm(self.chip_index,total=len(self.chip_index)):
            dotted_image = self._small_chip_dotted_dir + file
            clean_image = self._small_chip_unmarked_dir + file
            #print(dotted_image)
            #print(clean_image)
            assert dotted_image[-8:] == clean_image[-8:]
            """ Given two images - one dotted and its clean match - return blobs """
            # Read in images, grab dot locations
            dotted_image = cv2.imread(dotted_image)
            clean_image = cv2.imread(clean_image)
            #print(dotted_image.shape)
            #print(clean_image.shape)
            # absolute difference between Train and Train Dotted -- use to isolate dot locations
            dot_diff_image = cv2.absdiff(dotted_image, clean_image)
            mask_1 = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)
            mask_1[mask_1 < 50] = 0
            mask_1[mask_1 > 0] = 255
            dots_only_image = cv2.bitwise_or(dot_diff_image, dot_diff_image, mask=mask_1)

            mask_1 = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)

            # convert to grayscale to be accepted by skimage.feature.blob_log
            dots_only_max_image = np.max(dots_only_image, axis=2)

            # detect blobs
            blobs = feature.blob_log(dots_only_max_image, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

            """ iterate through blobs and identify what class they belong to. Aggregate the coordinates in lists,
                and append to coordinates dataframe."""

            # initialize lists - we will append a coordinate pair each time we find a dot of one of these classes of sea lion
            adult_males = []
            subadult_males = []
            pups = []
            juveniles = []
            adult_females = []

            # Loop through blobs - depending on type of sea lion type, mark them accordingly
            for blob in blobs:
                try:
                    # get the coordinates for each blob
                    y, x, s = blob

                    # get the color of the pixel from Train Dotted in the center of the blob
                    #print(y, x, s, type(dotted_image), dotted_image)
                    b, g, R = dotted_image[int(y)][int(x)][:]

                    # decision tree to pick the class of the blob by looking at the color in Train Dotted
                    # Adult males
                    if R > 200 and b < 50 and g < 50:
                        half_size = 60
                        adult_males.append((int(x), int(y)))
                        #print('am found')
                        #print(int(x),int(y),dotted_image)
                    # Subadult males
                    if R > 200 and b > 200 and g < 50:  # MAGENTA
                        half_size = 40
                        subadult_males.append((int(x), int(y)))
                        #print('sam found')
                        #print(int(x), int(y), dotted_image)
                    # Pups
                    if R < 100 and b < 100 and 150 < g < 200:  # GREEN
                        half_size = 20
                        pups.append((int(x), int(y)))
                        #print('pups found')
                        #print(int(x), int(y), dotted_image)
                    # Juveniles
                    if R < 100 and 100 < b and g < 100:  # BLUE
                        half_size = 30
                        juveniles.append((int(x), int(y)))
                        #print('juvie found')
                        #print(int(x), int(y), dotted_image)
                    # Females
                    if R < 150 and b < 50 and g < 100:  # BROWN
                        half_size = 40
                        adult_females.append((int(x), int(y)))
                        #print('female found')
                        #print(int(x), int(y), file)
                except Exception as e:
                    print(e)
                    continue
            self.dataframe["adult_males"][file] = adult_males
            self.dataframe["subadult_males"][file] = subadult_males
            self.dataframe["adult_females"][file] = adult_females
            self.dataframe["juveniles"][file] = juveniles
            self.dataframe["pups"][file] = pups  # Ideal chip sizes:

        self.dataframe.to_csv(self.dataframe_path)
        return
## TODO:
#### Run lob comparisons
#### Record coordinates
####    -- Save coordinates off as one dictionary per image - of centerpoints?
#### Create .xml file for each image with pascal VOC writer module
#### final desired inputs = directory of images and one .xml file for each image

# orig_filenames = [file[-6:] for file in glob(train_path+'/*.jpg')]
#
# for filename in orig_filenames:
#     image_1 = cv2.imread("../../data/TrainSmall2/TrainDotted/" + filename)
#     image_2 = cv2.imread("../../data/TrainSmall2/Train/" + filename)

images = Dataset()
images.reset_small_chips()
images.mini_chipper()
images.create_df()
images.retrieve_coords()
