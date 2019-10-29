import json
import os
import sys
from glob import glob

import cv2
import numpy as np

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


        self._target_size = target_size
        self._img_color_mode = img_color_mode

        # Setting target image and mask sizes as (height, width, number of channels).
        self._img_channels = self.find_numb_channels(self._img_color_mode)
        self._target_img_size = (self._target_size[0], self._target_size[1], self._img_channels)

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

    def mini_chipper(self, chip_size=(100,100)):
        """ Rip images into smaller chips.
                Args:
                    chip_size (tuple) : dimensions of chips to use

        """
        self._chip_size = chip_size
        img_count = 1
        r = 0.4 # Scaling down

        filenames = glob(self._unmarked_data_dir+'*.jpg')
        filenames = [f.split('\\')[1] for f in filenames]
        print(filenames)
        for file in filenames:
            #print(file)
            image_1 = cv2.imread(self._dotted_data_dir + file)
            image_2 = cv2.imread(self._unmarked_data_dir + file)
            h, w, d = image_1.shape

            width = self._chip_size[0]

            ma = cv2.cvtColor((1 * (np.sum(image_1, axis=2) > 20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
            img = cv2.resize(image_2 * ma, (int(w * r), int(h * r)))
            img_d = cv2.resize(image_1 * ma, (int(w * r), int(h * r)))

            h1, w1, d = img.shape

            for i in range(int(w1 // width)):
                for j in range(int(h1 // width)):
                    cv2.imwrite(os.path.join(self._small_chip_unmarked_dir, f'chip{img_count}.jpg'),
                                img[j * width:j * width + width, i * width:i * width + width, :])
                    cv2.imwrite(os.path.join(self._small_chip_dotted_dir, f'sm_chip{img_count}.jpg'),
                                img_d[j * width:j * width + width, i * width:i * width + width, :])
                    img_count += 1
                    if img_count % 200 == 0:
                        print(f'Processing chip no.', img_count)



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
images.mini_chipper()
