import glob
import os

import cv2
import numpy as np
from keras.preprocessing import image


class Dataset(object):
    
    def __init__(self, dotted_data_dir, unmarked_data_dir, target_size=(256, 256), img_color_mode = 'rgb'):
        """ Initiate class with the data directories needed.
        Args:
            dotted_data_dir (str): path to directory for labeled data
            unmarked_data_dir (str): path to directory for unlabeled data
        """
        
        self._dotted_data_dir = dotted_data_dir
        self._data_dir = unmarked_data_dir
        
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
        

    def read_images(self):
        """ Reads the images for the process of looking at images and to process
            new images by capturing individual parts of pictures
        Returns (array): Labeled data and unlabeld images
        """
        dots = [cv2.imread(file) for file in glob.glob(self._dotted_data_dir + "*.jpg")]
        images = [cv2.imread(file) for file in glob.glob(self._data_dir + "*.jpg")]
        return dots, images
    
    
    def load_image(self, filepath, target_size, color_mode):
        """ Will be used for training data in the future
        """
        img = image.load_img(filepath, target_size=target_size, color_mode=color_mode)
        img_data = image.img_to_array(img)
        return img_data
        
    def load_training_data(self):
        """ Handles loading the training images, resizing it to the target size and returning them as a
        list of Numpy arrays.
        Returns (tuple):
            ndarray: array of training images, where each image is a numpy array.
        """
        # Getting all image paths.
        dot_names = os.listdir(self._dotted_data_dir)
        dot_paths = [os.path.join(self._dotted_data_dir, dot_name) for dot_name in dot_names]
        img_names = os.listdir(self._data_dir)
        img_paths = [os.path.join(self._data_dir, img_name) for img_name in img_names]

        dot_arr = []
        img_arr = []

        for i in range(len(dot_names)):
            # load dotted images
            try:
                os.path.exists(dot_paths[i])
                dot = image.load_img(dot_paths[i], target_size=self._target_img_size, color_mode=self._img_color_mode)
                dot_data = image.img_to_array(dot)
                dot_arr.append(dot_data)
                
                # load unmarked images
                os.path.exists(img_paths[i])
                img = image.load_img(img_paths[i], target_size=self._target_img_size, color_mode=self._img_color_mode)
                img_data = image.img_to_array(img)
                img_arr.append(img_data)


            
            except OSError:
                pass
        
        dot_arr = np.array(dot_arr)
        img_arr = np.array(img_arr)
        
        return dot_arr, img_arr
    
    # def get_chip(self, c, img, size):
    #
    #     ### TODO ###
    #     # Document better, rename variables
    #     try:
    #         y, x = int(c[0]), int(c[1])
    #
    #         #width, height of bbox
    #         width = int(size * 0.5)
    #         height = int(size * 0.5)
    #
    #         #retrieve chip of blob
    #         chip = img[x - width:x + width, y - height:y + height]
    #
    #
    #     except Exception as e:
    #         print(e)
    #
    #     return chip
    #
    # def eval_chip_size(self, lion_type, chip_size, image):
    #     ### TODO ###
    #     # Document better
    #
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #
    #     fig = plt.figure(figsize=(12, 12))
    #     num_chips = len(coordinates_df[lion_type][0])
    #     try:
    #         for i in range(len(coordinates_df[lion_type][0])):
    #             ax = fig.add_subplot(num_chips / 10 + 1, 10, i + 1)
    #             plt.axis('off')
    #             plt.imshow(get_chip(coordinates_df[lion_type][0][i], image, chip_size))
    #     except ValueError:
    #         pass
