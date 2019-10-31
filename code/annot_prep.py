import json
import os
import sys
from glob import glob
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from tqdm import tqdm
from pascal_voc_writer import Writer
import cv2
import matplotlib.pyplot as plt
from ast import literal_eval

sys.path.append(os.getcwd()+'/code/')

class Labeler(object):

    def __init__(self, config='config.json'):
        """ Initiate class with the data directories needed.
        Args:
        """
        with open(config) as config_file:
            conf = json.load(config_file)

        self._small_chip_unmarked_dir = conf['small_chip_unmarked_dir']  # location of small unmarked chips (output)
        self._small_chip_dotted_dir = conf['small_chip_dotted_dir']  # location of small dotted chips (output)
        self.class_names = conf['class_names']  # list of sea lion classes
        self._chip_size = conf['chip_size']  # h/w of chip sizes
        self.dataframe_path = conf['dataframe_path']  # location of the coordinates dataframe
        self.annot_path = conf['annotation_file_path']
        self.chip_sizes = conf['chip_sizes']

    def create_dir(self):
        if not os.path.exists(self.annot_path):
            os.mkdir(self.annot_path)
        return

    def create_xml(self):
        coordinates = pd.read_csv(self.dataframe_path, index_col=0)
        for index, row in tqdm(coordinates.iterrows(),total=coordinates.shape[0]):
            image = self._small_chip_unmarked_dir + index
            image_array = cv2.imread(image)
            annot_writer = Writer(image, image_array.shape[1], image_array.shape[0])
            for sl_type in self.class_names:
                for pair in literal_eval(row[sl_type]):
                    center_y, center_x = pair[0], pair[1]
                    x_min = int(center_x - self.chip_sizes[sl_type]/2)
                    x_max = int(center_x + self.chip_sizes[sl_type]/2)
                    y_min = int(center_y - self.chip_sizes[sl_type]/2)
                    y_max = int(center_y + self.chip_sizes[sl_type]/2)
                    annot_writer.addObject(sl_type, x_min, y_min, x_max, y_max)

            annot_writer.save(self.annot_path+index.split('.')[0]+'.xml')

labeler = Labeler()
labeler.create_dir()
labeler.create_xml()

## TODO : Fix the chip sizes to reflect a smaller input image size. Original chip size still reflects full-sized image