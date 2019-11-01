import json
from glob import glob
with open('config.json') as config_file:
	conf = json.load(config_file)

_dotted_data_dir = conf['dotted_data_dir']  # location of original dotted images (input)
_unmarked_data_dir = conf['unmarked_data_dir']  # location of original clean images (input)
_small_chip_unmarked_dir = conf['small_chip_unmarked_dir']  # location of small unmarked chips (output)
_small_chip_dotted_dir = conf['small_chip_dotted_dir']  # location of small dotted chips (output)
class_names = conf['class_names']  # list of sea lion classes
_chip_size = conf['chip_size']  # h/w of chip sizes
r = conf['r']  # % scaling down of the chips
dataframe_path = conf['dataframe_path']  # location of the coordinates dataframe
# Setting target image and mask sizes as (height, width, number of channels).
#_img_color_mode = img_color_mode
filenames = glob(_unmarked_data_dir + '*.jpg')
filenames = [f.split('\\')[1] for f in filenames]