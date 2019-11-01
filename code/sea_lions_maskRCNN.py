# example of extracting bounding boxes from an annotation file
import json
import os
from xml.etree import ElementTree
from matplotlib import pyplot
import numpy as np
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=tfconfig))
from keras.callbacks import TensorBoard
from time import time



with open('config.json') as config_file:
	conf = json.load(config_file)
_small_chip_unmarked_dir = conf['small_chip_unmarked_dir']  # location of small unmarked chips (output)
# self._small_chip_dotted_dir = conf['small_chip_dotted_dir']  # location of small dotted chips (output)
class_names = conf['class_names']  # list of sea lion classes
# self._chip_size = conf['chip_size']  # h/w of chip sizes
# self.dataframe_path = conf['dataframe_path']  # location of the coordinates dataframe
annot_path = conf['annotation_file_path']
model_dir = conf['model_dir']
#tensorboard = TensorBoard(log_dir=model_dir+'/logs/{}'.format(time()))
tensorboard = TensorBoard(log_dir=model_dir+"logs/{}".format(time()))
# self.chip_sizes = conf['chip_sizes']
class SeaLionDataset(Dataset):

	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class('sea_lions', 1, "adult_females")
		self.add_class('sea_lions', 2, "adult_males")
		self.add_class('sea_lions', 3, "juveniles")
		self.add_class('sea_lions', 4, "pups")
		self.add_class('sea_lions', 5, "subadult_males")
		# define data locations
		images_dir = _small_chip_unmarked_dir # + '/images/'
		annotations_dir = annot_path
		# find all images
		for filename in os.listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip all images after 200 if we are building the training set
			if is_train and int(image_id) >= 200:
				continue
			# skip all images before 200 if we are building the test/val set
			if not is_train and int(image_id) < 200:
				continue
			# skip bad images
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('sea_lions', image_id=image_id, path=img_path, annotation=ann_path)


	# # function to extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//object'):
			xmin = int(box.find('bndbox/xmin').text)
			ymin = int(box.find('bndbox/ymin').text)
			xmax = int(box.find('bndbox/xmax').text)
			ymax = int(box.find('bndbox/ymax').text)
			name = box.find('name').text
			coors = [xmin, ymin, xmax, ymax, name]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height


	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		#print('path=',path)
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		#masks = np.zeros([h, w], dtype='uint8')
		masks = np.zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			#print('box=',box)
			row_s, row_e = box[0], box[2]
			col_s, col_e = box[1], box[3]
			masks[row_s:row_e, col_s:col_e,i] = 1  # i
			#print('self.class_names=',self.class_names)
			class_ids.append(self.class_names.index(box[4]))
			#print(len(masks))
		return masks, np.asarray(class_ids, dtype='int32')


	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class SeaLionConfig(Config):
	# define the name of the configuration
	NAME = "sea_lion_cfg"
	# number of classes (background + sea lion types)
	NUM_CLASSES = 1 + 5
	# number of training steps per epoch
	STEPS_PER_EPOCH = 200

	GPU_COUNT = 1

	IMAGES_PER_GPU = 1

if __name__ == "__main__":
	# prepare train set
	train_set = SeaLionDataset()
	train_set.load_dataset('sea_lions', is_train=True)
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))
	# prepare test/val set
	test_set = SeaLionDataset()
	test_set.load_dataset('sea_lions', is_train=False)
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))
	# prepare config
	config = SeaLionConfig()
	config.display()
	# define the model
	model = MaskRCNN(mode='training', model_dir=model_dir, config=config)
	# load weights (mscoco) and exclude the output layers
	model.load_weights(model_dir+'mask_rcnn_coco.h5', by_name=True,
	                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
	# train weights (output layers or 'heads')
	model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


	##### DOWNLOAD WEIGHTS FILE HERE: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 #####

