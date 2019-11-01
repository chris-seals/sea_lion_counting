# example of extracting bounding boxes from an annotation file
import json
import os
from xml.etree import ElementTree
from matplotlib import pyplot
import numpy as np
from mrcnn.utils import Dataset

with open('config.json') as config_file:
	conf = json.load(config_file)
_small_chip_unmarked_dir = conf['small_chip_unmarked_dir']  # location of small unmarked chips (output)
# self._small_chip_dotted_dir = conf['small_chip_dotted_dir']  # location of small dotted chips (output)
class_names = conf['class_names']  # list of sea lion classes
# self._chip_size = conf['chip_size']  # h/w of chip sizes
# self.dataframe_path = conf['dataframe_path']  # location of the coordinates dataframe
annot_path = conf['annotation_file_path']


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
			# skip all images after 270 if we are building the training set
			if is_train and int(image_id) >= 270:
				continue
			# skip all images before 270 if we are building the test/val set
			if not is_train and int(image_id) < 270:
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
		print('path=',path)
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = np.zeros([h, w], dtype='uint8')
		#masks = np.zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			print('box=',box)
			row_s, row_e = box[0], box[2]
			col_s, col_e = box[1], box[3]
			masks[row_s:row_e, col_s:col_e] = 1  # i
			print('self.class_names=',self.class_names)
			class_ids.append(self.class_names.index(box[4]))
			print(len(masks))
		return masks, np.asarray(class_ids, dtype='int32')

## TODO: Add this as last portion of load_mask to account for multiple classes
		# Return mask, and array of class IDs of each instance.
		# return mask, info['class_ids']


	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
# train_set = SeaLionDataset()
# train_set.load_dataset('sea_lions', is_train=True)
# train_set.prepare()
# print('Train: %d' % len(train_set.image_ids))
#
# # test/val set
# test_set = SeaLionDataset()
# test_set.load_dataset('sea_lions', is_train=False)
# test_set.prepare()
# print('Test: %d' % len(test_set.image_ids))
# extract details form annotation file
# boxes, w, h = extract_boxes('../results/smaller_bbox_chips/annot/chip19.xml')
# summarize extracted details
# print(boxes, w, h)


# train set
train_set = SeaLionDataset()
train_set.load_dataset('sea_lions', is_train=True)
train_set.prepare()
# load an image
image_id = 135
image = train_set.load_image(image_id)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)
# plot image
pyplot.imshow(image)
# plot mask
pyplot.imshow(mask[:, :], cmap='gray', alpha=0.5)
pyplot.show()
