# Make predictions the mask-rcnn sea lion model

import json
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset

# class that defines and loads the sea lion dataset
from sea_lions_maskRCNN import SeaLionDataset

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

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "sea_lion_cfg"
	# number of classes (background + sea lion classes)
	NUM_CLASSES = 1 + 5
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=2):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	pyplot.show()

# load the train dataset
train_set = SeaLionDataset()
train_set.load_dataset('sea_lions', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = SeaLionDataset()
test_set.load_dataset('sea_lions', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir=model_dir, config=cfg)
# load model weights
model_path = model_dir+'mask_rcnn_sea_lion_cfg_0005.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)