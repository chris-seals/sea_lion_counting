# Evaluate the mask-rcnn sea lion model
#### Results in runtime warning errors... all evals come back as nans likely due to some zero division.
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

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# print('AP=',AP,
		#       'gt_bbox=',gt_bbox,
		#       'gt_class_id=', gt_class_id,
		#       'gt_mask=', gt_mask,
		#       'r[rois]=', r["rois"],
		#       'r[class_ids]=', r["class_ids"],
		#       'r["scores"]=', r["scores"],
		#       #"r['masks']",  r['masks'], '\n',
		#       'r=',r)
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

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
model.load_weights(model_dir+'mask_rcnn_sea_lion_cfg_0005.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"])
#                  exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)