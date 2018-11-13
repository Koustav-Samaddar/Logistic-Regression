
import os
import glob

import numpy as np

from PIL import Image
from functools import reduce

from BinaryLogisticRegression import BinaryLogisticRegression

def eagle_falcon_bin_classifier():
	"""
	In this function we will train a binary classifier using Logistic Regression to differentiate between a Falcon and an Eagle

	Eagle -> 1
	Falcon -> 0

	:return: None
	"""
	# Setting up test and train set directories
	train_set_dir = "F:\\Training_Sets"
	test_set_dir = "F:\\Testing_Sets"

	# test_file = os.path.join(train_set_dir, 'Eagle_128', 'IMG_00002.png')

	## Creating input training set
	# Initialising variables
	V_train = None
	m = 0

	# Iterating over all training eagles
	for imfile in glob.glob(os.path.join(train_set_dir, 'Eagle_128', '*.png')):
		# Opening the image file
		im = Image.open(imfile)

		# Converting image to flattened numpy array
		im_arr = np.array(im)
		im_arr = im_arr.reshape((reduce(lambda x, y: x * y, im_arr.shape), 1))

		# Adding expected output at the bottom row
		im_arr = np.concatenate((im_arr, np.array([[1]], dtype=np.uint8)))

		# Adding new training vector to V_train
		if V_train is None:
			V_train = im_arr
		else:
			V_train = np.concatenate((V_train, im_arr), axis=1)

		# Update number of train cases for eagles
		m += 1

	# Iterating over all training falcons
	for imfile in glob.glob(os.path.join(train_set_dir, 'Falcon_128', '*.png')):
		# Opening the image file
		im = Image.open(imfile)

		# Converting image to flattened numpy array
		im_arr = np.array(im)
		im_arr = im_arr.reshape((reduce(lambda x, y: x * y, im_arr.shape), 1))

		# Adding expected output at the bottom row
		im_arr = np.concatenate((im_arr, np.array([[0]], dtype=np.uint8)))

		# Adding new training vector to V_train
		V_train = np.concatenate((V_train, im_arr), axis=1)

		# Update number of train cases for eagles
		m += 1

	# Creating training set variables
	X_train = V_train[:-1, :]
	Y_train = V_train[-1, :].reshape((1, m))
	x_n = X_train.shape[0]

	# Creating a new BLR
	classifier = BinaryLogisticRegression(x_n, 0.05)

if __name__ == '__main__':
	print("Hello Shallow Learning!")
	eagle_falcon_bin_classifier()