
import os
import glob
import warnings

import numpy as np

from PIL import Image
from functools import reduce

from BinaryLogisticRegression import BinaryLogisticRegression


def create_eagle_falcon_bin_classifier():
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
		with warnings.catch_warnings():
			warnings.filterwarnings(action="ignore", message="Palette images.*")
			im = Image.open(imfile).convert('RGB')

		# Converting image to flattened numpy array
		im_arr = np.array(im)
		im_arr = im_arr.reshape((reduce(lambda x, y: x * y, im_arr.shape), 1))

		# Adding expected output at the bottom row
		im_arr = np.concatenate((im_arr, np.ones((1, 1), dtype=np.uint8)))

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
		with warnings.catch_warnings():
			warnings.filterwarnings(action="ignore", message="Palette images.*")
			im = Image.open(imfile).convert('RGB')

		# Converting image to flattened numpy array
		im_arr = np.array(im)
		im_arr = im_arr.reshape((reduce(lambda x, y: x * y, im_arr.shape), 1))

		# Adding expected output at the bottom row
		im_arr = np.concatenate((im_arr, np.zeros((1, 1), dtype=np.uint8)))

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

	# Train the new BLR
	classifier.train(X_train, Y_train, iterations=5000, print_logs=True)

	# Save the new BLR
	classifier.save_BLR('eagle-v-falcon')


def test_eagle_falcon_bin_classifier():
	# Load BLR from save state
	classifier = BinaryLogisticRegression.load_BLR('F:\\Neural_Networks\\eagle-v-falcon.pck')

	# Setting up test and train set directories
	test_set_dir = "F:\\Testing_Sets"

	# test_file = os.path.join(train_set_dir, 'Eagle_128', 'IMG_00002.png')

	## Creating input training set
	# Initialising variables
	V_test = None
	m = 0

	# Iterating over all training eagles
	for imfile in glob.glob(os.path.join(test_set_dir, 'Eagle_128', '*.png')):
		# Opening the image file
		with warnings.catch_warnings():
			warnings.filterwarnings(action="ignore", message="Palette images.*")
			im = Image.open(imfile).convert('RGB')

		# Converting image to flattened numpy array
		im_arr = np.array(im)
		im_arr = im_arr.reshape((reduce(lambda x, y: x * y, im_arr.shape), 1))

		# Adding expected output at the bottom row
		im_arr = np.concatenate((im_arr, np.ones((1, 1), dtype=np.uint8)))

		# Adding new training vector to V_train
		if V_test is None:
			V_test = im_arr
		else:
			V_test = np.concatenate((V_test, im_arr), axis=1)

		# Update number of train cases for eagles
		m += 1

	# Iterating over all training falcons
	for imfile in glob.glob(os.path.join(test_set_dir, 'Falcon_128', '*.png')):
		# Opening the image file
		with warnings.catch_warnings():
			warnings.filterwarnings(action="ignore", message="Palette images.*")
			im = Image.open(imfile).convert('RGB')

		# Converting image to flattened numpy array
		im_arr = np.array(im)
		im_arr = im_arr.reshape((reduce(lambda x, y: x * y, im_arr.shape), 1))

		# Adding expected output at the bottom row
		im_arr = np.concatenate((im_arr, np.zeros((1, 1), dtype=np.uint8)))

		# Adding new training vector to V_train
		V_test = np.concatenate((V_test, im_arr), axis=1)

		# Update number of train cases for eagles
		m += 1

	# Creating training set variables
	X_test = V_test[:-1, :]
	Y_test = V_test[-1, :].reshape((1, m))
	x_n = X_test.shape[0]

	# Make sure that the correct classifier has been loaded in
	if x_n != classifier.x_n:
		raise ValueError("Wrong classifier has ben loaded")

	# Get predictions from the classifier
	Y_predict = classifier.predict(X_test)

	n_correct   = np.sum(Y_predict == Y_test)
	n_incorrect = np.sum(Y_predict != Y_test)

	print("{0:d} correct, {1:d} incorrect out of a total of {2:d} cases".format(n_correct, n_incorrect, m))
	print("Accuracy: {0:.2f}%".format(n_correct / m * 100))


if __name__ == '__main__':
	# create_eagle_falcon_bin_classifier()
	test_eagle_falcon_bin_classifier()
