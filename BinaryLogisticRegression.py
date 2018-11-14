
import os
import time
import pickle

import numpy as np

from commons import time_to_str

# noinspection PyRedeclaration
class BinaryLogisticRegression:

	def __init__(self, x_n, alpha):
		"""
		This constructor assigns the hyper parameters based on the training data and other arguments.

		:param x_n: Input size of a single input vector
		:param alpha: Learning rate of this classifier
		"""
		# Getting relevant dimension sizes
		self.x_n = x_n

		# Initialising parameters
		self.W = np.zeros((self.x_n, 1))
		self.b = 0

		# Setting hyper-parameters
		self.alpha = alpha

	@staticmethod
	def load_BLR(file_path):
		"""
		This method loads a pre-trained neural network from the provided save file.

		:param file_path: Path to the file
		"""
		# Reading saved parameters from given file
		with open(file_path, 'rb') as f:
			params = pickle.load(f)

		# Storing parameters read from file into new object member variables
		this = BinaryLogisticRegression(params['x_n'], params['alpha'])
		this.W = params['W']
		this.b = params['b']

		return this

	def _forward_prop(self, X):
		"""
		This method performs forward propagation using the current parameters of the model using the giver input vector(s)
		and returns the corresponding output value(s).

		:param X: Single or multiple input vector(s) of shape (x_n, m) where m can be 1
		:return: A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		"""
		# Linear function computation
		Z = np.dot(self.W.T, X)

		# Activation function computation
		A = 1 / (1 + np.exp(-Z))

		return A

	@staticmethod
	def loss(A, Y):
		"""
		This method calculates the loss after forward propagation.
		If A, Y are vectors shaped (1, m) then this function returns the cost instead of the loss.

		:param A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:param Y: Expected output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:return: Loss if A.shape == (1, 1) else Cost
		"""
		return -(Y * np.log(A) + (1 - Y) * np.log(1 - A))

	def _backward_prop(self, X, A, Y):
		"""
		This method performs backward propagation given the current output vector A.

		:param X: Input training vector of size (x_n, m)
		:param Y: Output training vector of size (1, m)
		:param A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:return: Dictionary with gradient descent results stored in it with keys - { dW, db }
		"""
		# Calculating gradient descent
		dZ = A - Y  # Should have shape (1, m)
		dW = np.mean(np.dot(X, dZ.T), axis=1, keepdims=True)  # Should have shape (x_n, 1)
		db = np.mean(dZ, axis=1, keepdims=True)  # Should have shape (1, 1)

		return { 'dW': dW, 'db': db }

	def train(self, X_train, Y_train, iterations=1000, print_logs=False):
		"""
		This method trains this model by running `iteration` number of forward and backward propagation.
		The model must be trained before trying to use it to make predictions.

		:param X_train: Input training vector of size (x_n, m)
		:param Y_train: Output training vector of size (1, m)
		:param iterations: The number of iterations we want it to run for
		:param print_logs: boolean to select whether or not to print log in stdout
		:return: None
		"""
		# Initialising logging variables
		is_first_pass = True        # Flag to determine whether or not this is the first pass
		fprop_times = []
		bprop_times = []
		pass_times  = []

		if print_logs:
			print("Input vector size (x_n) : {}".format(self.x_n))
			print("# of training sets  (m) : {}".format(Y_train.shape[1]))
			print()

		# Iterating `iterations` number of times
		for i in range(iterations):
			# Run forward prop to get current model output
			tic = time.time()
			A = self._forward_prop(X=X_train)
			toc = time.time()
			fprop_time = toc - tic

			# Compute gradient descent values using back prop
			tic = time.time()
			grad_descent = self._backward_prop(X_train, A, Y_train)
			toc = time.time()
			bprop_time = toc - tic

			# Update current parameters
			tic = time.time()
			self.W -= self.alpha * grad_descent['dW']
			self.b -= self.alpha * grad_descent['db']
			toc = time.time()
			pass_time = fprop_time + bprop_time + (toc - tic)

			# Logging time taken by first pass
			if is_first_pass:
				if print_logs:
					print("Pass #1: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
						*list(map(time_to_str, [fprop_time, bprop_time, pass_time]))))
					print()
				is_first_pass = False       # Removing flag

			# Adding times to their respective lists
			fprop_times.append(fprop_time)
			bprop_times.append(bprop_time)
			pass_times.append(pass_time)

		# Logging total training time taken
		if print_logs:
			mean = lambda x: sum(x) / len(x)
			# Print total times
			print("Training Total: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
				*list(map(time_to_str, map(sum, [fprop_times, bprop_times, pass_times])))))
			# Print average times
			print("Training Average: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
				*list(map(time_to_str, map(mean, [fprop_times, bprop_times, pass_times])))))
			# Print maximum times
			print("Training Max: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
				*list(map(time_to_str, map(max, [fprop_times, bprop_times, pass_times])))))
			print()

	def predict(self, X, Y=None):
		"""
		This method performs predictions using its current parameters on X and return the appropriate results.
		If Y is provided, it will instead return the accuracy of the predictions w.r.t Y.

		:param X: Input vector of size (x_n, m) where m can be 1
		:param Y: Output vector of size (x_n, m) where m can be 1 (optional)
		:return: Prediction vector on X if y is not provided, else accuracy vector
		"""
		# Compute Y_hat (A)
		if Y is None:
			return np.where(self._forward_prop(X) > 0.5, 1, 0)
		# Compute L1 cost
		else:
			return np.abs(self._forward_prop(X) - Y) / Y * 100

	def save_BLR(self, file_name, dir_path='F:\\Neural_Networks\\'):
		"""
		This method saves the current neural network's parameters into the provided file.

		:param file_name: Name of the file without any extensions
		:param dir_path: Path to the target directory
		:return: None
		"""
		params = {
			'x_n':      self.x_n,
			'W':        self.W,
			'b':        self.b,
			'alpha':    self.alpha
		}

		with open(os.path.join(dir_path, file_name + '.pck'), 'wb+') as f:
			pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
