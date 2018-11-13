
import os
import pickle
import numpy as np


# noinspection PyRedeclaration
class BinaryLogisticRegression:

	def __init__(self, file_path):
		"""
		This constructor loads a pre-trained neural network from the provided save file.

		:param file_path: Path to the file
		"""
		# Reading saved parameters from given file
		with open(file_path, 'r') as f:
			params = pickle.load(f)

		# Storing parameters read from file into class member variables
		self.x_n    = params['x_n']
		self.W      = params['W']
		self.b      = params['b']
		self.alpha  = params['alpha']

	def __init__(self, X_train, alpha):
		"""
		This constructor assigns the hyper parameters based on the training data and other arguments.

		:param X_train: Input training vector of size (x_n, m)
		:param alpha: Learning rate of this classifier
		"""
		# Getting relevant dimension sizes
		self.x_n = X_train.shape[0]

		# Initialising parameters
		self.W = np.zeros(self.x_n, 1)
		self.b = 0

		# Setting hyper-parameters
		self.alpha = alpha

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
		dW = np.mean(X * dZ.T, axis=1, keepdims=True)  # Should have shape (x_n, 1)
		db = np.mean(dZ, axis=1, keepdims=True)  # Should have shape (1, 1)

		return { 'dW': dW, 'db': db }

	def train(self, X_train, Y_train, iterations=1000):
		"""
		This method trains this model by running `iteration` number of forward and backward propagation.
		The model must be trained before trying to use it to make predictions.

		:param X_train: Input training vector of size (x_n, m)
		:param Y_train: Output training vector of size (1, m)
		:param iterations: The number of iterations we want it to run for
		:return: None
		"""
		# Iterating `iterations` number of times
		for i in range(iterations):
			# Run forward prop to get current model output
			A = self._forward_prop(X=X_train)

			# Compute gradient descent values using back prop
			grad_descent = self._backward_prop(X_train, A, Y_train)

			# Update current parameters
			self.W -= self.alpha * grad_descent['dW']
			self.b -= self.alpha * grad_descent['db']

	def predict(self, X, Y=None):
		"""
		This method performs predictions using its current parameters on X and return the appropriate results.
		If Y is provided, it will instead return the accuracy of the predictions w.r.t Y.

		:param X: Input vector of size (x_n, m) where m can be 1
		:param Y: Output vector of size (x_n, m) where m can be 1 (optional)
		:return: Prediction vector on X if y is not provided, else accuracy vector
		"""
		# Compute y_hat (A)
		if Y is None:
			return np.where(self._forward_prop(X) > 0.5, 1, 0)
		# Compute cost
		else:
			return np.abs(self._forward_prop(X) - Y) / Y * 100

	def save_params(self, file_name, dir_path='F:\\Neural_Networks\\'):
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

		with open(os.path.join(dir_path, file_name + '.pck')) as f:
			pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
