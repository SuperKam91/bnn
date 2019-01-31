#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import keras_models as kms
import input_tools
import output_tools

class bnn_predictor():
	"""
	steals parts of keras_model (in keras_forward.py) to make predictions (nn output)
	y from inputs x. uses nn parameter sets from chains .txt file, along with their relative weights
	"""
	def __init__(self, k_model, x, y, chains_file):
		"""
		assign model to class, calculate shape of weights, and arrays containing them (latter possibly redundant)
		"""
		self.weight_shapes = []
		self.model = k_model
		self.x = x
		self.y = y #for evaluation of performance
		self.get_weight_info()
		self.posterior_weights, self.LLs, self.nn_params, self.weight_norm = input_tools.get_chains_data(chains_file)
		self.nn_param_sets = [self.nn_params[i,:] for i in range(len(self.posterior_weights))]
		self.LL_var = 1. #for calculating LL on test data
		self.num_outputs = y.shape[1]

	def sample_prediction(self):
		"""
		sample a set of nn parameters according to their posterior weights, and make
		prediction from x using nn with these parameters.
		"""
		nn_param_set = np.random.choice(self.nn_param_sets, p = self.posterior_weights)
		self.set_k_weights(nn_param_set)
		return self.model.predict(self.x)

	def predictions_expectation(self):
		"""
		for each set of nn parameters, compute prediction from x,
		then take expectation over these predictions (w.r.t posterior) to get single prediction
		"""
		prediction = 0.
		for i in range(len(self.posterior_weights)):
			self.set_k_weights(self.nn_param_sets[i])
			prediction += self.posterior_weights[i] * self.model.predict(self.x)
		return prediction

	def expectation_prediction(self):
		"""
		calculate expected value of weights, then use this 
		for a nn and make a prediction with it.
		"""
		expected_param_set = (self.nn_params.T * self.posterior_weights).T.sum(axis = 0)
		self.set_k_weights(expected_param_set)
		return self.model.predict(self.x)

	def MPE_prediction(self):
		"""
		makes prediction with set of nn parameters corresponding to highest posterior weight.
		n.b. leaves keras model with parameters corresponding to MPE.
		"""
		argmax = np.argmax(self.posterior_weights)
		MPE_nn_param_set = self.nn_param_sets[argmax]
		self.set_k_weights(MPE_nn_param_set)
		return self.model.predict(self.x)		

	def MLE_prediction(self):
		"""
		makes prediction with set of nn parameters corresponding to highest likelihood.
		n.b. leaves keras model with parameters corresponding to MLE.
		"""
		argmax = np.argmax(self.LLs)
		MLE_nn_param_set = self.nn_param_sets[argmax]
		self.set_k_weights(MLE_nn_param_set)
		return self.model.predict(self.x)		

	def y_pred_chains(self, file = None, test_index = 0):
		"""
		essentially calculates p(f(theta, x)| X, Y) = p(y_pred(theta) | x, X, y) where theta are the nn parameters,
		f(.) is the function mapping from input x to output y (i.e. the nn and f(theta, x) = y_pred(theta)), x is the input
		data, y is true the output, y_pred is the predicted output, X and Y are the x and y training data respectively.
		note posterior weight and LL values here are from weight posterior.
		assumes self.x has shape (m,n) or (m,), not (n,).
		n.b. in chains LL is stored as -2 * LL for some reason
		n.b. in this chains file, weights are unnormalised again, as in original chains files (max has value 1)
		hence to use these weights for predictions, one should normalise them again
		"""
		try:
			x = self.x[test_index, :].reshape(1,-1)
		except KeyError:
			x = self.x[test_index].reshape(1,-1)	
		#get predictions of y from nn using each set of weights in posterior
		pred = self.get_pred_arr(x)
		pred_chains_arr = np.concatenate([self.posterior_weights.reshape(-1,1) * self.weight_norm, -2. * self.LLs.reshape(-1,1), pred], axis = 1)
		if file:
			np.savetxt(file, pred_chains_arr)
			return None
		return pred_chains_arr

	def y_test_prob(self, test_index = 0):
		"""
		n.b. need to call setup_LL before using this method, as it needs to calculate
		test likelihood values.
		returns single probability, as nn parameters are marginalised over:
		p(y_test | x_test, x_train, y_train) = sum_i p(y_test | y_pred(theta_i), x_test, x_train, y_train) p(y_pred(theta_i) | x_test, x_train, y_train)
		"""
		try:
			x = self.x[test_index, :].reshape(1,-1)
		except KeyError:
			x = self.x[test_index].reshape(1,-1)	
		try:
			y_true = self.y[test_index, :].reshape(1,-1)
		except KeyError:
			y_true = self.y[test_index].reshape(1,-1)
		#definitely not most efficient way of doing this, using both model.evaluate and model.predict,
		#but efficiency here not particularly important, and just stole methods from keras_forward, so it will do
		test_LL = self.get_test_LL(x, y_true)
		y_test_prob = np.sum(np.exp(test_LL) * self.posterior_weights)
		return y_test_prob

	def y_custom_prob(self, y, test_index = 0):
		"""
		as y_test_prob(), but takes y as argument, so don't have to use true value
		"""
		try:
			x = self.x[test_index, :].reshape(1,-1)
		except KeyError:
			x = self.x[test_index].reshape(1,-1)	
		LL = self.get_test_LL(x, y)
		y_prob = np.sum(np.exp(LL) * self.posterior_weights)
		return y_prob

	def get_test_LL(self, x, y):
		"""
		calculate log-likelihood LL(y_test | y_pred(weights), x_test)
		for all y_pred/weights from weights posterior file
		"""
		test_LL = np.zeros_like(self.posterior_weights)
		for i in range(len(self.posterior_weights)):
			self.set_k_weights(self.nn_param_sets[i])
			test_LL[i] = self.LL(x, y)
		return test_LL

	def get_pred_arr(self, x):
		"""
		get set of predictions (vectors) for each set of nn parameters
		"""
		try:
			pred = np.zeros((len(self.posterior_weights), self.y.shape[1]))
		except KeyError:
			pred = np.zeros((len(self.posterior_weights), self.y.shape[0]))
		for i in range(len(self.posterior_weights)):
			self.set_k_weights(self.nn_param_sets[i])
			pred[i,:] = self.model.predict(x)
		return pred

	def y_pred_paramnames(self, file_root):
		"""
		file_root doesn't need to include ".paramnames"
		"""
		#for now just writes file to keras chains dir
		output_tools.write_pred_paramnames(self.num_outputs, file_root, True, False, False, False)

	def setup_LL(self, loss):
		"""
		adapted from keras_model (keras_forward.py)
		"""
		self.LL_dim = self.num_outputs
		if loss == 'squared_error':
			self.model.compile(loss='mse', optimizer='rmsprop') 
			#temporary
			self.LL_const = -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))
			self.LL = self.calc_gauss_LL
			#longer term solution (see comments above)
			#self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif loss == 'categorical_crossentropy':
			self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
			self.LL_const = 0.
			self.LL = self.calc_cross_ent_LL
		elif loss == 'av_squared_error':
			self.model.compile(loss='mse', optimizer='rmsprop')
			#temporary
			self.LL_const = -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var) + np.log(self.LL_dim))
			self.LL = self.calc_av_gauss_LL
		elif loss == 'av_categorical_crossentropy':
			self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
			self.LL_const = 0.
			self.LL = self.calc_av_cross_ent_LL
		else:
			raise NotImplementedError

	def calc_gauss_LL(self, x, y):
		"""
		adapted from keras_model (keras_forward.py)
		"""
		LL = - self.LL_dim / (2. * self.LL_var) * self.model.evaluate(x, y, batch_size = 1, verbose = 0) + self.LL_const  
		return LL

	def calc_av_gauss_LL(self, x, y):
		"""
		adapted from keras_model (keras_forward.py)
		"""
		LL = - 1 / (2. * self.LL_var) * self.model.evaluate(x, y, batch_size = 1, verbose = 0) + self.LL_const  
		return LL

	def calc_cross_ent_LL(self, x, y):
		"""
		adapted from keras_model (keras_forward.py)
		"""
		return -1. * self.model.evaluate(x, y, batch_size = 1, verbose = 0)

	def calc_av_cross_ent_LL(self, x, y):
		"""
		adapted from keras_model (keras_forward.py)
		"""
		self.LL_const = -1 * np.log((self.model.predict(x)).prod(axis = 0).sum())
		return - 1. * self.model.evaluate(x, y, batch_size = 1, verbose = 0) + self.LL_const

	def get_weight_info(self):
		"""
		nicked from keras_model (keras_forward.py)
		"""
		trainable_weights = tf.keras.backend.get_session().run(self.model.trainable_weights)
		for layer_weight in trainable_weights:
			layer_shape = layer_weight.shape
			self.weight_shapes.append(layer_shape)

	def get_weight_shapes(self):
		"""
		nicked from keras_model (keras_forward.py)
		"""
		return self.weight_shapes

	def get_model_summary(self):
		"""
		nicked from keras_model (keras_forward.py)
		"""
		return self.model.summary()

	def get_model_weights(self):
		"""
		nicked from keras_model (keras_forward.py)
		"""
		return self.model.get_weights()

	def set_model_weights(self, weights):
		"""
		nicked from keras_model (keras_forward.py)
		"""
		self.model.set_weights(weights)

	def set_k_weights(self, new_oned_weights):
		"""
		nicked from keras_model (keras_forward.py)
		"""
		new_weights = []
		start_index = 0
		for weight_shape in self.get_weight_shapes():
			weight_size = np.prod(weight_shape)
			new_weights.append(new_oned_weights[start_index:start_index + weight_size].reshape(weight_shape))
			start_index += weight_size
		self.set_model_weights(new_weights)

def get_y_not_true(y_true):
	"""
	assumes y_true one hot vector of shape n, 
	returns n-1 vectors each with one 1 at index other than
	one that it occupies in y_true 
	"""
	indices = len(y_true)
	one_hot_list = []
	y_true_ind = np.argmax(y_true)
	y_indices = range(indices)
	del y_indices[y_true_ind]
	zero_arr = np.zeros_like(y_true).reshape(1, -1)
	for i in y_indices:
		temp = np.copy(zero_arr)
		temp[i] = 1.
		one_hot_list.append(temp)
	return one_hot_list


