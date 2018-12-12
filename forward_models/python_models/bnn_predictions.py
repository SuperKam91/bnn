#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import keras_models as kms
import input_tools

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
		self.posterior_weights, self.LL, self.nn_params = input_tools.get_chains_data(chains_file)
		self.nn_param_sets = [self.nn_params[i,:] for i in range(len(self.posterior_weights))]
		self.LL_var = 1. #for calculating LL on test data

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
		argmax = np.argmax(self.LL)
		MLE_nn_param_set = self.nn_param_sets[argmax]
		self.set_k_weights(MLE_nn_param_set)
		return self.model.predict(self.x)		

	def y_pred_chains(self, file = None, test_index = 0):
		"""
		note LL values here are LLs from weight posterior,
		not pred posterior.
		assumes self.x has shape (m,n) or (m,), not (n,)
		TODO: add option to include LLs from pred posterior instead
		of LLs from training,
		by evaluating LL of real y given pred y, x, weights.
		"""
		try:
			x = self.x[test_index, :].reshape(1,-1)
		except KeyError:
			x = self.x[test_index].reshape(1,-1)
		try:
			y_true = self.y[test_index, :].reshape(1,-1)
		except KeyError:
			y_true = self.y[test_index].reshape(1,-1)
		#definitely not most efficient way of doing this, using model.evaluate and model.predict,
		#but efficiency here not particularly important, and just stole methods from keras_forward, so it will do
		pred_LL = self.get_pred_LL(x, y_true)
		pred = self.get_pred_arr(x)
		pred_chains_arr = np.concatenate([self.posterior_weights.reshape(-1,1), self.LL.reshape(-1,1), pred], axis = 1)

	def get_pred_LL(x, y):
		preds = np.zeros((len(self.posterior_weights), 1))
		for i in range(len(self.posterior_weights)):
			self.set_k_weights(self.nn_param_sets[i])
			pred_LL[i] = self.LL(x, y)
		return pred_LL

	def get_pred_arr(x):
		preds = np.zeros((len(self.posterior_weights), y.shape()[1]))
		for i in range(len(self.posterior_weights)):
			self.set_k_weights(self.nn_param_sets[i])
			preds[i] = self.model.predict(x)
		return preds

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