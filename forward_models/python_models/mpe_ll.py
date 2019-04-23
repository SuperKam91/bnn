#########commercial modules
import numpy as np
import tensorflow as tf

class mpe_ll():
	"""
	takes compiles keras model and uses its current weights to evaluate ll values
	using model.predict (n.b. doesn't use model.evaluate as I think it includes regularisation terms).
	assumes x and y in same form as used when training network (i.e. whether they are scaled the same).
	"""
	def __init__(self, k_model_c, num_outputs, batch_size = None, LL_var = 1.):
		"""
		assign model to class, calculate shape of weights, and arrays containing them (latter possibly redundant)
		"""
		self.model = k_model_c
		self.x = x
		self.y = y
		self.num_outputs = self.y.shape[1]
		if not batch_size:
			self.batch_size = self.y.shape[0]
		else:
			self.batch_size = batch_size
		self.LL_var = LL_var 

	def gauss_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))

	def categorical_crossentropy_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return 0.

	def av_gauss_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var) + np.log(self.LL_dim))

	def av_categorical_crossentropy_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return 0.

	def setup_LL(self, ll_type):
		"""
		adapted from keras_forward.py
		needs to be re-ran if batch size or ll variance are changed
		"""
		if ll_type == 'gauss' or ll_type == 'squared_error':
		    #temporary
			self.LL_dim = self.batch_size * self.num_outputs
			self.LL_const_f = self.gauss_LL_c
			self.LL = self.calc_gauss_LL
			#longer term solution (see comments above in keras_forward)
			#self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif ll_type == 'categorical_crossentropy':
			self.LL_const_f = categorical_crossentropy_LL_c
			self.LL = self.calc_cross_ent_LL
		elif ll_type == 'av_gauss' or ll_type == 'av_squared_error':
			self.LL_dim = self.batch_size * self.num_outputs
			self.LL_const_f = av_gauss_LL_c
			self.LL = self.calc_av_gauss_LL
			#longer term solution (see comments above in keras_forward)
			#self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif ll_type == 'av_categorical_crossentropy':
			self.LL_const_f = av_categorical_crossentropy_LL_c
			self.LL = self.calc_av_cross_ent_LL
		else:
			raise NotImplementedError
		self.LL_const = self.LL_const_f() 
		self.stoc_var_setup()

	def calc_gauss_LL(self):
		"""
		adapted from np_forward.py
		"""
		pred = self.model.predict(self.x)
		diff = pred - self.y
		sq_diff = diff * diff
		chi_sq = -1. / (2. * self.LL_var) * np.sum(sq_diff)
		return self.LL_const + chi_sq

	def calc_av_gauss_LL(self):
		"""
		adapted from np_forward.py
		"""
		pred = self.model.predict(self.x)
		diff = pred - self.y
		sq_diff = diff * diff
		chi_sq = -1. / (2. * self.LL_var) * 1. / self.LL_dim * np.sum(sq_diff)
		return self.LL_const + chi_sq 

	def calc_cross_ent_LL(self):
	    """
		adapted from np_forward.py
	    """
	    pred = self.model.predict(self.x)
	    return np.sum(self.y * np.log(pred))

	def calc_av_cross_ent_LL(self):
	    """
		adapted from np_forward.py
	    """
	    pred = self.model.predict(self.x)
	    self.LL_const = -1 * np.log((pred**(1. / self.batch_size)).prod(axis = 0).sum())
	    return 1. / self.batch_size * np.sum(self.y * np.log(pred)) + self.LL_const

	def __call__(self):
		return self.LL()