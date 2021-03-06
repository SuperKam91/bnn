#for some reason if you import scipy.stats before tf, get ImportError on scipy.stats import
import inverse_priors
import inverse_stoc_hyper_priors as isp
import inverse_stoc_var_hyper_priors as isvp

#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import tf_graphs as tfgs
import tools
import PyPolyChord
import PyPolyChord.settings
import polychord_tools
import input_tools
import prior_tests
import forward_tests
import np_models as npms

class tf_model():
	"""
	WARNING
	some changes made to class won't be refelcted in tensorflow graph for some reason.
	e.g. if you change a class field in call operator, changes won't be propagated to tf graph calculation.
	To account for this in the case of stochastic likelihood variances, now pass values of var and const in placeholders via feed dict. 
	"""
	def __init__(self, tf_graph, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr = [], b_trainable_arr = [], n_stoc_var = 0):
		if len(m_trainable_arr) == 0:
			m_trainable_arr = [True] * (len(layer_sizes) + 1)
		if len(b_trainable_arr) == 0:
			b_trainable_arr = [True] * (len(layer_sizes) + 1)
		self.x_tr = x_tr
		self.y_tr = y_tr
		self.m = x_tr.shape[0]
		self.num_outputs = np.prod(y_tr.shape[1:], dtype = int) #assume same shape as output of nn
		self.num_inputs = np.prod(x_tr.shape[1:], dtype = int)
		self.batch_size = batch_size
		self.num_complete_batches = int(np.floor(float(self.m)/self.batch_size))
		self.num_batches = int(np.ceil(float(self.m)/self.batch_size))
		self.get_weight_shapes(layer_sizes, m_trainable_arr, b_trainable_arr) 
		self.weights_ph = tuple([tf.placeholder(dtype=tf.float64, shape=weight_shape) for weight_shape in self.weight_shapes]) #think feed_dict keys have to be immutable
		#n.b. if use uneven batch_sizes, need to change first dim to None
		self.x_ph = tf.placeholder(dtype = tf.float64, shape = [self.batch_size, self.num_inputs])
		self.y_ph = tf.placeholder(dtype = tf.float64, shape = [self.batch_size, self.num_outputs])
		self.LL_var_ph = tf.placeholder(dtype = tf.float64, shape = None)
		self.LL_const_ph = tf.placeholder(dtype = tf.float64, shape = None)
		self.pred = tf_graph(self.x_ph, self.weights_ph)
		self.LL_var = 1.
		self.x = 1
		self.x = 2
		self.instance_sess = tf.Session() 
		self.x = 3
		self.n_stoc_var = n_stoc_var 

	def get_weight_shapes(self, layer_sizes, m_trainable_arr, b_trainable_arr):
		"""
		adapted from tools.get_weight_shapes3.
		see calc_num_weights3 for relevance of trainable_arrs.
		"""
		self.weight_shapes = []
		input_size = self.num_inputs
		for i, layer in enumerate(layer_sizes):
			if m_trainable_arr[i]:	
				self.weight_shapes.append((input_size, layer))
			if b_trainable_arr[i]:
				self.weight_shapes.append((layer,))
			input_size = layer
		if m_trainable_arr[-1]:
			self.weight_shapes.append((input_size, self.num_outputs)) 
		if b_trainable_arr[-1]:
			self.weight_shapes.append((self.num_outputs,))

	def chisq_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))

	def categorical_crossentropy_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return 0.

	def av_chisq_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var) + np.log(self.LL_dim))

	def av_categorical_crossentropy_LL_c(self):
		"""
		copied from keras_forward.py
		"""
		return 0.

	def setup_LL(self, fit_metric):
		"""
		also only currently supports constant variance, but easily upgradable
		"""
		if self.m <= self.batch_size:
			self.batch_generator = None
		else:
			self.batch_generator = self.create_batch_generator()
		if fit_metric == 'chisq':
			#temporary
			self.LL_dim = self.batch_size * self.num_outputs
			self.LL_const_f = self.chisq_LL_c
			self.LL_const = self.LL_const_f()
			self.LL = self.calc_gauss_LL()
			#longer term solution (see comments above)
			#self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif fit_metric == 'categorical_crossentropy':
			self.LL_const_f = categorical_crossentropy_LL_c
			self.LL_const = self.LL_const_f()
			self.LL = self.calc_cross_ent_LL()
		elif fit_metric == 'av_chisq':
			self.LL_dim = self.batch_size * self.num_outputs
			self.LL_const_f = av_chisq_LL_c 
			self.LL_const = self.LL_const_f()
			self.LL = self.calc_av_gauss_LL()
			#longer term solution (see comments above in keras_forward)
			#self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif fit_metric == 'av_categorical_crossentropy':
			self.LL_const_f = av_categorical_crossentropy_LL_c
			self.LL_const = self.LL_const_f()
			self.LL = self.calc_av_cross_ent_LL()
		else:
			raise NotImplementedError
		self.LL_const = self.LL_const_f() 
		self.stoc_var_setup()

	def calc_gauss_LL(self):
	    """
	    currently only supports constant variance, but can easily be upgraded
	    if necessary.
	    not using explicit tf functions seems to speed up process
	    """
	    diff = self.pred - self.y_ph
	    chi_sq = -1. / self.LL_var_ph * tf.nn.l2_loss(diff)
	    return self.LL_const_ph + chi_sq 

	def calc_av_gauss_LL(self):
	    """
		adapted from non-average version.
		see np_forward.py implementation for more
		details concerning 'average'.
	    """
	    diff = self.pred - self.y_ph
	    chi_sq = -1. / (self.LL_var_ph * self.LL_dim) * tf.nn.l2_loss(diff)
	    return self.LL_const_ph + chi_sq 

	def calc_cross_ent_LL(self):
	    """
	    calc cross entropy and flip sign to get llhood
	    n.b. tf.losses.softmax_cross_entropy first applies softmax to pred before calculating
	    cross entropy, then takes average over batch_size.
	    pred should be of shape (batch_size, num_classes), y should be of shape (batch_size, num_classes) where each of the m elements
	    should be a one-hot vector (as is case with keras)
	    """
	    return - self.batch_size * tf.losses.softmax_cross_entropy(self.y_ph, self.pred)

	def calc_av_cross_ent_LL(self):
		"""
		adapted from non-average version.
		see np_forward.py implementation for more
		details concerning 'average'.	 
		"""
		self.LL_const = -1. * tf.log(tf.reduce_sum(tf.reduce_prod(tf.nn.softmax(self.pred)**(1. / self.batch_size), axis = 0)))
		# self.LL_const = -1 * np.log(np.sum(np.prod(npms.softmax(self.pred)**(1. / self.batch_size), axis = 0)))
		return - tf.losses.softmax_cross_entropy(self.y_ph, self.pred) + self.LL_const

	def stoc_var_setup(self):
		"""
		copied from keras_forward.py
		"""
		if self.n_stoc_var == 0:
			self.stoc_var_update = self.no_stoc_var_update
		elif self.n_stoc_var == 1:
			self.stoc_var_update = self.one_stoc_var_update
		else:
			print "only 0 or 1 stoc variances currently supported"
			raise NotImplementedError

	def no_stoc_var_update(self, LL_vars):
		"""
		copied from keras_forward.py
		"""
		return None

	def one_stoc_var_update(self, LL_vars):
		"""
		copied from keras_forward.py
		"""
		self.LL_var = LL_vars[0]
		self.LL_const = self.LL_const_f()

	def __call__(self, oned_weights):
		"""
		sets arrays of weights (in correct shapes for tf graph), gets new batch of training data (or full batch), 
		evaluates log likelihood function and returns its value by running session.
		uses feed_dict to feed values for x_ph, y_ph and weights_ph
		to be passed to polychord as loglikelihood function
		n.b. if non-constant var, LL_var and LL_const need to be updated before
		calculating LL
		"""
		x_batch, y_batch = self.get_batch()
		weights = self.get_tf_weights(oned_weights[self.n_stoc_var:])
		self.stoc_var_update(oned_weights[:self.n_stoc_var])
		LL = self.instance_sess.run(self.LL, feed_dict={self.x_ph: x_batch, self.y_ph: y_batch, self.weights_ph: weights, self.LL_var_ph: self.LL_var, self.LL_const_ph: self.LL_const})
		return LL

	def test_output(self, oned_weights):
		print "one-d weights:"
		print oned_weights
		weights = self.get_tf_weights(oned_weights[self.n_stoc_var:])
		x_batch, y_batch = self.get_batch()
		print "input batch:"
		print x_batch
		print "output batch:"
		print y_batch
		self.stoc_var_update(oned_weights[:self.n_stoc_var])
		print "LL var and const"
		print self.LL_var, self.LL_const
		pred, LL = self.instance_sess.run([self.pred, self.LL], feed_dict={self.x_ph: x_batch, self.y_ph: y_batch, self.weights_ph: weights, self.LL_var_ph: self.LL_var, self.LL_const_ph: self.LL_const})
		print "nn output:"
		print pred
		print "log likelihood:"
		print LL 

	def get_tf_weights(self, new_oned_weights):
		"""
		adapted from set_k_weights() in keras_forward.py
		weight matrices are still to right of previous activiation in multiplication
		"""
		weights = []
		start_index = 0
		for weight_shape in self.weight_shapes:
			weight_size = np.prod(weight_shape)
			weights.append(new_oned_weights[start_index:start_index + weight_size].reshape(weight_shape))
			start_index += weight_size
		return weights

	def get_batch(self):
	    """
	    copied from keras_forward.py
	    """
	    if self.m <= self.batch_size:
	        return self.x_tr, self.y_tr
	    else:
	        return self.batch_generator.next()
	        
	def create_batch_generator(self):
	    """
		copied from keras_forward.py
	    """
	    i = 0
	    batches = self.create_batches()
	    while True:
	        if i < self.num_batches:
	            pass #don't need to create new random shuffle of training data
	        else:
	            batches = self.create_batches()
	            i = 0
	        yield batches[i]
	        i += 1

	def create_batches(self):
	    """
	    copied from keras_forward.py
	    """
	    batches = []
	    # Step 1: Shuffle x, y
	    permutation = np.random.permutation(self.m)
	    shuffled_x = self.x_tr[permutation]
	    shuffled_y = self.y_tr[permutation]
	    # Step 2: Partition (shuffled_x, shuffled_y). Minus the end case.
	    # number of batches of size self.batch_size in your partitionning
	    for i in range(self.num_complete_batches):
	        batch_x = shuffled_x[self.batch_size * i: self.batch_size * (i + 1)]
	        batch_y = shuffled_y[self.batch_size * i: self.batch_size * (i + 1)]
	        batch = (batch_x, batch_y)
	        batches.append(batch)

	    # Handling the end case (last batch < self.batch_size)
	    if self.num_complete_batches != self.num_batches:
	        batch_x = shuffled_x[self.num_complete_batches * self.batch_size:]
	        batch_y = shuffled_y[self.num_complete_batches * self.batch_size:]
	        batch = (batch_x, batch_y)
	        batches.append(batch)
	    return batches

def main(run_string):
	###### load training data
	data = 'bh_50'
	data_suffix = '_tr_1.csv'
	data_dir = '../../data/uci/'
	data_prefix = data_dir + data
	x_tr, y_tr = input_tools.get_x_y_tr_data(data_prefix, data_suffix)
	x_tr, y_tr = input_tools.get_x_y_tr_data(data_prefix, data_suffix)
	x_tr = np.genfromtxt('../../data/linear_input_data.txt', delimiter = ',')
	y_tr = x_tr
	batch_size = x_tr.shape[0]
	###### get weight information
	weights_dir = '../../data/' #for forward test
	a1_size = 0
	num_inputs = tools.get_num_inputs(x_tr)
	num_outputs = tools.get_num_outputs(y_tr)
	layer_sizes = [1, num_inputs] * 2
	m_trainable_arr = [True, True] * 2 + [False]
	b_trainable_arr = [True, True] * 2 + [False]
	num_weights = tools.calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	###### check shapes of training data
	x_tr, y_tr = tools.reshape_x_y_twod(x_tr, y_tr)
	###### setup prior
	hyper_type = "deterministic" # "stochastic" or "deterministic"
	var_type = "deterministic" # "stochastic" or "deterministic"
	weight_shapes = tools.get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	if hyper_type == "deterministic" and var_type == "deterministic":
		prior_types = [4]
		prior_hyperparams = [[0., 1.]]
		param_prior_types = [0]
		prior = inverse_priors.inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
		n_stoc = 0
		n_stoc_var = 0
	elif hyper_type == "stochastic" and var_type == "deterministic":
		granularity = 'single'
		hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
		hyperprior_types = [9]
		prior_types = [4]
		hyperprior_params = [[1. / 2., 1. / (2. * 100)]]
		prior_hyperparams = [0.]
		param_hyperprior_types = [0]
		param_prior_types = [0]
		n_stoc = len(hyper_dependence_lengths)
		prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, num_weights)
		n_stoc_var = 0
	elif hyper_type == "stochastic" and var_type == "stochastic":
		granularity = 'single'
		hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
		var_dependence_lengths = [1]
		n_stoc_var = len(var_dependence_lengths)
		hyperprior_types = [9]
		var_prior_types = [10]
		prior_types = [4]
		hyperprior_params = [[1. / 2., 1. / (2. * 100)]]
		var_prior_params = [[1. / 2., 1. / (2. * 100)]]
		prior_hyperparams = [0.]
		param_hyperprior_types = [0]
		var_param_prior_types = [0]
		param_prior_types = [0]
		n_stoc = len(hyper_dependence_lengths)
		prior = isvp.inverse_stoc_var_hyper_prior(hyperprior_types, var_prior_types, prior_types, hyperprior_params, var_prior_params, prior_hyperparams, hyper_dependence_lengths, var_dependence_lengths, dependence_lengths, param_hyperprior_types, var_param_prior_types, param_prior_types, n_stoc, n_stoc_var, num_weights)
	###### test prior output from nn setup
	if "nn_prior_test" in run_string:
		prior_tests.nn_prior_test(prior, n_stoc + n_stoc_var + num_weights)
	#setup tf graph
	tf_graph = tfgs.mlp_ResNet_2
	tfm = tf_model(tf_graph, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr, n_stoc_var)
	fit_metric = 'chisq' # 'chisq', 'av_chisq', 'categorical_crossentropy', 'av_categorical_crossentropy'
	tfm.setup_LL(fit_metric)
	###### test llhood output
	if "forward_test_linear" in run_string:
		forward_tests.forward_test_linear([tfm], num_weights + n_stoc_var, weights_dir)
	###### setup polychord
	nDerived = 0
	settings = PyPolyChord.settings.PolyChordSettings(n_stoc + n_stoc_var + num_weights, nDerived)
	settings.base_dir = './tf_chains/'
	settings.file_root = data + '_slp_sh_sv_sm'
	settings.nlive = 1000
	###### run polychord
	if "polychord1" in run_string:
		PyPolyChord.run_polychord(tfm, n_stoc, n_stoc_var, num_weights, nDerived, settings, prior, polychord_tools.dumper)

if __name__ == '__main__':
	run_string = 'forward_test_linear'
	main(run_string)