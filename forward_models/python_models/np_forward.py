

#########commercial modules
import numpy as np

#in-house modules
import np_models as npms
import tools
import PyPolyChord
import PyPolyChord.settings
import inverse_priors
import inverse_stoc_hyper_priors as isp
import polychord_tools
import input_tools
import prior_tests
import forward_tests

class np_model():
	"""
	VERY similar to tf_model() class, could easily have these inherit from same class
	(same could probably be said for keras_model as well), might do this if i ever get
	time.
	"""
	def __init__(self, np_nn, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr = [], b_trainable_arr = []):
		"""
		subset of tf_model init()
		"""
		if len(m_trainable_arr) == 0:
			m_trainable_arr = [True] * (len(layer_sizes) + 1)
		if len(b_trainable_arr) == 0:
			b_trainable_arr = [True] * (len(layer_sizes) + 1)
		self.x_tr = x_tr
		self.y_tr = y_tr
		self.m = x_tr.shape[0]
		self.num_outputs = np.prod(y_tr.shape[1:], dtype = int) 
		self.num_inputs = np.prod(x_tr.shape[1:], dtype = int)
		self.batch_size = batch_size
		self.num_complete_batches = int(np.floor(float(self.m)/self.batch_size))
		self.num_batches = int(np.ceil(float(self.m)/self.batch_size))
		self.get_weight_shapes(layer_sizes, m_trainable_arr, b_trainable_arr) 
		self.model = np_nn
		self.LL_var = 1.

	def get_weight_shapes(self, layer_sizes, m_trainable_arr, b_trainable_arr):
		"""
		adapted from tools.get_weight_shapes3 (same as tf version)
		see tools.calc_num_weights3 for relevance of trainable_arrs
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

	def setup_LL(self, ll_type):
		"""
		adapted from tf_model version
		note i have tested whether using scipy.stats would be faster for case when
		batch_size == m (i.e. fitting y_true scipy.stats.rv_continuous(mean=y_true, cov=cov)
		and then evaluating llhood using fitted_rv_continuous.logpdf(pred))),
		but for constant variance (=1) it is slower than calculating the llhood as I did
		in tf_model (but using np functions where possible).
		thought this may be due to fact that it has to invert covariance matrix,
		but then realised it surely does this when fitting?
		may be worth testing again if i ever consider more complicated covariances.
		"""
		if self.m <= self.batch_size:
		    self.batch_generator = None
		else:
		    self.batch_generator = self.create_batch_generator()
		if ll_type == 'gauss':
		    #temporary
		    self.LL_dim = self.batch_size * self.num_outputs
		    self.LL_const = -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))
		    self.LL = self.calc_gauss_LL
		    #longer term solution (see comments above in keras_forward)
		    #self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif ll_type == 'categorical_crossentropy':
		    self.LL_const = 0.
		    self.LL = self.calc_cross_ent_LL
		elif ll_type == 'av_gauss':
		    self.LL_dim = self.batch_size * self.num_outputs
		    self.LL_const = -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var) + np.log(self.LL_dim))
		    self.LL = self.calc_av_gauss_LL
		    #longer term solution (see comments above in keras_forward)
		    #self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif ll_type == 'av_categorical_crossentropy':
		    self.LL_const = 0.
		    self.LL = self.calc_av_cross_ent_LL
		else:
		    raise NotImplementedError

	def calc_gauss_LL(self, x, y, weights):
		"""
		adapted from tf_model version, and like that only supports constant
		variance.
		"""
		pred = self.model(x, weights)
		diff = pred - y
		sq_diff = diff * diff
		chi_sq = -1. / (2. * self.LL_var) * np.sum(sq_diff)
		return self.LL_const + chi_sq

	def calc_av_gauss_LL(self, x, y, weights):
		"""
		calculates LL associated with average of cost function (which keras MPE uses)
		"""
		pred = self.model(x, weights)
		diff = pred - y
		sq_diff = diff * diff
		chi_sq = -1. / (2. * self.LL_var) * 1. / self.LL_dim * np.sum(sq_diff)
		return self.LL_const + chi_sq 

	def calc_cross_ent_LL(self, x, y, weights):
	    """
	    calculates categorical cross entropy (information entropy).
	    MAKES NO ASSUMPTIONS ABOUT FORM OF Y OR PRED (i.e. that they're normalised, and no softmax function is applied here).
	    Computes total LL, not average
	    """
	    pred = self.model(x, weights)
	    return np.sum(y * np.log(pred))

	def calc_av_cross_ent_LL(self, x, y, weights):
	    """
	    Computes LL associated with average cost function of 
	    categorical crossentropy LL (which keras MPE uses). 
	    note this LL has a normalisation factor which depends on the predictions,
	    and thus has to be re-calculated upon each LL call. 
	    this completely alters the 'associated' cost function, such that it no longer
	    lines up with the keras MPE one, so it probably isn't useful.
	    but is included for completeness. 
	    """
	    pred = self.model(x, weights)
	    self.LL_const = -1 * np.log((pred**(1. / self.batch_size)).prod(axis = 0).sum())
	    return self.LL_const
	    return 1. / self.batch_size * np.sum(y * np.log(pred)) + self.LL_const

	def __call__(self, oned_weights):
		"""
		sets arrays of weights to be used in nn, gets batch and evaluates LL
		n.b. if non-constant var, LL_var and LL_const need to be updated before
		calculating LL
		"""
		x_batch, y_batch = self.get_batch()
		weights = self.get_np_weights(oned_weights)
		LL = self.LL(x_batch, y_batch, weights)
		return LL

	def test_output(self, oned_weights):
		print "one-d weights:"
		print oned_weights
		weights = self.get_np_weights(oned_weights)
		x_batch, y_batch = self.get_batch()
		print "input batch:"
		print x_batch
		print "output batch:"
		print y_batch
		print "nn output:"
		print self.model(x_batch, weights)
		print "log likelihood:"
		print self.LL(x_batch, y_batch, weights) 

	def get_np_weights(self, new_oned_weights):
		"""
		taken from get_tf_weights()
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
	data = 'FIFA_2018_Statistics'
	data_suffix = '_tr_1.csv'
	data_dir = '../../data/kaggle/'
	data_prefix = data_dir + data
	x_tr, y_tr = input_tools.get_x_y_tr_data(data_prefix, data_suffix)
	batch_size = x_tr.shape[0]
	###### get weight information
	weights_dir = '../../data/'
	a1_size = 0
	layer_sizes = []
	m_trainable_arr = [True]
	b_trainable_arr = [True]
	num_inputs = tools.get_num_inputs(x_tr)
	num_outputs = tools.get_num_outputs(y_tr)
	num_weights = tools.calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	###### check shapes of training data
	x_tr, y_tr = tools.reshape_x_y_twod(x_tr, y_tr)
	#set up np model
	np_nn = npms.slp_sm
	npm = np_model(np_nn, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr)
	ll_type = 'av_categorical_crossentropy' # 'gauss', 'av_gauss', 'categorical_crossentropy', 'av_categorical_crossentropy'
	npm.setup_LL(ll_type)
	###### test llhood output
	if "forward_test_linear" in run_string:
		forward_tests.forward_test_linear([npm], num_weights, weights_dir)
	###### setup prior
    if hyper_type == "deterministic":
        prior_types = [4]
        prior_hyperparams = [[0., 1.]]
        param_prior_types = [0]
        prior = inverse_priors.inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
        n_stoc = 0
    elif hyper_type == "stochastic":
        granularity = 'single'
        hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
        hyperprior_types = [9]
        prior_types = [4]
        hyperprior_params = [[0.1 / 2., 0.1 / (2. * 100)]]
        prior_hyperparams = [0.]
        param_hyperprior_types = [0]
        param_prior_types = [0]
        n_stoc = len(hyper_dependence_lengths)
        prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, num_weights)
	###### test prior output from nn setup
	if "nn_prior_test" in run_string:
		prior_tests.nn_prior_test(prior, n_stoc + num_weights)
	###### setup polychord
	nDerived = 0
	settings = PyPolyChord.settings.PolyChordSettings(n_stoc + num_weights, nDerived)
	settings.base_dir = './np_chains/'
	settings.file_root = data + "_slp_sm"
	settings.nlive = 200
	###### run polychord
	if "polychord1" in run_string:
		PyPolyChord.run_polychord(npm, n_stoc, num_weights, nDerived, settings, prior, polychord_tools.dumper)

if __name__ == '__main__':
	run_string = 'forward_test_linear'
	main(run_string)