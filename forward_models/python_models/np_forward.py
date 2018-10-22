#########commercial modules
import numpy as np

#in-house modules
import np_models as nnns
import tools
import PyPolyChord
import PyPolyChord.settings
import priors
import polychord_tools
import output
import input_tools

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
		    LL_dim = self.batch_size * self.num_outputs
		    self.LL_const = -0.5 * LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))
		    self.LL = self.calc_gauss_LL
		    #longer term solution (see comments above in keras_forward)
		    #self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif ll_type == 'categorical_crossentropy':
		    self.LL_const = 0.
		    self.LL = self.calc_cross_ent_LL
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

	def calc_cross_ent_LL(self, x, y, weights):
	    """
	    calculates categorical cross entropy (information entropy).
	    MAKES NO ASSUMPTIONS ABOUT FORM OF Y OR PRED (i.e. that they're normalised, and no softmax function is applied here)
	    """
	    pred = self.model(x, weights)
	    return - np.sum(y * np.log(pred))

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


def main():
	###### load training data
	data = 'simple_tanh'
	data_dir = '../../data/'
	data_prefix = data_dir + data
	x_tr, y_tr = input_tools.get_x_y_tr_data(data_prefix)
	batch_size = x_tr.shape[0]
	###### get weight information
	a1_size = 2
	layer_sizes = [a1_size]
	m_trainable_arr = [True, False]
	b_trainable_arr = [False, False]
	num_inputs = tools.get_num_inputs(x_tr)
	num_outputs = tools.get_num_outputs(y_tr)
	num_weights = tools.calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	#set up np model
	np_nn = nnns.slp_nn
	x_tr, y_tr = tools.reshape_x_y_twod(x_tr, y_tr)
	npm = np_model(np_nn, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr)
	ll_type = 'gauss'
	npm.setup_LL(ll_type)
	###### test llhood output
	# weight_type = 'linear'
	# weight_f = data_dir + weight_type + '_weights.txt' 
	# w = input_tools.get_weight_data(weight_f, num_weights)
	# print npm(w)
	###### setup prior
	prior = priors.UniformPrior(-1, 1)
	###### setup polychord
	nDerived = 0
	settings = PyPolyChord.settings.PolyChordSettings(num_weights, nDerived)
	settings.base_dir = './np_chains/'
	settings.file_root = data
	settings.nlive = 200
	###### run polychord
	PyPolyChord.run_polychord(npm, num_weights, nDerived, settings, prior, polychord_tools.dumper)
	

if __name__ == '__main__':
	main()