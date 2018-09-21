#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import tf_graphs as tfgs

class tf_model():
	def __init__(self, tf_graph, x_tr, y_tr, batch_size, layer_sizes):
		self.x_tr = x_tr
		self.y_tr = y_tr
		self.m = x_tr.shape[0]
		self.num_outputs = np.prod(y_tr.shape[1:]) #assume same shape as output of nn
		self.num_inputs = np.prod(x_tr.shape[1:])
		self.batch_size = batch_size
		self.num_complete_batches = int(np.floor(float(self.m)/self.batch_size))
		self.num_batches = int(np.ceil(float(self.m)/self.batch_size))
		if type(layer_sizes[0]) == 'list' or type(layer_sizes[0]) == 'tuple':
			self.weight_shapes = layer_sizes #todo implement get_weight_shapes() to work for conv nets
		else:
			self.get_weight_shapes(layer_sizes) 
		self.weights_ph = tuple([tf.placeholder(dtype=tf.float64, shape=weight_shape) for weight_shape in self.weight_shapes]) #think feed_dict keys have to be immutable
		self.x_ph = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, self.num_inputs])
		self.y_ph = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, self.num_outputs])
		self.pred = tf_graph(self.x_ph, self.weights_ph)

	def get_weight_shapes(self, layer_sizes):
		"""
		currently only works for vanilla nns.
		layer_sizes should be a list of number of nodes for each
		*hidden* layer
		"""
		self.weight_shapes = []
		input_size = self.num_inputs
		for layer in layer_sizes:
			self.weight_shapes.append((input_size, layer))
			self.weight_shapes.append((layer,))
			input_size = layer
		#output layer
		self.weight_shapes.append((input_size, self.num_outputs)) #use input_size in case layer_sizes is empty
		self.weight_shapes.append((self.num_outputs,))

	def setup_LL(self, fit_metric, LL_var = 1.):
		"""
		also only currently supports constant variance, but easily upgradable
		"""
		if self.m <= self.batch_size:
		    self.batch_generator = None
		else:
		    self.batch_generator = self.create_batch_generator()
		if fit_metric == 'chisq':
		    #temporary
		    LL_dim = self.m * self.num_outputs
		    self.LL_const = -0.5 * LL_dim * (np.log(2. * np.pi) + np.log(LL_var))
		    self.LL = self.calc_gauss_LL()
		    #longer term solution (see comments above)
		    #self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
		elif fit_metric == 'categorical_crossentropy':
		    self.LL_const = 0.
		    self.LL = self.calc_cross_ent_LL()
		else:
		    raise NotImplementedError

	def calc_gauss_LL(self, LL_var = 1.):
	    """
	    currently only supports constant variance, but can easily be upgraded
	    if necessary.
	    not using explicit tf functions seems to speed up process
	    """
	    diff = self.pred - self.y_ph
	    sq_diff = diff * diff
	    chi_sq = -1. / (2. * LL_var) * tf.reduce_sum(sq_diff)
	    return self.LL_const + chi_sq 

	def calc_cross_ent_LL(self):
	    """
	    calc cross entropy and flip sign to get llhood
	    n.b. tf.losses.softmax_cross_entropy first applies softmax to pred before calculating
	    cross entropy, then takes average over m.
	    pred should be of shape (m, num_classes), y should be of shape (m, num_classes) where each of the m elements
	    should be a one-hot vector (as is case with keras)
	    """
	    return - self.m * tf.losses.softmax_cross_entropy(self.y_ph, self.pred)

	def __call__(self, oned_weights):
		"""
		sets keras.Model weights, gets new batch of training data (or full batch), 
		evaluates log likelihood function and returns its value.
		to be passed to polychord as loglikelihood function
		"""
		x_batch, y_batch = self.get_batch()
		weights = self.get_tf_weights(oned_weights)
		LL = tf.Session().run(self.LL, feed_dict={self.x_ph: x_batch, self.y_ph: y_batch, self.weights_ph: weights})
		return LL

	def get_tf_weights(self, new_oned_weights):
		"""
		adapted from set_k_weights() in keras_forward.py
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

	num_inputs = 2
	num_outputs = 2
	m = 3
	batch_size = 3
	a1_size = 5
	np.random.seed(1337)
	x_tr = np.random.random((m, num_inputs))
	y_tr = np.array([1,0,0,1,1,0]).reshape(3,2)
	tf_graph = tfgs.slp_graph
	layer_sizes = [a1_size]
	tfm = tf_model(tf_graph, x_tr, y_tr, batch_size, layer_sizes)
	w = np.arange(27)
	fit_metric = 'chisq'
	tfm.setup_LL(fit_metric)
	tfm(w)

if __name__ == '__main__':
	main()