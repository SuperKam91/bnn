#########commercial modules
import numpy as np

#####could definitely merge some of these functions with some kwarg magic, but cba

def calc_num_weights(num_inputs, layer_sizes, num_outputs):
	"""
	calculate total number of weights (and biases), 
	i.e. the dimensionality of the inference problem
	"""
	n = (num_inputs + 1) * layer_sizes[0]
	for i in range(1, len(layer_sizes)):
		n += (layer_sizes[i-1] + 1) * layer_sizes[i] 
	n += (layer_sizes[-1] + 1) * num_outputs
	return n

def calc_num_weights2(num_inputs, layer_sizes, num_outputs, trainable_arr):
	"""
	accounts for fact that layers may not be trainable 
	"""
	n = 0
	if trainable_arr[0]:
		n += (num_inputs + 1) * layer_sizes[0]
	for i in range(1, len(layer_sizes)):
		if trainable_arr[i]:	
			n += (layer_sizes[i-1] + 1) * layer_sizes[i] 
	if trainable_arr[-1]:
		n += (layer_sizes[-1] + 1) * num_outputs
	return n

def calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr):
	"""
	accounts for fact that certain weight matrices / bias vectors may not be
	trainable
	"""
	n = 0
	if m_trainable_arr[0]:
		n += num_inputs * layer_sizes[0]
	if b_trainable_arr[0]:
		n += layer_sizes[0]
	for i in range(1, len(layer_sizes)):
		if m_trainable_arr[i]:	
			n += layer_sizes[i-1] * layer_sizes[i]
		if b_trainable_arr[i]:
			n += layer_sizes[i]
	if m_trainable_arr[-1]:
		n += layer_sizes[-1] * num_outputs
	if b_trainable_arr[-1]:
		n += num_outputs
	return n

def get_weight_shapes(num_inputs, layer_sizes, num_outputs):
	"""
	adapted from original tf_model.get_weight_shapes() 
	to convert from method to function
	"""
	weight_shapes = []
	input_size = num_inputs
	for i, layer in enumerate(layer_sizes):
		weight_shapes.append((input_size, layer))
		weight_shapes.append((layer,))
		input_size = layer
	weight_shapes.append((input_size, num_outputs)) 
	weight_shapes.append((num_outputs,))
	return weight_shapes	


def get_weight_shapes2(num_inputs, layer_sizes, num_outputs, trainable_arr):
	"""
	see calc_num_weights2 for relevant of suffix
	"""
	weight_shapes = []
	input_size = num_inputs
	for i, layer in enumerate(layer_sizes):
		if trainable_arr[i]:	
			weight_shapes.append((input_size, layer))
			weight_shapes.append((layer,))
		input_size = layer
	if trainable_arr[-1]:
		weight_shapes.append((input_size, num_outputs)) 
		weight_shapes.append((num_outputs,))
	return weight_shapes	

def get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr):
	"""
	see calc_num_weights3 for relevant of suffix
	"""
	weight_shapes = []
	input_size = num_inputs
	for i, layer in enumerate(layer_sizes):
		if m_trainable_arr[i]:	
			weight_shapes.append((input_size, layer))
		if b_trainable_arr[i]:
			weight_shapes.append((layer,))
		input_size = layer
	if m_trainable_arr[-1]:
		weight_shapes.append((input_size, num_outputs)) 
	if b_trainable_arr[-1]:
		weight_shapes.append((num_outputs,))
	return weight_shapes	

def get_num_inputs(x):
	return np.prod(x.shape[1:], dtype = int)

def get_num_outputs(y):
	return np.prod(y.shape[1:], dtype = int)		

def reshape_x_y_twod(x, y):
	"""
	tf placeholders expected 2-d arrays not 1-d
	so if x or y are 1-d, converts to 2-d of shape
	(n,1). np needs twod arrays to matmul (apparently).
	probably best to always do it with 1d input / output arrays
	or it will no doubt cause an issue
	tested to ensure it does everything by viewing (not copying)
	"""
	if len(x.shape) == 1:
		x = oned_arr_2_twod_arr(x)
	if len(y.shape) == 1:
		y = oned_arr_2_twod_arr(y)
	return x, y

def oned_arr_2_twod_arr(x):
	return x.reshape(-1,1)

def get_degen_dependence_lengths(weight_shapes, independent = False):
	"""
	get dependence_lengths for inverse_prior class,
	i.e. the lengths in the param array which correspond
	to contiguous dependent random variables.
	assumes each layer is degenerate in the parameters
	across the nodes
	"""
	if independent:
		return [1]
	else:
		dependence_lengths = []
		for weight_shape in weight_shapes:
			if len(weight_shape) == 1: #bias
				dependence_lengths.append(weight_shape[0])
			else:
				dependence_lengths.extend([weight_shape[1]] * weight_shape[0])
	return dependence_lengths

num_inputs = 3
layer_sizes = [4,5]
num_outputs = 6
m_trainable_arr = [True, True, True]
b_trainable_arr = [False, True, False]
weight_shapes = get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
print weight_shapes
print get_degen_dependence_lengths(weight_shapes)

