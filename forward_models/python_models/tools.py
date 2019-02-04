#########commercial modules
import numpy as np

#####could definitely merge some of these functions with some kwarg magic, but cba

def calc_num_weights(num_inputs, layer_sizes, num_outputs):
	"""
	calculate total number of weights (and biases), 
	i.e. the dimensionality of the inference problem
	"""
	#no hidden layers
	if len(layer_sizes) == 0:
		return (num_inputs + 1) * num_outputs
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
	if len(layer_sizes) == 0:
		if trainable_arr[0]:
			return (num_inputs + 1) * num_outputs
		else:
			return n
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
	if len(layer_sizes) == 0:
		if m_trainable_arr[0]:
			n += num_inputs * num_outputs
		if b_trainable_arr[0]:
			n += num_outputs
		return n			
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

def calc_num_weights_layers(weight_shapes):
	"""
	calculate total number of (trainable) weights (and biases) in each layer, 
	"""
	#no hidden layers
	#don't think this bit is needed, but not tested yet. kj feb 19
	# if len(weight_shapes) == 2:
		# return [np.prod(weight_shapes[0]) + weight_shapes[1][0]]
	n_layer_weights = []
	for i in range(len(weight_shapes) / 2):
		n_layer_weights.append(np.prod(weight_shapes[2 * i]) + weight_shapes[2 * i + 1][0])
	return n_layer_weights

def calc_num_weights_layers2(weight_shapes, trainable_arr):
	"""
	accounts for fact layers may not be trainable.
	in this case, same as calc_num_weights_layers
	"""
	return calc_num_weights_layers(weight_shapes) 

def calc_num_weights_layers3(weight_shapes, m_trainable_arr, b_trainable_arr):
	"""
	accounts for fact that some biases or weight matrices may not be trainable
	"""
	n_layer_weights = []
	j = 0
	for i in range(len(m_trainable_arr)):
		if m_trainable_arr[i] and b_trainable_arr[i]:
			n_layer_weights.append(np.prod(weight_shapes[j]) + weight_shapes[j + 1][0])
			j += 2
		elif m_trainable_arr[i]:
			n_layer_weights.append(np.prod(weight_shapes[j]))
			j += 1
		elif b_trainable_arr[i]:
			n_layer_weights.append(weight_shapes[j][0])
			j += 1
	return n_layer_weights

def calc_num_weights(num_inputs, layer_sizes, num_outputs):
	"""
	calculate total number of weights (and biases), 
	i.e. the dimensionality of the inference problem
	"""
	#no hidden layers
	if len(layer_sizes) == 0:
		return (num_inputs + 1) * num_outputs
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
	if len(layer_sizes) == 0:
		if trainable_arr[0]:
			return (num_inputs + 1) * num_outputs
		else:
			return n
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
	if len(layer_sizes) == 0:
		if m_trainable_arr[0]:
			n += num_inputs * num_outputs
		if b_trainable_arr[0]:
			n += num_outputs
		return n			
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

def get_degen_dependence_lengths2(weight_shapes, independent = False):
	"""
	same as above but assumes that one just wants one weight (the first weight per node) from each node in a layer
	to be sorted w.r.t. corresponding weight in other nodes. This should be enough to ensure nodes aren't degenerate.
	does this for weight corresponding to first node of previous layer, for each node in current layer.
	doesn't do it for biases, i.e. assumes these aren't dependent. this means in the unlikely event a layer only has biases, they will be
	treated as independent, and will be degenerate. However, this scenario should never occur in practice.
	e.g. for given layer, one weight per node will be assumed to be dependent with the corresponding weight in the other
	nodes in the layer. the remaining weights, plus the biases are assumed to be independent.
	"""
	if independent:
		return [1]
	else:
		dependence_lengths = []
		for weight_shape in weight_shapes:
			if len(weight_shape) == 1: #bias
				dependence_lengths.extend(weight_shape[0] * [1])
			else:
				dependence_lengths.append(weight_shape[1])
				dependence_lengths.extend((weight_shape[0] - 1) * weight_shape[1] * [1])
	return dependence_lengths

def get_degen_dependence_lengths3(weight_shapes, independent = False):
	"""
	same as get_degen_dependence_lengths() but treats all weights/biases in final layer as independent.
	note it also assumes that the final layer has both weights and biases
	"""
	if independent:
		return [1]
	else:
		dependence_lengths = []
		for i in range(len(weight_shapes) - 2):
			if len(weight_shapes[i]) == 1: #bias
				dependence_lengths.append(weight_shapes[i][0])
			else:
				dependence_lengths.extend([weight_shapes[i][1]] * weight_shapes[i][0])
		dependence_lengths.extend((weight_shapes[-2][0] + 1) * weight_shapes[-2][1] * [1]) #+1 is for biases. assumes 2nd dim of final two weight_shapes are same (should be)
	return dependence_lengths

def get_degen_dependence_lengths4(weight_shapes, independent = False):
	"""
	essentially combines get_degen_dependence_lengths2 and get_degen_dependence_lengths3
	"""
	if independent:
		return [1]
	else:
		dependence_lengths = []
		for i in range(len(weight_shapes) - 2):
			if len(weight_shapes[i]) == 1: #bias
				dependence_lengths.extend(weight_shapes[i][0] * [1])
			else:
				dependence_lengths.append(weight_shapes[i][1])
				dependence_lengths.extend((weight_shapes[i][0] - 1) * weight_shapes[i][1] * [1])
		dependence_lengths.extend((weight_shapes[-2][0] + 1) * weight_shapes[-2][1] * [1]) #+1 is for biases. assumes 2nd dim of final two weight_shapes are same (should be)
	return dependence_lengths

def get_hyper_dependence_lengths(weight_shapes, granularity):
	"""
	calculates hyper dependence lengths from weight shapes based on granularity.
	if granularity is single, means a single set (two) of hyperparams is used for
	whole nn, so returns a list containing 1.
	if granularity is layer, calculates number of params in each layer, and returns
	a list of these values.
	if granularity is input_size, calculates dependence_lengths as required in inverse_prior class,
	as this gives the contiguous parameters which multiply the same input taken from the previous
	layer. this is the granularity used in neal's ARD model, as it essentially determines which inputs
	are relevant and which aren't
	"""
	if granularity == 'single':
		return [1]
	elif granularity == 'layer':
		return calc_num_weights_layers(weight_shapes)
	elif granularity == 'input_size':
		return get_degen_dependence_lengths(weight_shapes)

def round_probabilities(p):
	"""
	for given row, set element with highest value to one,
	and all others to zero.
	e.g.
	[[0.1,0.2,0.7], 
	[0.5,0.2,0.3]]
	- > 
	[[0,0,1], 
	[1,0,0]].
	not been tested for case of equal probabilities in a row
	"""
	return (p == p.max(axis=1)[:,None]).astype(float) 

def decode_onehot(one_hot_v):
	"""
	take set of one hot vectors (represented by each row of the matrix)
	and convert each one to a scalar corresponding to the index of the one hot.
	i.e. takes an (m, num_classes) array and returns a (m,) array where each element in the
	latter can take integer values in [0, num_classes).
	e,g,
	[[0,0,1], [1,0,0], [0,1,0]]
	- >
	[2,0,1] 
	"""
	return one_hot_v.argmax(axis=1)

def check_dtypes(y_true, y_pred):
	"""
	ensures both y_true and y_pred are numpy arrays of type np.float64.
	if not some keras metric functions may screw up
	"""
	y_true = y_true.astype(np.float64, copy = False)
	y_pred = y_pred.astype(np.float64, copy = False)
	return y_true, y_pred

def get_km_weight_magnitudes(km):
	model_weight_arrs = km.get_weights()
	weight_mags = []
	for model_weight_arr in model_weight_arrs:
		weight_mags.append(np.linalg.norm(model_weight_arr))
	return weight_mags, np.linalg.norm(weight_mags)

if __name__ == '__main__':
	num_inputs = 2
	layer_sizes = [3,2]
	num_outputs = 2
	m_trainable_arr = [False, False, True]
	b_trainable_arr = [False, True, False]
	print calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	weight_shapes = get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	print get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	print calc_num_weights_layers3(weight_shapes, m_trainable_arr, b_trainable_arr)
	print get_degen_dependence_lengths(weight_shapes, independent = False)