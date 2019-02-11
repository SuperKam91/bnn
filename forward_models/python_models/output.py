#########commercial modules
import numpy as np

#in-house modules
import tools
import PyPolyChord.output

#####could definitely merge some of these functions with some kwarg magic, but cba

def get_param_names(num_inputs, layer_sizes, num_outputs):
	"""
	writes list of param_names for weight array which corresponds to (row-wise)
	weight matrices and bias vector.
	Length of weight_shapes should be even
	"""
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	param_names = []
	i = 0
	for j in range(len(weight_shapes) / 2):
		weight_row = weight_shapes[2 * j][0]
		weight_col = weight_shapes[2 * j][1]
		bias = weight_shapes[2 * j + 1][0]
		for k in range(weight_row):
			for l in range(weight_col):
				param_names.append(('p%i' %(i+1), 'w_{%i,%i,%i}' %(j+1, k+1, l+1)))
				i += 1
		for k in range(bias):
			param_names.append(('p%i' %(i+1), 'b_{%i,%i}' %(j+1, k+1)))
			i += 1
	return param_names

def get_param_names2(num_inputs, layer_sizes, num_outputs, trainable_arr):
	"""
	see calc_num_weights2 for relevant of suffix
	"""
	weight_shapes = tools.get_weight_shapes2(num_inputs, layer_sizes, num_outputs, trainable_arr)
	param_names = []
	i = 0
	layer_offset = 1
	if np.all(~np.array(trainable_arr)): #for loop won't make sense if no trainable layers
		"no trainable parameters"
		return param_names
	for j in range(len(weight_shapes) / 2):
		if not trainable_arr[j]:
			layer_offset += 1
		weight_row = weight_shapes[2 * j][0]
		weight_col = weight_shapes[2 * j][1]
		bias = weight_shapes[2 * j + 1][0]
		for k in range(weight_row):
			for l in range(weight_col):
				param_names.append(('p%i' %(i+1), 'w_{%i,%i,%i}' %(j+layer_offset, k+1, l+1)))
				i += 1
		for k in range(bias):
			param_names.append(('p%i' %(i+1), 'b_{%i,%i}' %(j+layer_offset, k+1)))
			i += 1
	return param_names

def get_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr):
	"""
	see calc_num_weights3 for relevant of suffix
	probably a way to code this function with less conditionals, but not work
	spending any more time on
	"""
	weight_shapes = tools.get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	param_names = []
	i = 0
	j = 0
	weight_offset = 1
	bias_offset = 1
	if np.all(~np.array(m_trainable_arr)) and np.all(~np.array(b_trainable_arr)): 
		print "no trainable parameters"
		return param_names
	for k in range(len(m_trainable_arr)):
		if m_trainable_arr[k] and b_trainable_arr[k]:
			weight_row = weight_shapes[j][0]
			weight_col = weight_shapes[j][1]
			bias = weight_shapes[j + 1][0]
			j += 2
		elif m_trainable_arr[k] and not b_trainable_arr[k]:
			weight_row = weight_shapes[j][0]
			weight_col = weight_shapes[j][1]
			bias = 0
			j += 1
		elif not m_trainable_arr[k] and b_trainable_arr[k]:
			weight_row = 0
			weight_col = 0
			bias = weight_shapes[j][0]
			j += 1
		else:
			weight_row = 0
			weight_col = 0
			bias = 0
		for l in range(weight_row):
			for m in range(weight_col):
				param_names.append(('p%i' %(i+1), 'w_{%i,%i,%i}' %(k+weight_offset, l+1, m+1)))
				i += 1
		for l in range(bias):
			param_names.append(('p%i' %(i+1), 'b_{%i,%i}' %(k+bias_offset, l+1)))
			i += 1
	return param_names

def get_param_names_sh(num_inputs, layer_sizes, num_outputs, weight_shapes, granularity = None):
	"""
	writes list of param_names for weight array which corresponds to (row-wise)
	weight matrices and bias vector.
	Length of weight_shapes should be even
	"""
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	param_names = []
	i = 0
	if granularity == 'single':
		param_names.append(('p1', 'h_{1}'))
		i += 1
	elif granularity == 'layer':
		for _ in range(len(weight_shapes) / 2):
			param_names.append(('p%i' %(i+1), 'h_{%i}' %(i+1)))
			i += 1
	elif granularity == 'input_size':
		for w in range(len(weight_shapes) / 2):
			for n in range(weight_shape[2 * w][0] + 1): #+1 is for bias hyperparam term
				param_names.append(('p%i' %(i+1), 'h_{%i, %i}' %(w+1, n+1)))
				i += 1
	else: #no stochastic prior hyperparams
		pass
	for j in range(len(weight_shapes) / 2):
		weight_row = weight_shapes[2 * j][0]
		weight_col = weight_shapes[2 * j][1]
		bias = weight_shapes[2 * j + 1][0]
		for k in range(weight_row):
			for l in range(weight_col):
				param_names.append(('p%i' %(i+1), 'w_{%i,%i,%i}' %(j+1, k+1, l+1)))
				i += 1
		for k in range(bias):
			param_names.append(('p%i' %(i+1), 'b_{%i,%i}' %(j+1, k+1)))
			i += 1
	return param_names

def make_param_names(num_inputs, layer_sizes, num_outputs, base_dir, file_root):
	"""
	make .paramnames file from scratch, with nn architecture params and file paths.
	note file is saved to base_dir/file_root.paramnames
	"""
	param_names = get_param_names(num_inputs, layer_sizes, num_outputs)
	param_f = base_dir + file_root + '.paramnames'
	make_param_names_file(param_names, param_f)

def make_param_names2(num_inputs, layer_sizes, num_outputs, trainable_arr, base_dir, file_root):
	"""
	see calc_num_weights2 for relevant of suffix	
	"""
	param_names = get_param_names2(num_inputs, layer_sizes, num_outputs, trainable_arr)
	param_f = base_dir + file_root + '.paramnames'
	make_param_names_file(param_names, param_f)

def make_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr, base_dir, file_root):
	"""
	see calc_num_weights3 for relevant of suffix
	"""
	param_names = get_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	param_f = base_dir + file_root + '.paramnames'
	make_param_names_file(param_names, param_f)

def make_param_names_sh(num_inputs, layer_sizes, num_outputs, weight_shapes, granularity = None, base_dir, file_root):
	"""
	see calc_num_weights3 for relevant of suffix
	"""
	param_names = get_param_names_sh(num_inputs, layer_sizes, num_outputs, weight_shapes, granularity)
	param_f = base_dir + file_root + '.paramnames'
	make_param_names_file(param_names, param_f)


def make_param_names_pc(num_inputs, layer_sizes, num_outputs, poly_out):
	"""
	make .paramnames from nn architecture params and pc output
	"""
	param_names = get_param_names(num_inputs, layer_sizes, num_outputs, trainable_arr)
	poly_out.make_paramnames_file(param_names)

def make_param_names_pc2(num_inputs, layer_sizes, trainable_arr, num_outputs, poly_out):
	"""
	see calc_num_weights2 for relevant of suffix
	"""
	param_names = get_param_names(num_inputs, layer_sizes, num_outputs, trainable_arr)
	poly_out.make_paramnames_file(param_names)

def make_param_names_pc3(num_inputs, layer_sizes, m_trainable_arr, b_trainable_arr, num_outputs, poly_out):
	"""
	see calc_num_weights3 for relevant of suffix
	"""
	param_names = get_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	poly_out.make_paramnames_file(param_names)

def make_param_names_file(param_names, filename):
	"""
	stolen from PyPolyChord.output
	"""
	with open(filename, 'w') as f:
	    for name, latex in param_names:
	        f.write('%s   %s\n' % (name, latex))

def get_pred_param_names(num_outputs):
	"""
	get param names for pred chains file
	"""
	param_names = []
	for i in range(num_outputs):
		param_names.append(('p%i' %(i+1), 'y_{p,%i}' %(i+1)))
	return param_names

def make_pred_param_names(num_outputs, base_dir, file_root):
	"""
	make pred .paramnames file from scratch.
	note file is saved to base_dir/file_root.paramnames
	"""
	param_names = get_pred_param_names(num_outputs)
	param_f = base_dir + file_root + '.paramnames'
	make_param_names_file(param_names, param_f)

if __name__ == '__main__':
	num_inputs = 1
	layer_sizes = [3, 2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print weight_shapes
	granularity = 'input_size'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "n_stoc"
	print n_stoc
	print "granularity"
	print hyper_dependence_lengths
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes)
	print "param names"
	print get_param_names_sh(num_inputs, layer_sizes, num_outputs, weight_shapes, granularity)