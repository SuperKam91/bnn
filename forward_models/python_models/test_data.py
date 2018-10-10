#########commercial modules
import numpy as np

def get_test_data():
	num_inputs = 2
	num_outputs = 2
	m = 3
	batch_size = 3	
	x_tr = np.arange(m * num_inputs).reshape(m, num_inputs)
	y_tr = 6 * x_tr * x_tr + 17 * x_tr + 13
	w = np.arange(27)
	return x_tr, y_tr, w