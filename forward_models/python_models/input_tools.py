#########commercial modules
import numpy as np

def get_x_y_tr_data(data_prefix):
	return np.genfromtxt(data_prefix + '_x.txt'), np.genfromtxt(data_prefix + '_y.txt')

def get_weight_data(weight_file, length):
	return np.genfromtxt(weight_file, max_rows = length)