#########commercial modules
import numpy as np

def get_x_y_tr_data(data_prefix, data_suffix):
	if not data_suffix:
		try:
			return np.genfromtxt(data_prefix + '_x.txt'), np.genfromtxt(data_prefix + '_y.txt')
		except IOError:
			return np.genfromtxt(data_prefix + '_x.csv', delimiter = ','), np.genfromtxt(data_prefix + '_y.csv', delimiter = ',')
	else:
		if 'csv' in data_suffix:
			return np.genfromtxt(data_prefix + '_x' + data_suffix, delimiter = ','), np.genfromtxt(data_prefix + '_y' + data_suffix, delimiter = ',')		
		elif 'txt' in data_suffix:
			return np.genfromtxt(data_prefix + '_x' + data_suffix), np.genfromtxt(data_prefix + '_y' + data_suffix)					

def get_weight_data(weights_file, length):
	return np.genfromtxt(weights_file, max_rows = length)

def get_chains_data(chains_file, n_stoc = 0, n_stoc_var = 0):
	"""
	returns normalised posterior weights,
	LL, and parameter values
	"""
	data = np.genfromtxt(chains_file)
	weight_norm = data[:,0].sum()
	return data[:,0] / weight_norm, -0.5 * data[:, 1], data[:, 2: 2 + n_stoc], data[:, 2 + n_stoc: 2 + n_stoc + n_stoc_var], data[:, 2 + n_stoc + n_stoc_var:], weight_norm

def get_chains_data2(chains_file):
	"""
	returns normalised posterior weights,
	and parameter values
	"""
	data = np.genfromtxt(chains_file)
	return data[:,0] / data[:,0].sum(), data[:, 2:]

def get_ev(stats_file):
	"""
	get E[log z] and std[log z] from .stats file
	"""
	with open(stats_file) as f:
		ss = f.readlines()
		l = ss[8]
		Z = float(l.split()[2])
		Z_err = float(l.split()[4])
	return Z, Z_err