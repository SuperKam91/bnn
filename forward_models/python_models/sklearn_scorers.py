#########commercial modules
import numpy as np
import sklearn.metrics

#in-house modules
import metrics as mets

#################################################################################
# following only work for scores where .predict() should be used
# to generate y_pred (as opposed to e.g. .predict_proba())
#
# n.b. scores should be maximised, so losses are negated to get scoring equivs
#################################################################################


def squared_error_score(estimator, x_true, y_true):
	y_pred = estimator.predict(x_true)
	m_times_n_out = np.prod(y_pred.shape)
	return -1. * m_times_n_out * sklearn.metrics.mean_squared_error(y_true, y_pred)

def twenty_one_cm_rmse_score(estimator, x_true, y_true, discretise = False):
	"""
	see metrics.py for loss equivalent
	"""
	y_pred = estimator.predict(x_true)
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	return -1. * mets.twenty_one_cm_rmse(y_true, y_p)

def twenty_one_cm_rmse_ts_score(estimator, x_true, y_true, n_z = 136, discretise = False):
	"""
	see metrics.py for loss equivalent
	"""
	y_pred = estimator.predict(x_true)
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	return -1. * mets.twenty_one_cm_rmse_ts(y_true, y_p, n_z)

def twenty_one_cm_rmse_ts_mean_score(estimator, x_true, y_true, n_z = 136, discretise = False):
	"""
	see metrics.py for loss equivalent
	"""
	y_pred = estimator.predict(x_true)
	return -1. * mets.twenty_one_cm_rmse_ts_mean(y_true, y_pred, n_z, discretise)

def twenty_one_cm_rmse_higher_order_score_2(mean, var):
	"""
	see metrics.py for loss equivalent
	"""
	met_ho = mets.twenty_one_cm_rmse_higher_order(mean, var)
	def twenty_one_cm_rmse(estimator, x_true, y_true):
		y_pred = estimator.predict(x_true)
		return -1. * met_ho(y_true, y_pred)
	return twenty_one_cm_rmse

def twenty_one_cm_rmse_ts_higher_order_score(mean, var, n_z):
	"""
	see metrics.py for loss equivalent
	"""
	met_ho = mets.twenty_one_cm_rmse_ts_higher_order(mean, var, n_z)
	def twenty_one_cm_rmse_ts(estimator, x_true, y_true):	
		y_pred = estimator.predict(x_true)
		return -1. * met_ho(y_true, y_pred)
	return twenty_one_cm_rmse_ts

def twenty_one_cm_rmse_ts_mean_higher_order_score(y_true, y_pred, mean, var, n_z):
	"""
	see metrics.py for loss equivalent
	"""
	met_ho = mets.twenty_one_cm_rmse_ts_mean_higher_order(mean, var, n_z)
	def twenty_one_cm_rmse_ts_mean(estimator, x_true, y_true):	
		y_pred = estimator.predict(x_true)
		return -1. * met_ho(y_true, y_pred)
	return twenty_one_cm_rmse_ts_mean




