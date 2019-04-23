#########commercial modules
import tensorflow as tf

def sum_of_squared_errors(y_true, y_pred):
	"""
	sums squared errors, rather than averaging over m,n.
	more consistent with 'typical' gaussian likelihood
	"""
	return tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true))

def gauss_loss_higher_order(var = 1.):
	"""
	higher order function required to feed m variable to loss function, without
	it taking it explicitly as an argument (keras losses can only have y's as args)
	"""
	def gauss_loss(y_true, y_pred):
		"""
		loss corresponding to (negative) of exponent of gauss
		likelihood function, with scalar variance.
		"""
		return 1. / (2. * var) * sum_of_squared_errors(y_true, y_pred)
	return gauss_loss

def sum_of_categorical_crossentropy_higher_order(m):
	"""
	higher order function required to feed m variable to loss function, without
	it taking it explicitly as an argument (keras losses can only have y's as args)
	WARNING, SINCE M IS FIXED DURING MODEL.COMPILE, SAME M WILL BE USED
	DURING TRAINING AND MODEL EVALUATION, I.E. IF EVALUATING ON A SET OF DATA
	WHICH ISN'T SAME SIZE AS TRAINED DATA, WON'T GIVE CORRECT ANSWER, AS VALUE OF M
	WON'T BE CORRECT.
	use non higher order function sum_of_categorical_crossentropy below instead.
	"""
	def sum_of_categorical_crossentropy(y_true, y_pred):
		"""
		doesn't average over m, unlike typical cat_crossent.
		does this by multiplying by m which is defined in higher order
		function (and thus is fixed), before keras divides by y_true.shape[0]
		somewhere. 
		unfortunately multiplying by y_true.shape[0] here explicitly doesn't work,
		as when keras uses this function, doesn't know size of y_true, apparently
		more consistent with 'typical' multinomial likelihood.
		assumes softmax has already been applied.
		"""
		return m * tf.keras.backend.categorical_crossentropy(y_true, y_pred)
	return sum_of_categorical_crossentropy

def sum_of_categorical_crossentropy(y_true, y_pred):
	"""
	doesn't average over m, unlike typical cat_crossent.
	assumes softmax has already been applied.
	"""
	return tf.keras.backend.sum(tf.keras.backend.categorical_crossentropy(y_true, y_pred))

def twenty_one_cm_rmse(y_true, y_pred):
	return tf.keras.backend.sqrt(tf.keras.backend.mean((y_true - y_pred)**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_true))	

def twenty_one_cm_rmse_ts(y_true, y_pred):
	n_z = 136
	m = tf.keras.backend.shape(y_pred).eval(session = tf.keras.backend.get_session())[0]
	errs = []
	for i in range(m / n_z - 1):
		errs.append(tf.keras.backend.sqrt(tf.keras.backend.mean((y_true[i * n_z:(i + 1) * n_z] - y_pred[i * n_z:(i + 1) * n_z])**2.)) / tf.keras.backend.max(np.abs(y_true[i * n_z:(i + 1) * n_z])))
	errs.append(tf.keras.backend.sqrt(tf.keras.backend.mean((y_true[-1 * n_z:] - y_pred[-1 * n_z:])**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_true[-1 * n_z:])))
	return tf.stack(errs)

def twenty_one_cm_rmse_ts_mean(y_true, y_pred):
	return tf.keras.backend.mean(twenty_one_cm_rmse_ts(y_true, y_pred))

def twenty_one_cm_rmse_higher_order(mean, var):
	"""
	higher-order version of twenty_one_cm_rmse which (inverse) transforms ys using
	mean and var of standard scale transformation before calculating 21cm rmse error
	"""
	#convert to tensors
	mean_t = tf.convert_to_tensor(mean, dtype = tf.float32)
	var_t = tf.convert_to_tensor(var, dtype = tf.float32)	
	def twenty_one_cm_rmse(y_true, y_pred):
		y_t = tf.keras.backend.sqrt(var_t) * y_true + mean_t
		y_p = tf.keras.backend.sqrt(var_t) * y_pred + mean_t
		return tf.keras.backend.sqrt(tf.keras.backend.mean((y_t - y_p)**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_t))	
	return twenty_one_cm_rmse

def twenty_one_cm_rmse_higher_order_ts(mean, var, n_z, m):
	"""
	higher-order version of twenty_one_cm_rmse which (inverse) transforms ys using
	mean and var of standard scale transformation before calculating 21cm rmse error
	per timeseries
	"""
	mean_t = tf.convert_to_tensor(mean, dtype = tf.float32)
	var_t = tf.convert_to_tensor(var, dtype = tf.float32)	
	def twenty_one_cm_rmse_ts(y_true, y_pred):
		y_t = tf.keras.backend.sqrt(var_t) * y_true + mean_t
		y_p = tf.keras.backend.sqrt(var_t) * y_pred + mean_t
		errs = []
		for i in range(m / n_z - 1):
			errs.append(tf.keras.backend.sqrt(tf.keras.backend.mean((y_t[i * n_z:(i + 1) * n_z] - y_p[i * n_z:(i + 1) * n_z])**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_t[i * n_z:(i + 1) * n_z])))
			errs.append(tf.keras.backend.sqrt(tf.keras.backend.mean((y_t[-1 * n_z:] - y_p[-1 * n_z:])**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_t[-1 * n_z:])))
		return tf.stack(errs)
	return twenty_one_cm_rmse_ts

def twenty_one_cm_rmse_higher_order_ts_mean(mean, var, n_z, m):
	"""
	as above but calculates mean of timeseries errors
	"""
	mean_t = tf.convert_to_tensor(mean, dtype = tf.float32)
	var_t = tf.convert_to_tensor(var, dtype = tf.float32)	
	def twenty_one_cm_rmse_ts_mean(y_true, y_pred):
		y_t = tf.keras.backend.sqrt(var_t) * y_true + mean_t
		y_p = tf.keras.backend.sqrt(var_t) * y_pred + mean_t
		errs = []
		for i in range(m / n_z - 1):
			errs.append(tf.keras.backend.sqrt(tf.keras.backend.mean((y_t[i * n_z:(i + 1) * n_z] - y_p[i * n_z:(i + 1) * n_z])**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_t[i * n_z:(i + 1) * n_z])))
			errs.append(tf.keras.backend.sqrt(tf.keras.backend.mean((y_t[-1 * n_z:] - y_p[-1 * n_z:])**2.)) / tf.keras.backend.max(tf.keras.backend.abs(y_t[-1 * n_z:])))
		return tf.keras.backend.mean(tf.stack(errs))
	return twenty_one_cm_rmse_ts_mean

if __name__ == '__main__':
	import numpy as np
	np.random.seed(1)
	y_true = np.random.normal(size = 136 * 5)
	y_pred = np.random.normal(size = 136 * 5)
	y_t_t = tf.convert_to_tensor(y_true)
	y_p_t = tf.convert_to_tensor(y_pred)
	a = twenty_one_cm_rmse_ts(y_t_t, y_p_t)
	print a.eval(session = tf.keras.backend.get_session())
	b = twenty_one_cm_rmse_ts_mean(y_t_t, y_p_t)
	print b.eval(session = tf.keras.backend.get_session())
	c = twenty_one_cm_rmse_higher_order_ts(0., 1., 136, 136 * 5)(y_t_t, y_p_t)
	print c.eval(session = tf.keras.backend.get_session())
	d = twenty_one_cm_rmse_higher_order_ts_mean(0., 1., 136, 136 * 5)(y_t_t, y_p_t)
	print d.eval(session = tf.keras.backend.get_session())


