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