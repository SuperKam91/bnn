#########commercial modules
import tensorflow as tf
import numpy as np
import sklearn.metrics

#in-house modules
import tools

"""
y_true and y_pred are assumed to be in one-hot format throughout this module.
first few functions piggy back off keras implementations, while the latter use sklearn.
wrappers essentially correct format of inputs to these functions, using expected outputs from nns.
many other scoring metrics are available in sklearn.metrics, see:
https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
"""

def mean_squared_error(y_true, y_pred, discretise = False):
	"""
	requires input arrays to be same np.dtype.
	returns average, not sum of errors.
	discretising (for classification problems) makes little sense to me,
	but may be necessary in some obscure scenarios
	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	mse_a = tf.Session().run(tf.keras.losses.mean_squared_error(y_true, y_p))#
	return mse_a.mean()

def mean_absolute_error(y_true, y_pred, discretise = False):
	"""
	requires input arrays to be same np.dtype.
	returns average, not sum of errors
	discretising (for classification problems) makes little sense to me,
	but may be necessary in some obscure scenarios
	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	mae_a = tf.Session().run(tf.keras.losses.mean_absolute_error(y_true, y_p))
	return mae_a.mean()

def binary_crossentropy(y_true, y_pred, discretise = False):
	"""
	requires input arrays to be np.float64 type.
	returns average, not sum of crossentropy.
	discretising (for classification problems) makes little sense to me,
	but may be necessary in some obscure scenarios

	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	y_t = y_true
	y_true_t = tf.Variable(y_t, dtype=tf.float64)
	y_pred_t = tf.Variable(y_p, dtype=tf.float64)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		bce_a = sess.run(tf.keras.losses.binary_crossentropy(y_true_t, y_pred_t))
	return bce_a.mean()

def categorical_crossentropy(y_true, y_pred, discretise = False):	
	"""
	requires input arrays to be np.float64 type.
	returns average, not sum of crossentropy.
	discretising (for classification problems) makes little sense to me,
	but may be necessary in some obscure scenarios

	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	y_t = y_true
	y_true_t = tf.Variable(y_t, dtype=tf.float64)
	y_pred_t = tf.Variable(y_p, dtype=tf.float64)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		cce_a = sess.run(tf.keras.losses.categorical_crossentropy(y_true_t, y_pred_t))
	return cce_a.mean()

def binary_accuracy(y_true, y_pred):
	"""
	requires input arrays to be same np.dtype.
	returns average, not sum of errors.
	Accuracy does not perform well with imbalanced data sets.
	For example, if you have 95 negative and 5 positive samples, 
	classifying all as negative gives 0.95 accuracy score. 
	"""
	ba_a = tf.Session().run(tf.keras.metrics.binary_accuracy(y_true, y_pred))
	ba = ba_a.sum() / ba_a.shape[0]
	return ba

def categorical_accuracy(y_true, y_pred):
	"""
	requires input arrays to be same np.dtype.
	returns average, not sum of errors.
	"""
	ca_a = tf.Session().run(tf.keras.metrics.categorical_accuracy(y_true, y_pred))
	ca = ca_a.sum() / ca_a.shape[0]
	return ca

#the following three functions are taken and adapted from:
#https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
#NOTE THESE DO NOT WORK, DO NOT USE!!!
#probably would have been easier/more efficient to implemented these in np but oh well.
#-------------------------------------------------------------------------------------------

def precision(y_true, y_pred):
	"""Precision metric.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	y_true_t = tf.Variable(y_true, dtype=tf.float64)
	y_pred_t = tf.Variable(y_pred, dtype=tf.float64)
	true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true_t * y_pred_t, 0, 1)))
	predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred_t, 0, 1)))
	precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		p = sess.run(precision)
	return p

def recall(y_true, y_pred):
	"""Recall metric.
	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	epsilon is to prevent nan values.
	"""
	y_true_t = tf.Variable(y_true, dtype=tf.float64)
	y_pred_t = tf.Variable(y_pred, dtype=tf.float64)
	true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true_t * y_pred_t, 0, 1)))
	possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true_t, 0, 1)))
	recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		r = sess.run(recall)
	return r

def f1(y_true, y_pred):
	"""
	f1 score calculated from recall and precision. gives a measure of both
	quantities in a single number. recommended by andrew ng.
	"""
	epsilon = np.finfo(float).eps
	prec = precision(y_true, y_pred)
	rec = recall(y_true, y_pred)
	return 2 * ((prec * rec) / (prec + rec + epsilon))

#-------------------------------------------------------------------------------------------

def decode_onehot_outputs(y_true, y_pred):
	"""
	converts one-hot true ys and predicted ys (probabilities)
	into decoded vectors, for use in sklearn.metric functions.
	e.g.
	y_true = [[0,0,1], [0,1,0], [1,0,0]]
	y_pred = [[0.3,0.4,0.3], [0.1,0.5,0.4], [0.2,0.1,0.7]]
	- >
	y_true = [2,1,0]
	y_pred = [1,1,2]
	"""
	y_pred_r = tools.round_probabilities(y_pred)
	y_pred_d = tools.decode_onehot(y_pred_r)
	y_true_d = tools.decode_onehot(y_true)
	return y_pred_d, y_true_d

def precision(y_true, y_pred):
	"""
	calculates precision for each output class.
	high precision means that the classifier is very "picky",
	and is reserved about assigning a prediction to the class, thus it
	misses a lot of the instances.
	a large proportion of cases which is does attribute to the class,
	probably are the class
	"""
	if y_true.shape[1] == 2:
		average = 'binary'
	else:
		average = None
	y_pred_d, y_true_d = decode_onehot_outputs(y_true, y_pred)
	return sklearn.metrics.precision_score(y_true_d, y_pred_d, average = average)

def recall(y_true, y_pred):
	"""
	calculates recall for each output class.
	high recall means that the classifier is "generous" in
	assigning to a class, meaning it captures a lot of the instances.
	on the other hand, it also assigns a number of examples to the class,
	which don't actually belong to it.
	"""
	if y_true.shape[1] == 2:
		average = 'binary'
	else:
		average = None
	y_pred_d, y_true_d = decode_onehot_outputs(y_true, y_pred)
	return sklearn.metrics.recall_score(y_true_d, y_pred_d, average = average)

def f1(y_true, y_pred):
	"""
	calculates f1 score for each output class.
	there is a trade-off between having high precision
	and high recall. one generally cannot be "generous" and "picky"
	in assigning to a class at the same time. the f1 score gives a measure
	of the average of the precision and recall. maximising it should give an
	'optimimum' combination of the two.
	"""	
	if y_true.shape[1] == 2:
		average = 'binary'
	else:
		average = None
	y_pred_d, y_true_d = decode_onehot_outputs(y_true, y_pred)
	return sklearn.metrics.f1_score(y_true_d, y_pred_d, average = average)

def confusion_matrix(y_true, y_pred):
	"""
	see https://www.youtube.com/watch?v=FAr2GmWNbT0 for a great video
	explaining the confusion matrix for multiclasses.
	rows correspond to true classes, cols correspond to predicted
	(n.b. wikipedia does it other way round).
	for the binary case:
		top left square is true negatives (TN)
		top right square is false negatives (FN)
		bottom left square is false positives (FP)
		bottom right square is true positives (TP)
		quick way to remember these is: 
		first letter is whether prediction is correct,
		second letter is nature of prediction (0 = neg, 1 = pos)
	for multiclass case, each class has its own TP, TN, FP, FN,
	and the number of actual examples belonging to each class is the sum
	of the elements in its row. the four measures:
		values along diagonal are the TPs
		the values along a class' (true) row excluding the TP value sum to the FN
		for that class
		the number of FPs for a class is the sum of the elements along its
		(predicted) column minus the TP value
		the number of TNs for a class is the sum of the elements in the matrix
		excluding the (predicted) column and the (true) row elements corresponding
		to the class
		quick way to remember these is:
		first letter is whether prediction is 'quasi-correct' w.r.t. that class
		i.e. if true value is that class, predict that class or if true value is not
		that class, did not predict that class (but didn't necessarily predict correct class).
		complimentary cases, i.e. predicted that class but isn't that class or didn't predict 
		that class but true value is that class, are both false.
		second letter is nature of prediction (predict other class = neg, predicted class = pos)
	multiclass rules apply to binary case, as long as TN is applied before TP rule 
	(and thus diagonal is just one element, so not to double count in TN and TP).
	"""
	y_pred_d, y_true_d = decode_onehot_outputs(y_true, y_pred)
	return sklearn.metrics.confusion_matrix(y_true_d, y_pred_d)

def explained_variance_score(y_true, y_pred, discretise = False):
	"""
	not entirely sure what it is, but see
	https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
	not sure whether it makes sense to discretise
	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	y_t = y_true
	return sklearn.metrics.explained_variance_score(y_t, y_p, multioutput='uniform_average')

def regression_r2_score(y_true, y_pred, discretise = False):
	"""
	returns well-known R^2 regression coefficient.
	calculates separately for each output, then takes average over these
	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	y_t = y_true
	return sklearn.metrics.r2_score(y_t, y_p, multioutput='uniform_average')

def balanced_accuracy(y_true, y_pred):
	"""
	Balanced accuracy overcomes the problem of imbalanced data, 
	by normalising true positive and true negative predictions by 
	the number of positive and negative samples, respectively, and divides their sum into two.
	"""
	y_pred_d, y_true_d = decode_onehot_outputs(y_true, y_pred)
	return sklearn.metrics.balanced_accuracy_score(y_true_d, y_pred_d)

def twenty_one_cm_rmse(y_true, y_pred, discretise = False):
	"""
	used in 21cm paper to evaluate performance
	"""
	if discretise:
		y_p = tools.round_probabilities(y_pred)
	else:
		y_p = y_pred
	return np.sqrt(np.mean((y_true - y_pred)**2.)) / np.max(np.abs(y_pred))


