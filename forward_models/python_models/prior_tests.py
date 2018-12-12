#commerical modules
import numpy as np

#in-house modules
import inverse_priors

def nn_prior_test(prior):
	"""
	takes prior object setup for nn calculation,
	and uses arbitrary hypercube created here to calculate
	inverse prior (parameter values).
	p should be same length as dimensionality of nn
	"""
	p = np.array([0.1, 0.9]*35)	
	theta = prior(p)
	print "hypercube = "
	print p
	print "theta = "
	print theta

def prior_test():
	"""
	setup inverse_prior object manually,
	and calculate parameter values corresponding to arbitary
	hypercube set here.
	p should be same length as num_weights
	"""
	num_weights = 2
	p = np.array([0.1, 0.9])
	prior_types = [7]
	prior_hyperparams = [[-2., 2.]]
	dependence_lengths = [2]
	param_prior_types = [0]
	prior = inverse_priors.inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
	theta = prior(p)
	print "hypercube = "
	print p
	print "theta = "
	print theta