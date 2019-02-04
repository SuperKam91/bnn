#commerical modules
import numpy as np

#in-house modules
import inverse_priors as ip
import tools
import inverse_stoc_hyper_priors as isp

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
	prior = ip.inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
	theta = prior(p)
	print "hypercube = "
	print p
	print "theta = "
	print theta

def prior_functions_test():
	p = np.array([0.1, 0.6, 0.8])
	prior_hyperparam1, prior_hyperparam2 = 1., 4.
	u = ip.uniform_prior(prior_hyperparam1, prior_hyperparam2)
	plu = ip.pos_log_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	nlu = ip.neg_log_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	lu = ip.log_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	g = ip.gaussian_prior(prior_hyperparam1, prior_hyperparam2)
	l = ip.laplace_prior(prior_hyperparam1, prior_hyperparam2)
	c = ip.cauchy_prior(prior_hyperparam1, prior_hyperparam2)
	d = ip.delta_prior(prior_hyperparam1, prior_hyperparam2)
	ga = ip.gamma_prior(prior_hyperparam1, prior_hyperparam2)
	srga = ip.sqrt_recip_gamma_prior(prior_hyperparam1, prior_hyperparam2)
	su = ip.sorted_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	splu = ip.sorted_pos_log_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	snlu = ip.sorted_neg_log_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	slu = ip.sorted_log_uniform_prior(prior_hyperparam1, prior_hyperparam2)
	sg = ip.sorted_gaussian_prior(prior_hyperparam1, prior_hyperparam2)
	sl = ip.sorted_laplace_prior(prior_hyperparam1, prior_hyperparam2)
	sc = ip.sorted_cauchy_prior(prior_hyperparam1, prior_hyperparam2)
	sd = ip.sorted_delta_prior(prior_hyperparam1, prior_hyperparam2)
	sga = ip.sorted_gamma_prior(prior_hyperparam1, prior_hyperparam2)
	ssrga = ip.sorted_sqrt_rec_gam_prior(prior_hyperparam1, prior_hyperparam2)
	print u(p)
	print plu(p)
	print nlu(p)
	print lu(p)
	print g(p)
	print l(p)
	print c(p)
	print d(p)
	print ga(p)
	print srga(p)
	print su(p)
	print splu(p)
	print snlu(p)
	print slu(p)
	print sg(p)
	print sl(p)
	print sc(p)
	print sd(p)
	print sga(p)
	print ssrga(p)

def stoc_prior_functions_test():
	p = np.array([0.1, 0.6, 0.8])
	prior_hyperparam1, prior_hyperparam2 = 1., 4.
	u = isp.uniform_prior()
	plu = isp.pos_log_uniform_prior()
	nlu = isp.neg_log_uniform_prior()
	lu = isp.log_uniform_prior()
	g = isp.gaussian_prior()
	l = isp.laplace_prior()
	c = isp.cauchy_prior()
	d = isp.delta_prior()
	ga = isp.gamma_prior()
	srga = isp.sqrt_recip_gamma_prior()
	su = isp.sorted_uniform_prior()
	splu = isp.sorted_pos_log_uniform_prior()
	snlu = isp.sorted_neg_log_uniform_prior()
	slu = isp.sorted_log_uniform_prior()
	sg = isp.sorted_gaussian_prior()
	sl = isp.sorted_laplace_prior()
	sc = isp.sorted_cauchy_prior()
	sd = isp.sorted_delta_prior()
	sga = isp.sorted_gamma_prior()
	ssrga = isp.sorted_sqrt_rec_gam_prior()
	print u(p, prior_hyperparam1, prior_hyperparam2)
	print plu(p, prior_hyperparam1, prior_hyperparam2)
	print nlu(p, prior_hyperparam1, prior_hyperparam2)
	print lu(p, prior_hyperparam1, prior_hyperparam2)
	print g(p, prior_hyperparam1, prior_hyperparam2)
	print l(p, prior_hyperparam1, prior_hyperparam2)
	print c(p, prior_hyperparam1, prior_hyperparam2)
	print d(p, prior_hyperparam1, prior_hyperparam2)
	print ga(p, prior_hyperparam1, prior_hyperparam2)
	print srga(p, prior_hyperparam1, prior_hyperparam2)
	print su(p, prior_hyperparam1, prior_hyperparam2)
	print splu(p, prior_hyperparam1, prior_hyperparam2)
	print snlu(p, prior_hyperparam1, prior_hyperparam2)
	print slu(p, prior_hyperparam1, prior_hyperparam2)
	print sg(p, prior_hyperparam1, prior_hyperparam2)
	print sl(p, prior_hyperparam1, prior_hyperparam2)
	print sc(p, prior_hyperparam1, prior_hyperparam2)
	print sd(p, prior_hyperparam1, prior_hyperparam2)
	print sga(p, prior_hyperparam1, prior_hyperparam2)
	print ssrga(p, prior_hyperparam1, prior_hyperparam2)

def inverse_stoc_hyper_priors_test1():
	"""
	single hyperparam, single param, gamma hyper, gauss prior
	"""
	hyperprior_types = [9]
	prior_types = [4]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0.]
	hyper_dependence_lengths = [1]
	dependence_lengths = [1]
	param_hyperprior_types = [0]
	param_prior_types = [0]
	n_stoc = 1
	n_dims = 1
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.6])
	prior(p)
	print isp.gaussian_prior()(p[1:], 0., 4.35688457)

def inverse_stoc_hyper_priors_test2():
	"""
	single hyperparam, multiple params, gamma hyper, gauss prior
	"""
	hyperprior_types = [9]
	prior_types = [4]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0.]
	hyper_dependence_lengths = [1]
	dependence_lengths = [1]
	param_hyperprior_types = [0]
	param_prior_types = [0]
	n_stoc = 1
	n_dims = 2
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.6, 0.7])
	prior(p)
	print isp.gaussian_prior()(p[1:], 0., 4.35688457)

def inverse_stoc_hyper_priors_test3():
	"""
	2 hyperparam, multiple params, gamma and delta hyper, gauss prior
	"""
	hyperprior_types = [9, 7]
	prior_types = [4]
	hyperprior_params = [[1., 2.], [1., 1.]]
	prior_hyperparams = [0.]
	hyper_dependence_lengths = [1, 1]
	dependence_lengths = [1]
	param_hyperprior_types = [0, 1]
	param_prior_types = [0]
	n_stoc = 2
	n_dims = 2
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7])
	prior(p)
	print isp.gaussian_prior()(p[-2], 0., 4.35688457)
	print isp.gaussian_prior()(p[-1], 0., 1.)

def inverse_stoc_hyper_priors_test4():
	"""
	3 hyperparam, 4 params, gamma, delta, gamma hyper, gauss, laplace prior
	"""
	hyperprior_types = [9, 7, 9]
	prior_types = [4, 5]
	hyperprior_params = [[1., 2.], [1., 1.], [1., 5.]]
	prior_hyperparams = [0., 1.]
	hyper_dependence_lengths = [1, 2, 1]
	dependence_lengths = [1, 3]
	param_hyperprior_types = [0, 1, 2]
	param_prior_types = [0, 1]
	n_stoc = 3
	n_dims = 4
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4])
	prior(p)
	v = [4.35688457, 1., 1., 2.33597589]
	print isp.gaussian_prior()(p[-4], 0., v[0])
	print isp.laplace_prior()(p[-3], 1., v[1])
	print isp.laplace_prior()(p[-2], 1., v[2])
	print isp.laplace_prior()(p[-1], 1., v[3])

def inverse_stoc_hyper_priors_test5():
	"""
	4 hyperparam, 6 params, mix of gamma, delta hyper, mix of gauss, laplace prior
	"""
	hyperprior_types = [9, 7, 9, 7]
	prior_types = [4, 5, 4]
	hyperprior_params = [[1., 2.], [1., 1.], [1., 5.], [3., 1.]]
	prior_hyperparams = [0., 1., 2.]
	hyper_dependence_lengths = [1, 3, 1, 1]
	dependence_lengths = [1, 3, 2]
	param_hyperprior_types = [1, 0, 3, 2]
	param_prior_types = [1, 2, 0]
	n_stoc = 4
	n_dims = 6
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5])
	prior(p)
	v = [1., 1.6986436, 1.6986436, 1.6986436, 3., 2.03787088]
	print isp.laplace_prior()(p[-6], 1., v[0])
	print isp.gaussian_prior()(p[-5], 2., v[1])
	print isp.gaussian_prior()(p[-4], 2., v[2])
	print isp.gaussian_prior()(p[-3], 2., v[3])
	print isp.gaussian_prior()(p[-2], 0., v[4])
	print isp.gaussian_prior()(p[-1], 0., v[5])

def inverse_stoc_hyper_priors_test6():
	"""
	real nn arch with 16 nn params, one hyperparam, single granularity
	all independent
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'single'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	hyperprior_types = [9]
	prior_types = [4]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0.]
	param_hyperprior_types = [0]
	param_prior_types = [0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9])
	prior(p)
	u = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457]
	print isp.gaussian_prior()(p[1], u[0], v[0])
	print isp.gaussian_prior()(p[2], u[1], v[1])
	print isp.gaussian_prior()(p[3], u[2], v[2])
	print isp.gaussian_prior()(p[4], u[3], v[3])
	print isp.gaussian_prior()(p[5], u[4], v[4])
	print isp.gaussian_prior()(p[6], u[5], v[5])
	print isp.gaussian_prior()(p[7], u[6], v[6])
	print isp.gaussian_prior()(p[8], u[7], v[7])
	print isp.gaussian_prior()(p[9], u[8], v[8])
	print isp.gaussian_prior()(p[10], u[9], v[9])
	print isp.gaussian_prior()(p[11], u[10], v[10])
	print isp.gaussian_prior()(p[12], u[11], v[11])
	print isp.gaussian_prior()(p[13], u[12], v[12])
	print isp.gaussian_prior()(p[14], u[13], v[13])
	print isp.gaussian_prior()(p[15], u[14], v[14])
	print isp.gaussian_prior()(p[16], u[15], v[15])
	print isp.gaussian_prior()(p[17], u[16], v[16])

def inverse_stoc_hyper_priors_test7():
	"""
	real nn arch with 16 nn params, one hyperparam, single granularity, 
	degen dependent params (2 priors)
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'single'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes)
	hyperprior_types = [9]
	prior_types = [4, 5]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0., 1.]
	param_hyperprior_types = [0]
	param_prior_types = [0, 1, 0, 1, 0, 0, 1, 1, 0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9])
	prior(p)
	u = [0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457]
	print isp.gaussian_prior()(p[1], u[0], v[0])
	print isp.gaussian_prior()(p[2], u[1], v[1])
	print isp.gaussian_prior()(p[3], u[2], v[2])
	print isp.laplace_prior()(p[4], u[3], v[3])
	print isp.laplace_prior()(p[5], u[4], v[4])
	print isp.laplace_prior()(p[6], u[5], v[5])
	print isp.gaussian_prior()(p[7], u[6], v[6])
	print isp.gaussian_prior()(p[8], u[7], v[7])
	print isp.laplace_prior()(p[9], u[8], v[8])
	print isp.laplace_prior()(p[10], u[9], v[9])
	print isp.gaussian_prior()(p[11], u[10], v[10])
	print isp.gaussian_prior()(p[12], u[11], v[11])
	print isp.gaussian_prior()(p[13], u[12], v[12])
	print isp.gaussian_prior()(p[14], u[13], v[13])
	print isp.laplace_prior()(p[15], u[14], v[14])
	print isp.laplace_prior()(p[16], u[15], v[15])
	print isp.gaussian_prior()(p[17], u[16], v[16])

def inverse_stoc_hyper_priors_test8():
	"""
	real nn arch with 16 nn params, one hyperparam, layer granularity, 
	independent params
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'layer'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	hyperprior_types = [9]
	prior_types = [4]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0.]
	param_hyperprior_types = [0, 0, 0]
	param_prior_types = [0]
	print "n_dims"
	print n_dims
	print "n_stoc"
	print n_stoc
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7])
	prior(p)
	u = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 4.35688457, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.47740087, 1.47740087, 1.47740087]
	print isp.gaussian_prior()(p[3], u[0], v[0])
	print isp.gaussian_prior()(p[4], u[1], v[1])
	print isp.gaussian_prior()(p[5], u[2], v[2])
	print isp.gaussian_prior()(p[6], u[3], v[3])
	print isp.gaussian_prior()(p[7], u[4], v[4])
	print isp.gaussian_prior()(p[8], u[5], v[5])
	print isp.gaussian_prior()(p[9], u[6], v[6])
	print isp.gaussian_prior()(p[10], u[7], v[7])
	print isp.gaussian_prior()(p[11], u[8], v[8])
	print isp.gaussian_prior()(p[12], u[9], v[9])
	print isp.gaussian_prior()(p[13], u[10], v[10])
	print isp.gaussian_prior()(p[14], u[11], v[11])
	print isp.gaussian_prior()(p[15], u[12], v[12])
	print isp.gaussian_prior()(p[16], u[13], v[13])
	print isp.gaussian_prior()(p[17], u[14], v[14])
	print isp.gaussian_prior()(p[18], u[15], v[15])
	print isp.gaussian_prior()(p[19], u[16], v[16])

def inverse_stoc_hyper_priors_test9():
	"""
	real nn arch with 16 nn params, two hyperparams, layer granularity, 
	independent params
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'layer'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	hyperprior_types = [9, 7]
	prior_types = [4]
	hyperprior_params = [[1., 2.], [2., 0.]]
	prior_hyperparams = [0.]
	param_hyperprior_types = [1, 0, 1]
	param_prior_types = [0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7])
	prior(p)
	u = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
	v = [2., 2., 2., 2., 2., 2., 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 2., 2., 2.]
	print isp.gaussian_prior()(p[3], u[0], v[0])
	print isp.gaussian_prior()(p[4], u[1], v[1])
	print isp.gaussian_prior()(p[5], u[2], v[2])
	print isp.gaussian_prior()(p[6], u[3], v[3])
	print isp.gaussian_prior()(p[7], u[4], v[4])
	print isp.gaussian_prior()(p[8], u[5], v[5])
	print isp.gaussian_prior()(p[9], u[6], v[6])
	print isp.gaussian_prior()(p[10], u[7], v[7])
	print isp.gaussian_prior()(p[11], u[8], v[8])
	print isp.gaussian_prior()(p[12], u[9], v[9])
	print isp.gaussian_prior()(p[13], u[10], v[10])
	print isp.gaussian_prior()(p[14], u[11], v[11])
	print isp.gaussian_prior()(p[15], u[12], v[12])
	print isp.gaussian_prior()(p[16], u[13], v[13])
	print isp.gaussian_prior()(p[17], u[14], v[14])
	print isp.gaussian_prior()(p[18], u[15], v[15])
	print isp.gaussian_prior()(p[19], u[16], v[16])

def inverse_stoc_hyper_priors_test10():
	"""
	real nn arch with 16 nn params, two hyperparams, layer granularity, 
	degen dependent params (2 priors)
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'layer'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes)
	hyperprior_types = [9, 7]
	prior_types = [4, 5]
	hyperprior_params = [[1., 2.], [2., 0.]]
	prior_hyperparams = [0., 1.]
	param_hyperprior_types = [1, 0, 1]
	param_prior_types = [0, 1, 0, 1, 0, 0, 1, 1, 0]
	print "n_dims"
	print n_dims
	print "n_stoc"
	print n_stoc
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7])
	prior(p)
	u = [0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0.]
	v = [2., 2., 2., 2., 2., 2., 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 1.6986436, 2., 2., 2.]
	print isp.gaussian_prior()(p[3], u[0], v[0])
	print isp.gaussian_prior()(p[4], u[1], v[1])
	print isp.gaussian_prior()(p[5], u[2], v[2])
	print isp.laplace_prior()(p[6], u[3], v[3])
	print isp.laplace_prior()(p[7], u[4], v[4])
	print isp.laplace_prior()(p[8], u[5], v[5])
	print isp.gaussian_prior()(p[9], u[6], v[6])
	print isp.gaussian_prior()(p[10], u[7], v[7])
	print isp.laplace_prior()(p[11], u[8], v[8])
	print isp.laplace_prior()(p[12], u[9], v[9])
	print isp.gaussian_prior()(p[13], u[10], v[10])
	print isp.gaussian_prior()(p[14], u[11], v[11])
	print isp.gaussian_prior()(p[15], u[12], v[12])
	print isp.gaussian_prior()(p[16], u[13], v[13])
	print isp.laplace_prior()(p[17], u[14], v[14])
	print isp.laplace_prior()(p[18], u[15], v[15])
	print isp.gaussian_prior()(p[19], u[16], v[16])

def inverse_stoc_hyper_priors_test11():
	"""
	real nn arch with 16 nn params, one hyperparam, input_size granularity, 
	independent params
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'input_size'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "granularity"
	print tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	hyperprior_types = [9]
	prior_types = [4]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0.]
	param_hyperprior_types = [0, 0, 0, 0, 0, 0, 0, 0, 0] 
	param_prior_types = [0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8])
	prior(p)
	u = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 1.6986436, 1.6986436, 1.6986436, 1.47740087, 1.47740087, 1.28886271, 1.28886271, 1.11475103, 1.11475103, 0.9319812, 0.9319812, 1.97869411, 2.9938003, 4.35688457]
	print isp.gaussian_prior()(p[9], u[0], v[0])
	print isp.gaussian_prior()(p[10], u[1], v[1])
	print isp.gaussian_prior()(p[11], u[2], v[2])
	print isp.gaussian_prior()(p[12], u[3], v[3])
	print isp.gaussian_prior()(p[13], u[4], v[4])
	print isp.gaussian_prior()(p[14], u[5], v[5])
	print isp.gaussian_prior()(p[15], u[6], v[6])
	print isp.gaussian_prior()(p[16], u[7], v[7])
	print isp.gaussian_prior()(p[17], u[8], v[8])
	print isp.gaussian_prior()(p[18], u[9], v[9])
	print isp.gaussian_prior()(p[19], u[10], v[10])
	print isp.gaussian_prior()(p[20], u[11], v[11])
	print isp.gaussian_prior()(p[21], u[12], v[12])
	print isp.gaussian_prior()(p[22], u[13], v[13])
	print isp.gaussian_prior()(p[23], u[14], v[14])
	print isp.gaussian_prior()(p[24], u[15], v[15])
	print isp.gaussian_prior()(p[25], u[16], v[16])

def inverse_stoc_hyper_priors_test12():
	"""
	real nn arch with 16 nn params, two hyperparams, input_size granularity, 
	independent params
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'input_size'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "granularity"
	print tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes, independent = True)
	hyperprior_types = [9, 7]
	prior_types = [4]
	hyperprior_params = [[1., 2.], [2., 0]]
	prior_hyperparams = [0.]
	param_hyperprior_types = [0, 1, 0, 0, 1, 1, 1, 0, 0] 
	param_prior_types = [0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8])
	prior(p)
	u = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 2., 2., 2., 1.47740087, 1.47740087, 1.28886271, 1.28886271, 2., 2., 2., 2., 2., 2.9938003, 4.35688457]
	print isp.gaussian_prior()(p[9], u[0], v[0])
	print isp.gaussian_prior()(p[10], u[1], v[1])
	print isp.gaussian_prior()(p[11], u[2], v[2])
	print isp.gaussian_prior()(p[12], u[3], v[3])
	print isp.gaussian_prior()(p[13], u[4], v[4])
	print isp.gaussian_prior()(p[14], u[5], v[5])
	print isp.gaussian_prior()(p[15], u[6], v[6])
	print isp.gaussian_prior()(p[16], u[7], v[7])
	print isp.gaussian_prior()(p[17], u[8], v[8])
	print isp.gaussian_prior()(p[18], u[9], v[9])
	print isp.gaussian_prior()(p[19], u[10], v[10])
	print isp.gaussian_prior()(p[20], u[11], v[11])
	print isp.gaussian_prior()(p[21], u[12], v[12])
	print isp.gaussian_prior()(p[22], u[13], v[13])
	print isp.gaussian_prior()(p[23], u[14], v[14])
	print isp.gaussian_prior()(p[24], u[15], v[15])
	print isp.gaussian_prior()(p[25], u[16], v[16])

def inverse_stoc_hyper_priors_test13():
	"""
	real nn arch with 16 nn params, one hyperparam, input_size granularity, 
	degen dependent params (2 priors)
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'input_size'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "granularity"
	print tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes)
	hyperprior_types = [9]
	prior_types = [4, 5]
	hyperprior_params = [[1., 2.]]
	prior_hyperparams = [0., 1.]
	param_hyperprior_types = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	param_prior_types = [0, 1, 0, 1, 0, 0, 1, 1, 0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8])
	prior(p)
	u = [0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 1.6986436, 1.6986436, 1.6986436, 1.47740087, 1.47740087, 1.28886271, 1.28886271, 1.11475103, 1.11475103, 0.9319812, 0.9319812, 1.97869411, 2.9938003, 4.35688457]
	print isp.gaussian_prior()(p[9], u[0], v[0])
	print isp.gaussian_prior()(p[10], u[1], v[1])
	print isp.gaussian_prior()(p[11], u[2], v[2])
	print isp.laplace_prior()(p[12], u[3], v[3])
	print isp.laplace_prior()(p[13], u[4], v[4])
	print isp.laplace_prior()(p[14], u[5], v[5])
	print isp.gaussian_prior()(p[15], u[6], v[6])
	print isp.gaussian_prior()(p[16], u[7], v[7])
	print isp.laplace_prior()(p[17], u[8], v[8])
	print isp.laplace_prior()(p[18], u[9], v[9])
	print isp.gaussian_prior()(p[19], u[10], v[10])
	print isp.gaussian_prior()(p[20], u[11], v[11])
	print isp.gaussian_prior()(p[21], u[12], v[12])
	print isp.gaussian_prior()(p[22], u[13], v[13])
	print isp.laplace_prior()(p[23], u[14], v[14])
	print isp.laplace_prior()(p[24], u[15], v[15])
	print isp.gaussian_prior()(p[25], u[16], v[16])

def inverse_stoc_hyper_priors_test14():
	"""
	real nn arch with 16 nn params, two hyperparams, input_size granularity, 
	degen dependent params
	"""
	num_inputs = 1
	layer_sizes = [3,2]
	num_outputs = 1
	print "num weights"
	n_dims = tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	print tools.calc_num_weights(num_inputs, layer_sizes, num_outputs)
	weight_shapes = tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	print "weight shapes"
	print tools.get_weight_shapes(num_inputs, layer_sizes, num_outputs)
	granularity = 'input_size'
	hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	n_stoc = len(hyper_dependence_lengths)
	print "granularity"
	print tools.get_hyper_dependence_lengths(weight_shapes, granularity)
	print "number of weights per layer"
	print tools.calc_num_weights_layers(weight_shapes)
	dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes)
	print "degen dependence lengths"
	print tools.get_degen_dependence_lengths(weight_shapes)
	hyperprior_types = [9, 7]
	prior_types = [4, 5]
	hyperprior_params = [[1., 2.], [2., 0.]]
	prior_hyperparams = [0., 1.]
	param_hyperprior_types = [0, 1, 0, 0, 1, 1, 1, 0, 0]
	param_prior_types = [0, 1, 0, 1, 0, 0, 1, 1, 0]
	prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
	p = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8])
	prior(p)
	u = [0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0.]
	v = [4.35688457, 4.35688457, 4.35688457, 2., 2., 2., 1.47740087, 1.47740087, 1.28886271, 1.28886271, 2., 2., 2., 2., 2., 2.9938003, 4.35688457]
	print isp.gaussian_prior()(p[9], u[0], v[0])
	print isp.gaussian_prior()(p[10], u[1], v[1])
	print isp.gaussian_prior()(p[11], u[2], v[2])
	print isp.laplace_prior()(p[12], u[3], v[3])
	print isp.laplace_prior()(p[13], u[4], v[4])
	print isp.laplace_prior()(p[14], u[5], v[5])
	print isp.gaussian_prior()(p[15], u[6], v[6])
	print isp.gaussian_prior()(p[16], u[7], v[7])
	print isp.laplace_prior()(p[17], u[8], v[8])
	print isp.laplace_prior()(p[18], u[9], v[9])
	print isp.gaussian_prior()(p[19], u[10], v[10])
	print isp.gaussian_prior()(p[20], u[11], v[11])
	print isp.gaussian_prior()(p[21], u[12], v[12])
	print isp.gaussian_prior()(p[22], u[13], v[13])
	print isp.laplace_prior()(p[23], u[14], v[14])
	print isp.laplace_prior()(p[24], u[15], v[15])
	print isp.gaussian_prior()(p[25], u[16], v[16])
