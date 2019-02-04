#########commercial modules
import numpy as np
import scipy.stats

##############in-house modules
import inverse_priors as ip

######one-dimensional prior distributions

class inverse_stoc_hyper_prior:
	def __init__(self, hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims):
		"""
		class where variance/scale/precision/second hyperparameters of parameter priors are all stochastic.
		not really suitable for uniform priors, as only second limit can be stochastic (though these will probably never be used anyway).
		sort of works for log uniform priors, but again, only second limit is stochastic 
		(presumably first is ~0 and it is sensible for it to stay at this value)
		n.b. if you want first hyperparam to behave non-stochastically, assign them delta priors on the desired value.
		although not the most efficient way, the stoch and deterministic hyperparams for the param priors are stored in arrays of size
		n_dims for ease of implementation. but note this is not the case for the hyperprior hyperparams
		hyperparam_lengths dictates which parameters share stochastic hyperparameters, but limits sharing to
		contiguous parameters.
		level of granularity of hyperparams is either one hyperparam for whole nn, one hyperparam per layer, or one hyperparam per number of inputs to each layer.
		hyper_dependence_lengths specifies how many (contiguous) parameters share the same hyperparameters. If it has length one, assumes that all parameters share the same single stochastic and deterministic hyperparam.
		hyper_dependence_lengths should be same size as param_hyperprior_types. The latter should (similar to what param_prior_types does) index hyperprior_types and hyperprior_params, which should be of the same length as each other.
		length of hyper_dependence_lengths should equal n_stoc, sum of hyper_dependence_lengths should equal n_dims if same hyper isn't being used for whole nn.
		way stoch hyperparams are sampled is different to way params are sampled. for a given hyperparam_dependence length one value of the hyperparam is sampled. this is then added to the hyperprior_params hyperparam_dependence length times. similarly, the deterministic hyperparam is 
		"""
		assert (len(hyperprior_types) == len(hyperprior_params)), "length of hyperprior_types and hyperprior_params should be same"
		assert (len(prior_types) == len(prior_hyperparams)), "prior_types and prior_hyperparams should be the same length"
		assert (len(hyper_dependence_lengths) == len(param_hyperprior_types)), "length of hyper_dependence_lengths should be same as length of param_hyperprior_types"
		assert (len(dependence_lengths) == len(param_prior_types)), "dependence_lengths and param_prior_types should be the same length"
		assert (len(hyper_dependence_lengths) == n_stoc), "length of hyper_dependence_lengths should equal n_stoc"
		if len(dependence_lengths) != 1:
			assert (np.sum(dependence_lengths) == n_dims), "in case of dependent parameters, sum of dependence_lengths should equal n_dims"
		if len(hyper_dependence_lengths) != 1:
			assert (np.sum(hyper_dependence_lengths) == n_dims), "in case of not using single hyperparam, sum of hyper_dependence_lengths should equal n_dims"
		self.hyperprior_types = hyperprior_types
		self.prior_types = prior_types
		self.hyperprior_params = hyperprior_params
		self.prior_hyperparams = prior_hyperparams
		self.n_stoc = n_stoc #number of stochastic hyperparameters
		self.n_dims = n_dims
		self.hyperprior_ppfs = []
		self.prior_ppfs = []
		self.hyperparams = np.zeros(n_dims)
		self.stoc_hyperparams = np.zeros(n_dims)
		self.params = np.zeros(n_dims)
		self.get_hyperprior_ppf_objs()
		self.get_ppf_objs()
		#just one stochastic hyperparameter
		if (len(hyper_dependence_lengths) == 1):
			self.hyperparams.fill(prior_hyperparams[0])
			self.hyperprior_call = self.hyperprior_call_ind 
		else:
			self.param_hyperprior_types = param_hyperprior_types
			self.hyper_dependence_lengths = hyper_dependence_lengths
			self.hyperprior_call = self.hyperprior_call_by_hyper_dependence_lengths
		#just one deterministic hyperparameter
		if (len(prior_hyperparams) == 1):
			self.hyperparams.fill(prior_hyperparams[0])
		#several deterministic hyperparameters
		else:
			self.dependence_lengths = dependence_lengths
			self.param_prior_types = param_prior_types
			self.fill_det_hypers()
		#all parameters are independent and have same prior
		if  (len(param_prior_types) == 1) and (len(dependence_lengths) == 1):
			self.prior_call = self.prior_call_ind_same
		#assumes dependent variables and ones which share same stochastic hyperparams are contiguous in the param/hypercube array
		else:
			self.prior_call = self.prior_call_by_dependence_lengths

	def fill_det_hypers(self):
		"""
		fills hyperparams array with the deterministic value
		as indicated by prior_hyperparams, dependence_lengths and param_prior_types.
		note this doesn't depend on hyper_dependence_lengths, as some params may depend on
		different deterministic hyperparams but the same stochastic ones, or vice versa.
		note this function isn't used in case of one stochastic hyperparameter, in which case,
		only a single deterministic hyperparam is considered as well.
		"""
		start_index = 0
		for i, dependence_length in enumerate(self.dependence_lengths):
			self.hyperparams[start_index: start_index + dependence_length] = self.prior_hyperparams[self.param_prior_types[i]]
			start_index += dependence_length

	def get_hyperprior_ppf_objs(self):
		"""
		same as get_ppf_objs in inverse_prior class, but
		for hyper param priors
		"""
		for i, p_type in enumerate(self.hyperprior_types):
			hyperprior_hyperparam1 = self.hyperprior_params[i][0]
			hyperprior_hyperparam2 = self.hyperprior_params[i][1]
			if p_type == 0:
				self.hyperprior_ppfs.append(ip.uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 1:
				self.hyperprior_ppfs.append(ip.pos_log_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 2:
				self.hyperprior_ppfs.append(ip.neg_log_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 3:
				self.hyperprior_ppfs.append(ip.log_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 4:
				self.hyperprior_ppfs.append(ip.gaussian_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 5:
				self.hyperprior_ppfs.append(ip.laplace_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 6:
				self.hyperprior_ppfs.append(ip.cauchy_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 7:
				self.hyperprior_ppfs.append(ip.delta_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 8:
				self.hyperprior_ppfs.append(ip.gamma_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 9:
				self.hyperprior_ppfs.append(ip.sqrt_recip_gamma_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 10:
				self.hyperprior_ppfs.append(ip.sorted_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 11:
				self.hyperprior_ppfs.append(ip.sorted_pos_log_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 12:
				self.hyperprior_ppfs.append(ip.sorted_neg_log_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 13:
				self.hyperprior_ppfs.append(ip.sorted_log_uniform_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 14:
				self.hyperprior_ppfs.append(ip.sorted_gaussian_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 15:
				self.hyperprior_ppfs.append(ip.sorted_laplace_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 16:
				self.hyperprior_ppfs.append(ip.sorted_cauchy_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 17:
				self.hyperprior_ppfs.append(ip.sorted_delta_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 18:
				self.hyperprior_ppfs.append(ip.sorted_gamma_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
			elif p_type == 19:
				self.hyperprior_ppfs.append(ip.sorted_sqrt_rec_gam_prior(hyperprior_hyperparam1, hyperprior_hyperparam2))
	
	def get_ppf_objs(self):
		for p_type in self.prior_types:
			if p_type == 0:
				self.prior_ppfs.append(uniform_prior())
			elif p_type == 1:
				self.prior_ppfs.append(pos_log_uniform_prior())
			elif p_type == 2:
				self.prior_ppfs.append(neg_log_uniform_prior())
			elif p_type == 3:
				self.prior_ppfs.append(log_uniform_prior())
			elif p_type == 4:
				self.prior_ppfs.append(gaussian_prior())
			elif p_type == 5:
				self.prior_ppfs.append(laplace_prior())
			elif p_type == 6:
				self.prior_ppfs.append(cauchy_prior())
			elif p_type == 7:
				self.prior_ppfs.append(delta_prior())
			elif p_type == 8:
				self.prior_ppfs.append(gamma_prior())
			elif p_type == 9:
				self.prior_ppfs.append(sqrt_recip_gamma_prior())
			elif p_type == 10:
				self.prior_ppfs.append(sorted_uniform_prior())
			elif p_type == 11:
				self.prior_ppfs.append(sorted_pos_log_uniform_prior())
			elif p_type == 12:
				self.prior_ppfs.append(sorted_neg_log_uniform_prior())
			elif p_type == 13:
				self.prior_ppfs.append(sorted_log_uniform_prior())
			elif p_type == 14:
				self.prior_ppfs.append(sorted_gaussian_prior())
			elif p_type == 15:
				self.prior_ppfs.append(sorted_laplace_prior())
			elif p_type == 16:
				self.prior_ppfs.append(sorted_cauchy_prior())
			elif p_type == 17:
				self.prior_ppfs.append(sorted_delta_prior())
			elif p_type == 18:
				self.prior_ppfs.append(sorted_gamma_prior())
			elif p_type == 19:
				self.prior_ppfs.append(sorted_sqrt_rec_gam_prior())

	def __call__(self, hypercube):
		self.hyperprior_call(hypercube[:self.n_stoc])
		print "det hyperparams"
		print self.hyperparams
		print "stoc hyperparams"
		print self.stoc_hyperparams
		return self.prior_call(hypercube[self.n_stoc:])

	def hyperprior_call_ind(self, hypercube):
		"""
		n.b. even though one hyper param used, stoc_hyperparams has
		length same as dimensions of params (for implementation efficiency)
		""" 
		self.stoc_hyperparams.fill(self.hyperprior_ppfs[0](hypercube)[0])

	def hyperprior_call_by_hyper_dependence_lengths(self, hypercube):
		start_ind = 0
		for i, dependence_length in enumerate(self.hyper_dependence_lengths):
			hyperparam_sample = self.hyperprior_ppfs[self.param_hyperprior_types[i]](hypercube[i])
			self.stoc_hyperparams[start_ind:start_ind + dependence_length].fill(hyperparam_sample)
			start_ind += dependence_length

	def prior_call_ind_same(self, hypercube):
		self.params[:] = self.prior_ppfs[0](hypercube, self.hyperparams, self.stoc_hyperparams)
		print "params"
		print self.params
		return self.params

	def prior_call_by_par_type(self, hypercube):
		#TODO
		#will have to iterate through prior_param_types and count consecutive values which are same,
		#save these to a list and use them to index hypercube to specify which slices go to which function
		#(so vectorisation on each slice can be used)
		#or could count total number of params that uses each function, create copy of each of these blocks
		#and pass each to function (more vectorisation at cost of copy)
		pass

	def prior_call_by_dependence_lengths(self, hypercube):
		"""
		could be vectorised more for further efficiency in certain situations
		e.g. if all dependent sets of variables have same prior_param_type and same dependence_length (unlikely), 
		reshape hypercube to (m/dependence_length, dependence_length) and evaluate array in one go
		rather than iterating with for loop
		"""
		start_ind = 0
		for i, dependence_length in enumerate(self.dependence_lengths):
			self.params[start_ind:start_ind + dependence_length] = self.prior_ppfs[self.param_prior_types[i]](hypercube[start_ind:start_ind + dependence_length], self.hyperparams[start_ind:start_ind + dependence_length], self.stoc_hyperparams[start_ind:start_ind + dependence_length])
			start_ind += dependence_length
		print "params"
		print self.params
		return self.params

class base_prior:
	"""
	assumes scipy object takes two hyperparams 1 and 2,
	will need to override this if it requires more/less,
	or if requires anything other than the
	second and third default arguments of ppf call 
	"""
	def __call__(self, p, hyperparam1, hyperparam2):
		return self.rv.ppf(p, hyperparam1, hyperparam2)

class uniform_prior(base_prior):
	"""
	uniform on [a,b]
	"""
	def __init__(self):
		self.rv = scipy.stats.uniform
	def __call__(self, p, a, b):
		return self.rv.ppf(p, a, b - a)

class pos_log_uniform_prior(base_prior):
	"""
	log-uniform on [a,b], b > a > 0

	"""
	def __init__(self):
		self.rv = scipy.stats.reciprocal

class neg_log_uniform_prior(pos_log_uniform_prior):
	"""
	log-uniform on -[b, a], b > a > 0
	"""
	def __call__(self, p, a, b):
		return -1. * pos_log_uniform_prior.__call__(self, p, a, b)

class log_uniform_prior(pos_log_uniform_prior, neg_log_uniform_prior):
	"""
	assumes positive and negative parts of function
	are symmetric (i.e. interval is [-b, -a] union [a, b], b > a > 0 
	"""
	def __init__(self):
		self.nrv = neg_log_uniform_prior.__init__(self)
		self.prv = pos_log_uniform_prior.__init__(self)

	def __call__(self, p, a, b):
		params = np.zeros_like(p)
		neg_indices = p < 0.5
		params[neg_indices] = neg_log_uniform_prior.__call__(self, (0.5 - p[neg_indices]) * 2., a, b)
		params[~neg_indices] = pos_log_uniform_prior.__call__(self, (p[~neg_indices] - 0.5) * 2., a, b)
		return params

class gaussian_prior(base_prior):
	def __init__(self):
		self.rv = scipy.stats.norm

class laplace_prior(base_prior):
	def __init__(self):
		self.rv = scipy.stats.laplace

class cauchy_prior(base_prior):
	def __init__(self):
		self.rv = scipy.stats.cauchy

class delta_prior(base_prior):
	"""
	doesn't need to inherit from base_prior,
	and doesn't need sigma argument in init,
	but have been included for consistency.
	"""
	def __init__(self):
		"""
		nothing needed upon construction here
		"""
		pass

	def __call__(self, p, value, sigma = 0.):
		vals = np.zeros_like(p)
		try:
			#checks if value is a list/array etc
			getattr(value, '__getitem__')
			if len(value) == len(p):
				return value
			else:
				vals.fill(value[0])
		except AttributeError:
			vals.fill(value)
		return vals

class gamma_prior(base_prior):
	def __init__(self):
		"""
		n.b. a and b are shape and scale parameters respectively
		with a dictating how bell shaped the curve is (low a is exponential-like, high a is more bell shaped)
		and b inversely proportional to spread.
		1 / b is used as scipy.stats scale argument.
		mu value is not given, and is assumed to be zero
		"""
		self.rv = scipy.stats.gamma

	def __call__(self, p, a, b):
		return self.rv.ppf(p, a = a, scale = 1. / b)

class sqrt_recip_gamma_prior(gamma_prior):
	"""
	return sqrt(1 / sample) from gamma distribution, which can be used as 
	standard deviation for conjugate distribution
	"""
	def __call__(self, p, a, b):
		return np.sqrt(1. / gamma_prior.__call__(self, p, a, b))

#FOLLOWING ARE REPLICAS OF THOSE FOUND IN INVERSE_PRIORS.PY
#INCLUDED AGAIN BY MISTAKE REALLY (BUT IF REMOVED FUNCTIONS IN THIS FILE RELYING
#ON THEM WILL NEED TO BE CALLED FROM INVERSE_PRIORS.PY INSTEAD
#-------------------------------------------------------------------------

def forced_identifiability_transform(p):
	"""
	don't think this can be vectorised in python,
	see failed attempts below
	"""
	n = len(p)
	t = np.zeros(n)
	t[-1] = p[-1]**(1. / n)
	for i in range(n - 2, -1, -1):
	    t[i] = p[i]**(1. / (i + 1)) * t[i + 1]
	return t

def forced_identifiability_transform2(p):   
	"""
	first attempt at vectorising.
	all but last two elements are (incorrectly) evaluated to be 0.
	this is because of dependence of t[i] on t[i+1], upon evaluation of
	t[i], t[i+1] still evaluates to zero (as t is initialised with zeros).
	initially thought this may be solved by ensuring each t[i] only depends
	on t[i-1] (see below).
	"""             
	n = len(p)
	i = np.arange(1,n)         
	t = np.zeros(n)
	t[-1] = p[-1]**(1. / n)
	t[:-1] = p[:-1]**(1. / i) * t[1:]   
	return t

def forced_identifiability_transform3(p):
	"""
	second attempt at vectorising. 
	tried swapping order in which t is calculated
	(so each element of t only depends on previous element).
	doesn't work.
	guess this shows vectorisation doesn't act sequentially along array,
	and each component assumes old value of rest of array until full calculation is done.
	"""
	n = len(p)
	i = np.arange(n - 1, 0, -1)
	t = np.zeros(n)
	t[0] = p[-1]**(1. / n)
	t[1:] = p[-2::-1]**(1. / i) * t[:-1]
	return t

#--------------------------------------------------------------------------------------

class sorted_uniform_prior(uniform_prior):
	def __call__(self, p, a, b):
		t = forced_identifiability_transform(p)
		return uniform_prior.__call__(self, t,  a, b)

class sorted_pos_log_uniform_prior(pos_log_uniform_prior):
	def __call__(self, p, a, b):
		t = forced_identifiability_transform(p)
		return pos_log_uniform_prior.__call__(self, t, a, b)

class sorted_neg_log_uniform_prior(neg_log_uniform_prior):
	def __call__(self, p, a, b):
		t = forced_identifiability_transform(p)
		return neg_log_uniform_prior.__call__(self, t, a, b)

class sorted_log_uniform_prior(log_uniform_prior):
	def __call__(self, p, a, b):
		t = forced_identifiability_transform(p)
		return log_uniform_prior.__call__(self, t, a, b)

class sorted_gaussian_prior(gaussian_prior):
	def __call__(self, p, mu, sigma):
		t = forced_identifiability_transform(p)
		return gaussian_prior.__call__(self, t, mu, sigma)

class sorted_laplace_prior(laplace_prior):
	def __call__(self, p, mu, sigma):
		t = forced_identifiability_transform(p)
		return laplace_prior.__call__(self, t, mu, sigma)

class sorted_cauchy_prior(cauchy_prior):
	def __call__(self, p, mu, sigma):
		t = forced_identifiability_transform(p)
		return cauchy_prior.__call__(self, t, mu, sigma)

class sorted_delta_prior(delta_prior):
	def __call__(self, p, value, sigma):
		t = forced_identifiability_transform(p)
		return delta_prior.__call__(self, t, value, sigma)

class sorted_gamma_prior(gamma_prior):
	def __call__(self, p, a, b):
		t = forced_identifiability_transform(p)
		return gamma_prior.__call__(self, t, a, b)

class sorted_sqrt_rec_gam_prior(sqrt_recip_gamma_prior):
	def __call__(self, p, a, b):
		t = forced_identifiability_transform(p)
		return sqrt_recip_gamma_prior.__call__(self, t, a, b)
