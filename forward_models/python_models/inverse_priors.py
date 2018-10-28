#########commercial modules
import numpy as np
import scipy.stats

######one-dimensional prior distributions

class inverse_prior:
	def __init__(self, prior_types, prior_hyperparams, dependence_lengths, param_prior_types, n_dims):
		"""
		prior_types are form of prior functions, given by the following index:
		0: univariate-uniform
		1: univariate-gaussian
		TODO
		prior_hyperparams are the hyperparameters for each prior function specified
		in prior_types,
		dependence_lengths specify groups of dependent parameters in contiguous memory.
		using dependence_lengths is efficient as it means copies of parts of hypercube don't
		need to be made, and no dependence_array has to be searched to find which parameters are
		dependent on one another. 
		using a dependence_array is probably necessary if dependent variables are non-contiguous.
		param_prior_types specifies function to use for each set of dependent parameters. note each value should be the
		in the set of indices of prior_types, not the set of values that prior_types can take,
		as it is used to index self.prior_ppfs.
		dependence_lengths and param_prior_types should be the same length
		NOTE if all parameters are independent and use same prior, set dependence_lengths and
		prior_param_types to have length = 1 
		"""
		self.prior_types = prior_types
		self.prior_hyperparams = prior_hyperparams
		self.prior_ppfs = []
		#declaring here as a member variable vs declaring in function which evaluates inv prior
		#is a trade-off between having to store member variable between calls but only allocating memory once
		#vs not storing member variable but having to allocate at every call. exacts depend on polychord implementation
		#(both python and fortran part) and whether it copies param array or uses in-place
		self.params = np.zeros(n_dims)
		self.get_ppf_objs()
		#all parameters are independent and have same prior
		if (len(param_prior_types) == 1) and (len(dependence_lengths) == 1):
			self.__call__ = self.prior_call_ind_same
		#independent parameters with different priors
		#TODO
		# elif (np.all(dependence_lengths) == 1):
			# self.__call__ = self.prior_call_by_par_type()
		#assumes dependent variables are contiguous in the param/hypercube array
		else:
			self.param_prior_types = param_prior_types
			self.dependence_lengths = dependence_lengths
			self.__call__ = self.prior_call_by_dependence_lengths
		#other ways of calculating self.prior_call may need to be implemented in the future
		#either for more efficiency in certain cases, or to handle non-contiguous dependent variables
		#e.g. with a separate array specifying which parameters are dependent (dependence_array)

	def get_ppf_objs(self):
		for i, p_type in enumerate(prior_types):
			prior_hyperparam1 = self.prior_hyperparams[i][0]
			prior_hyperparam2 = self.prior_hyperparams[i][1]
			if p_type == 0:
				self.prior_ppfs.append(uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 1:
				self.prior_ppfs.append(pos_log_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 2:
				self.prior_ppfs.append(neg_log_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 3:
				self.prior_ppfs.append(log_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 4:
				self.prior_ppfs.append(gaussian_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 5:
				self.prior_ppfs.append(laplace_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 6:
				self.prior_ppfs.append(cauchy_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 7:
				self.prior_ppfs.append(sorted_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 8:
				self.prior_ppfs.append(sorted_pos_log_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 9:
				self.prior_ppfs.append(sorted_neg_log_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 10:
				self.prior_ppfs.append(sorted_log_uniform_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 11:
				self.prior_ppfs.append(sorted_gaussian_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 12:
				self.prior_ppfs.append(sorted_laplace_prior(prior_hyperparam1, prior_hyperparam2))
			elif p_type == 13:
				self.prior_ppfs.append(sorted_cauchy_prior(prior_hyperparam1, prior_hyperparam2))

	def prior_call_ind_same(self, hypercube):
		self.params[:] = self.prior_ppfs[0](hypercube)
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
		func_count = 0
		for dependence_length in self.dependence_lengths:
			self.params[start_ind:start_ind + dependence_length] = self.prior_ppfs[self.param_prior_types[func_count]](hypercube[start_ind:start_ind + dependence_length])
			start_ind += dependence_length
			func_count += 1
		return self.params

class base_prior:
	"""
	abstract class- instances of this class shouldn't be called.
	NOTE with regard to whether passing array to priors and changing them in-function,
	as opposed to setting them equal to a function return is more efficient, I probably
	need to do the same tests as the cpp case, as I think it's the same trade-off.
	if passing by reference is faster, then the only way to do this is by redefining
	the ppf method for each prior (i.e. not using scipy.stats ones) and have them all take
	an additional np array for the rvalue to be copied into in-function, 
	rather than create a return value, copy this to an rvalue outside the function,
	then copy the rvalue into the array. 
	note the reference method is used in the cpp versions.
	if hyperparameters ever become stochastic, __call__ will have to return scipy.stats.prob_func(hyper_params).ppf(p).
	so base_prior will need __init__ which will choose what __call__ points to (a method will need to be created for stochastic
	and non stochastic versions, one of which, __call__ will point to).
	derived class __init__'s will then call base __init__, and depending on hyperparam stochasticity, 
	do nothing else or set self.rv to frozen scipy.stats object/ 
	"""
	def __call__(self, p):
		return self.rv.ppf(p)

class uniform_prior(base_prior):
	"""
	uniform on [a,b]
	"""
	def __init__(self, a, b):
		self.rv = scipy.stats.uniform(a, b - a)

class pos_log_uniform_prior(base_prior):
	"""
	log-uniform on [a,b], a > b > 0

	"""
	def __init__(self, a, b):
		self.rv = scipy.stats.reciprocal(a, b)

class neg_log_uniform_prior(pos_log_uniform_prior):
	"""
	log-uniform on -[b, a], a > b > 0
	"""
	def __call__(self, p):
		return -1. * pos_log_uniform_prior.__call__(self, p)

class log_uniform_prior(pos_log_uniform_prior, neg_log_uniform_prior):
	"""
	assumes positive and negative parts of function
	are symmetric (i.e. interval is [-b, a] union [a, b], a > b > 0 
	can be easily extended to asymmetric case.
	might be able to make more efficient by defining own ppf function defined on
	positive and negative domain i.e. works with un re-normalised probabilities.
	sgn(x) function may be useful for this.
	"""
	def __init__(self, a, b):
		self.nrv = neg_log_uniform_prior.__init__(self, a, b)
		self.prv = pos_log_uniform_prior.__init__(self, a, b)

	def __call__(self, p):
		#using return np.where(p<0.5, neg_log_uniform_prior.__call__(self, (0.5 - p) * 2.), pos_log_uniform_prior.__call__(self, (p - 0.5) * 2.))
		#may be faster.
		params = np.zeros_like(p)
		neg_indices = p < 0.5
		params[neg_indices] = neg_log_uniform_prior.__call__(self, (0.5 - p[neg_indices]) * 2.)
		params[~neg_indices] = pos_log_uniform_prior.__call__(self, (p[~neg_indices] - 0.5) * 2.)
		return params

class gaussian_prior(base_prior):
	def __init__(self, mu, sigma):
		self.rv = scipy.stats.norm(mu, sigma)

class laplace_prior(base_prior):
	def __init__(self, mu, sigma):
		self.rv = scipy.stats.laplace(mu, sigma)

class cauchy_prior(base_prior):
	def __init__(self, mu, sigma):
		self.rv = scipy.stats.cauchy(mu, sigma)

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


class sorted_uniform_prior(uniform_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return uniform_prior.__call__(self, t)

class sorted_pos_log_uniform_prior(pos_log_uniform_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return pos_log_uniform_prior.__call__(self, t)

class sorted_neg_log_uniform_prior(neg_log_uniform_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return neg_log_uniform_prior.__call__(self, t)

class sorted_log_uniform_prior(log_uniform_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return log_uniform_prior.__call__(self, t)

class sorted_gaussian_prior(gaussian_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return gaussian_prior.__call__(self, t)

class sorted_laplace_prior(laplace_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return laplace_prior.__call__(self, t)

class sorted_cauchy_prior(cauchy_prior):
	def __call__(self, p):
		t = forced_identifiability_transform(p)
		return cauchy_prior.__call__(self, t)


p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3])
prior_types = [7,8,9,10,11,12,13]
prior_hyperparams = [[1, 100]]*7
dependence_lengths = [2,2,2,2,2,2,2]
param_prior_types = [0,1,2,3,4,5,6]
n_dims = 14
ip = inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, n_dims)
print ip(p)