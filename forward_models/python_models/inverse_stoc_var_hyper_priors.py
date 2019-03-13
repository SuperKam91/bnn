#########commercial modules
import numpy as np
import scipy.stats

##############in-house modules
import inverse_priors as ip
import inverse_stoc_hyper_priors as ishp

######one-dimensional prior distributions

class inverse_stoc_var_hyper_prior(ishp.inverse_stoc_hyper_prior):
	def __init__(self, hyperprior_types, var_prior_types, prior_types, hyperprior_params, var_prior_params, prior_hyperparams, hyper_dependence_lengths, var_dependence_lengths, dependence_lengths, param_hyperprior_types, var_param_prior_types, param_prior_types, n_stoc, n_stoc_var, n_dims):
		"""
		likelihood variances are also stochastic. n.b. their treatment is essentially same as nn parameters,
		but are treated separately for clarity.
		granularity of likelihood variances is either one var for all outputs or one var for each output.
		"""
		ishp.inverse_stoc_hyper_prior.__init__(self, hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims)
		assert (len(var_prior_types) == len(var_prior_params)), "length of var_prior_types and var_prior_params should be same"
		assert (len(var_dependence_lengths) == len(var_param_prior_types)), "var_dependence_lengths and var_param_prior_types should be the same length"		
		assert np.all(np.array(var_dependence_lengths, dtype = np.int64) == 1), "var_dependence_lengths must be [1] or [1]*n_stoc_var" 
		if len(var_dependence_lengths) != 1:
			assert (np.sum(var_dependence_lengths) == n_stoc_var), "in case of dependent var parameters, sum of var_dependence_lengths should equal n_stoc_var"
		self.var_prior_types = var_prior_types
		self.var_prior_params = var_prior_params
		self.n_stoc_var = n_stoc_var
		self.var_prior_ppfs = []
		self.params = np.zeros(n_stoc + n_stoc_var + n_dims)
		self.get_var_prior_ppf_objs()
		if (len(var_param_prior_types) == 1) and (len(var_dependence_lengths) == 1):
			self.var_prior_call = self.var_prior_call_ind_same
		else:
			self.var_param_prior_types = var_param_prior_types
			self.var_dependence_lengths = var_dependence_lengths
			self.var_prior_call = self.var_prior_call_by_dependence_lengths

	def get_var_prior_ppf_objs(self):
		"""
		same as get_ppf_objs in inverse_prior class, but
		for lhood variance param priors
		"""
		for i, p_type in enumerate(self.var_prior_types):
			var_prior_hyperparam1 = self.var_prior_params[i][0]
			var_prior_hyperparam2 = self.var_prior_params[i][1]
			if p_type == 0:
				self.var_prior_ppfs.append(ip.uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 1:
				self.var_prior_ppfs.append(ip.pos_log_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 2:
				self.var_prior_ppfs.append(ip.neg_log_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 3:
				self.var_prior_ppfs.append(ip.log_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 4:
				self.var_prior_ppfs.append(ip.gaussian_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 5:
				self.var_prior_ppfs.append(ip.laplace_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 6:
				self.var_prior_ppfs.append(ip.cauchy_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 7:
				self.var_prior_ppfs.append(ip.delta_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 8:
				self.var_prior_ppfs.append(ip.gamma_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 9:
				self.var_prior_ppfs.append(ip.sqrt_recip_gamma_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 10:
				self.var_prior_ppfs.append(ip.recip_gamma_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 11:
				self.var_prior_ppfs.append(ip.sorted_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 12:
				self.var_prior_ppfs.append(ip.sorted_pos_log_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 13:
				self.var_prior_ppfs.append(ip.sorted_neg_log_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 14:
				self.var_prior_ppfs.append(ip.sorted_log_uniform_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 15:
				self.var_prior_ppfs.append(ip.sorted_gaussian_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 16:
				self.var_prior_ppfs.append(ip.sorted_laplace_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 17:
				self.var_prior_ppfs.append(ip.sorted_cauchy_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 18:
				self.var_prior_ppfs.append(ip.sorted_delta_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 19:
				self.var_prior_ppfs.append(ip.sorted_gamma_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 20:
				self.var_prior_ppfs.append(ip.sorted_sqrt_rec_gam_prior(var_prior_hyperparam1, var_prior_hyperparam2))
			elif p_type == 21:
				self.var_prior_ppfs.append(ip.sorted_rec_gam_prior(var_prior_hyperparam1, var_prior_hyperparam2))
	
	def __call__(self, hypercube):
		self.hyperprior_call(hypercube[:self.n_stoc])
		# print "det hyperparams"
		# print self.hyperparams
		# print "stoc hyperparams"
		# print self.stoc_hyperparams
		self.var_prior_call(hypercube[self.n_stoc: self.n_stoc + self.n_stoc_var])
		return self.prior_call(hypercube[self.n_stoc + self.n_stoc_var:])

	def var_prior_call_ind_same(self, hypercube):
		# print "hypercube"
		# print hypercube
		# print 
		self.params[self.n_stoc: self.n_stoc + self.n_stoc_var] = self.var_prior_ppfs[0](hypercube)
		# print "var params"
		# print self.params[self.n_stoc: self.n_stoc + self.n_stoc_var]
		return self.params

	def var_prior_call_by_dependence_lengths(self, hypercube):
		start_ind = 0
		for i, dependence_length in enumerate(self.var_dependence_lengths):
			self.params[start_ind + self.n_stoc: start_ind + self.n_stoc + dependence_length] = self.var_prior_ppfs[self.var_param_prior_types[i]](hypercube[start_ind: start_ind + dependence_length])
			start_ind += dependence_length
		# print "var params"
		# print self.params[self.n_stoc: self.n_stoc + self.n_stoc_var]
		return self.params

	def prior_call_ind_same(self, hypercube):
		"""
		adapted from inverse_stoc_hyper_priors class to account for stoc vars in indexing
		"""
		self.params[self.n_stoc + self.n_stoc_var:] = self.prior_ppfs[0](hypercube, self.hyperparams, self.stoc_hyperparams)
		# print "params"
		# print self.params[self.n_stoc + self.n_stoc_var:]
		# print "total params"
		# print self.params
		return self.params

	def prior_call_by_par_type(self, hypercube):
		"""
		adapted from inverse_stoc_hyper_priors class to account for stoc vars in indexing
		"""
		pass

	def prior_call_by_dependence_lengths(self, hypercube):
		"""
		adapted from inverse_stoc_hyper_priors class to account for stoc vars in indexing
		"""
		start_ind = 0
		for i, dependence_length in enumerate(self.dependence_lengths):
			self.params[start_ind + self.n_stoc + self.n_stoc_var: start_ind + self.n_stoc + self.n_stoc_var + dependence_length] = self.prior_ppfs[self.param_prior_types[i]](hypercube[start_ind: start_ind + dependence_length], self.hyperparams[start_ind: start_ind + dependence_length], self.stoc_hyperparams[start_ind: start_ind + dependence_length])
			start_ind += dependence_length
		# print "params"
		# print self.params[self.n_stoc + self.n_stoc_var:]
		# print "total params"
		# print self.params
		return self.params