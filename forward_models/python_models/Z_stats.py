#########commercial modules
import numpy as np
import scipy.special
import warnings

"""
These functions were copied from gns module.
Notes about Keeton calcs aren't relevant here as polychord
doesn't produce evidence for main loop and for final contributions,
but hopefully the difference between the correct Keeton calcs. and the
ones here is minimal
"""

def calcEofLogZ(EofZ, EofZ2, space = 'linear'):
	"""
	calc E[log(Z)] from E[Z] and E[Z^2] or log(E[Z]) and log(E[Z^2])
	as given in Will's thesis (assumes Z log normal r.v.)
	NOTE in case of Keeton's total value, won't give correct result as won't account
	for covariance between loop and final contributions
	space refers to inputs, not output
	"""
	if space == 'linear':
		return 2. * np.log(EofZ) - 0.5 * np.log(EofZ2)
	elif space == 'log':
		return 2. * EofZ - 0.5 * EofZ2

def calcVarLogZ(EofZ, EofZ2, space = 'linear'):
	"""
	calc var[log(Z)] from E[Z] and E[Z^2] or log(E[Z]) and log(E[Z^2])
	as given in Will's thesis (assumes Z log normal r.v.)
	NOTE in case of Keeton's total value, won't give correct result as won't account
	for covariance between loop and final contributions
	space refers to inputs, not output
	"""
	if space == 'linear':
		return np.log(EofZ2) - 2. * np.log(EofZ)
	elif space == 'log':
		return EofZ2 - 2. * EofZ

def calcVarLogZII(EofZ, varZ, space):
	"""
	Uses propagation of uncertainty formula or 
	relationship between log-normal r.v.s and the normally distributed log of the log-normal r.v.s
	to calculate var[logZ] from EofZ and varZ (taken from Wikipedia)
	Apart from for Keeton total (where variance doesn't just come from E[Z] and E[Z^2] moments), 
	should be same as first implementation
	space refers to inputs, not output
	"""
	if space == 'linear':
		return np.log(1. + varZ / (EofZ)**2.)
	elif space == 'log':
		return np.logaddexp(0, varZ - 2. * EofZ)

def calcEofLogZII(EofZ, varZ, space):
	"""
	Calc E[log(Z)] from E[Z] and Var[Z]. Assumes Z is log-normally distributed (taken from Wikipedia)
	Apart from for Keeton total (where variance doesn't just come from E[Z] and E[Z^2] moments), 
	should be same as first implementation
	space refers to inputs, not output
	"""
	if space == 'linear':
		return np.log(EofZ**2. / (np.sqrt(varZ + EofZ**2.)))
	elif space == 'log':
		return EofZ - 0.5 * calcVarLogZII(EofZ, varZ, 'log')

def calcEofZII(EofLogZ, varLogZ, space):
	"""
	calc E[Z] (log(E[Z])) from E[logZ] and var[logZ]. Assumes Z is log-normal (taken from Wikipedia)
	space refers to output, not inputs. returning linear output runs the risk of under/overflow
	"""
	if space == 'linear':
		return np.exp(EofLogZ + 0.5 * varLogZ)
	elif space == 'log':
		return EofLogZ + 0.5 * varLogZ

def calcVarZII(varLogZ, EofLogZ, EofZ, space):
	"""
	Uses propagation of uncertainty formula or 
	relationship between log-normal r.v.s and the normally distributed log of the log-normal r.v.s
	to calculate var[Z] (log(var[Z])) from EofZ and varLogZ (taken from Wikipedia)
	space refers to output, not inputs. returning linear output runs the risk of under/overflow
	"""
	if space == 'linear':
		return np.exp(2. * EofLogZ + varLogZ) * (np.exp(varLogZ) - 1.)
	elif space == 'log':
		warnings.filterwarnings("error")
		try:
			ret =  2. * EofLogZ + varLogZ + np.log(np.exp(varLogZ) - 1)
			warnings.filterwarnings("default")
			return ret
		except RunTimeWarning: #assumed to be overflow associated with np.log(np.exp(varLogZ), so assume np.log(np.exp(varLogZ) - 1) ~ varLogZ
			ret =  2. * EofLogZ + varLogZ + varLogZ
			warnings.filterwarnings("default")
			return ret

def calcEofZCombII(EofLogZs, varLogZs, weights, space):
	"""
	calculate E(Z_t) from E(log(Z_i)) and Var(log(Z_i)) values where Z_t = sum_i W_i * Z_i
	space refers to output, not inputs. returning linear output runs the risk of under/overflow
	"""
	logEofZs = calcEofZII(EofLogZs, varLogZs, space = 'log')
	logWEofZs = np.log(weights) + logEofZs
	logWEofZ = scipy.special.logsumexp(logWEofZs)
	if space == 'linear':
		return np.exp(logWEofZ)
	elif space == 'log':
		return logWEofZ

def calcVarZCombII(EofLogZs, varLogZs, weights, space):
	logEofZs = calcEofZII(EofLogZs, varLogZs, space = 'log')
	logVarZs = calcVarZII(varLogZs, EofLogZs, logEofZs, space = 'log')
	logWVarZs = np.log(np.square(weights)) + logVarZs
	logWVarZ = scipy.special.logsumexp(logWVarZs) 
	if space == 'linear':
		return np.exp(logWVarZ)
	elif space == 'log':
		return logWVarZ

def calcEofLogZCombII(EofLogZs, varLogZs, weights):
	logWEofZ = calcEofZCombII(EofLogZs, varLogZs, weights, space = 'log')
	logWVarZ = calcVarZCombII(EofLogZs, varLogZs, weights, space = 'log')
	return calcEofLogZII(logWEofZ, logWVarZ, space = 'log')

def calcVarLogZCombII(EofLogZs, varLogZs, weights):
	logWEofZ = calcEofZCombII(EofLogZs, varLogZs, weights, space = 'log')
	logWVarZ = calcVarZCombII(EofLogZs, varLogZs, weights, space = 'log')
	return calcVarLogZII(EofZ, varZ, space = 'log')

