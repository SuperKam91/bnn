#########commercial modules
import numpy as np

#in-house modules

def report(results, n_top = 3):
	"""
	Utility function to report best scores of grid/randomsearch.
	results should be grid_search.cv_results_ or random_search_object.cv_results_.
	"""
	for i in range(1, n_top + 1):
	    candidates = np.flatnonzero(results['rank_test_score'] == i)
	    for candidate in candidates:
	        print("Model with rank: {0}".format(i))
	        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
	              results['mean_test_score'][candidate],
	              results['std_test_score'][candidate]))
	        print("Parameters: {0}".format(results['params'][candidate]))
	        print("")

