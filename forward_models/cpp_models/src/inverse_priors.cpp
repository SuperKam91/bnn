/* external codebases */
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

/* in-house code */
#include "inverse_priors.hpp"
#include "mathematics.hpp"

base_prior::~base_prior() {
}

//in functions which take theta by reference, could call functions which return theta,
//but may be overhead in copying from return value to rvalue.

uniform_prior::uniform_prior(double a_, double b_) :
	a(a_),
	b(b_) {
}

Eigen::VectorXd uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return a + ((b - a) * p_m).array();
}

void uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = a + ((b - a) * p_m).array();
}

Eigen::VectorXd pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return (log(a) + ((log(b) - log(a)) * p_m).array()).array().exp();
}

void pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = (log(a) + ((log(b) - log(a)) * p_m).array()).array().exp();
}

Eigen::VectorXd neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return -1. * pos_log_uniform_prior::operator()(p_m);
}

void neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	pos_log_uniform_prior::operator()(p_m, theta_m);
	theta_m *= -1.; 
}

//as with python version, more efficient to define single ppf over positive + negative domain
//version in depracated.cpp may be more efficient, but less readable
Eigen::VectorXd log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd p_m_rescaled(p_m.size());
	p_m_rescaled = (p_m.array() < 0.5).select((0.5 - p_m.array()) * 2., (p_m.array() - 0.5) * 2.); 
    return (p_m.array() < 0.5).select(-1. * pos_log_uniform_prior::operator()(p_m_rescaled), pos_log_uniform_prior::operator()(p_m_rescaled));
}

void log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd p_m_rescaled(p_m.size());
	p_m_rescaled = (p_m.array() < 0.5).select((0.5 - p_m.array()) * 2., (p_m.array() - 0.5) * 2.); 
	//have to make copies here in .select() unless completely new method is used (i.e. single function over positive + negative domain)
    theta_m = (p_m.array() < 0.5).select(-1. * pos_log_uniform_prior::operator()(p_m_rescaled), pos_log_uniform_prior::operator()(p_m_rescaled));
}

gaussian_prior::gaussian_prior(double mu_, double sigma_) :
	mu(mu_),
	sigma(sigma_) {
}

//could define std::ptr_fun(inv_erf) as member variable instead of having to create temporary everytime it is called.
//or make it a global
Eigen::VectorXd gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return mu + sigma * sqrt(2) * ((2 * p_m).array() - 1.).unaryExpr(std::ptr_fun(inv_erf));
}

void gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = mu + sigma * sqrt(2) * ((2 * p_m).array() - 1.).unaryExpr(std::ptr_fun(inv_erf));
}

laplace_prior::laplace_prior(double mu_, double b_) :
	mu(mu_),
	b(b_) {
}

Eigen::VectorXd laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return mu - b * ((p_m.array() - 0.5).unaryExpr(std::ptr_fun(sgn))) * ((1. - 2. * (p_m.array() - 0.5).abs())).log();
}

void laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = mu - b * ((p_m.array() - 0.5).unaryExpr(std::ptr_fun(sgn))) * ((1. - 2. * (p_m.array() - 0.5).abs())).log();
}

cauchy_prior::cauchy_prior(double x0_, double gamma_) :
	x0(x0_),
	gamma(gamma_) {
}

Eigen::VectorXd cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return x0 + gamma * (M_PI * (p_m.array() - 0.5)).tan();
}

void cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = x0 + gamma * (M_PI * (p_m.array() - 0.5)).tan();
}

delta_prior::delta_prior(double value_, double sigma_) :
	value(value_),
	sigma(sigma_) {
}

Eigen::VectorXd delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return Eigen::VectorXd::Constant(p_m.size(), value);
}

//test to ensure this doesn't reallocate theta_m to different memory location
void delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = Eigen::VectorXd::Constant(p_m.size(), value);
}

gamma_prior::gamma_prior(double a_, double b_) :
	a(a_),
	b(b_) {
}

Eigen::VectorXd gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	return x0 + gamma * (M_PI * (p_m.array() - 0.5)).tan();
}

void gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	theta_m = x0 + gamma * (M_PI * (p_m.array() - 0.5)).tan();
}

Eigen::VectorXd forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	for (long int i = n - 2; i >= 0; --i) {
	    t_m(i) = pow(p_m(i), (1. / (static_cast<double>(i) + 1))) * t_m(i + 1);
	}
	return t_m;
}

//in most cases I think t_m will be declared locally and assigned only once (at initialisation so NRVO can be used)
//thus I don't think these pass by reference functions will be used
void forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	for (long int i = n - 2; i >= 0; --i) {
	    t_m(i) = pow(p_m(i), (1. / (static_cast<double>(i) + 1))) * t_m(i + 1);
	}
}

//as is the case with python version, doesn't work (only gets last two elements correct)
Eigen::VectorXd forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	Eigen::VectorXd i = Eigen::VectorXd::LinSpaced(n - 1, 1., static_cast<double>(n) - 1.);  	
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(i.array().inverse()) * t_m(indices.array() + 1).array();
	return t_m;
}

void forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	Eigen::VectorXd i = Eigen::VectorXd::LinSpaced(n - 1, 1., static_cast<double>(n) - 1.);  	
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(i.array().inverse()) * t_m(indices.array() + 1).array();
}

//same as forced_identifiability_transform2 but doesn't require extra eigen::vector for pow exponents
Eigen::VectorXd forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(((indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() + 1).array();
	return t_m;
}

void forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(((indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() + 1).array();
}

//actually works unlike the python version! though I'm not sure how stable it is
Eigen::VectorXd forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> rev_indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, n - 2, 0);	  
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(rev_indices).array().pow(((rev_indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	return t_m.reverse();
}

//not tested (.reverseInPlace() not been tested)
void forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> rev_indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, n - 2, 0);	  
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(rev_indices).array().pow(((rev_indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	t_m.reverseInPlace();
}

//same as forced_identifiability_transform3 but doesn't require an extra eigen::vector for reversed indices
Eigen::VectorXd forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices.reverse().array() - 1).array().pow((((indices.reverse().array() - 1).cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	return t_m.reverse();
}

void forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices.reverse().array() - 1).array().pow((((indices.reverse().array() - 1).cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	t_m.reverseInPlace();
}

Eigen::VectorXd sorted_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return uniform_prior::operator()(t_m);
}

void sorted_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	uniform_prior::operator()(t_m, theta_m);
}

Eigen::VectorXd sorted_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return pos_log_uniform_prior::operator()(t_m);
}

void sorted_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	pos_log_uniform_prior::operator()(t_m, theta_m);
}

Eigen::VectorXd sorted_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return neg_log_uniform_prior::operator()(t_m);
}

void sorted_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	neg_log_uniform_prior::operator()(t_m, theta_m);
}

Eigen::VectorXd sorted_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return log_uniform_prior::operator()(t_m);
}

void sorted_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	log_uniform_prior::operator()(t_m, theta_m);
}

Eigen::VectorXd sorted_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return gaussian_prior::operator()(t_m);
}

void sorted_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	gaussian_prior::operator()(t_m, theta_m);
}

Eigen::VectorXd sorted_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return laplace_prior::operator()(t_m);
}

void sorted_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	laplace_prior::operator()(t_m, theta_m);
}

Eigen::VectorXd sorted_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	return cauchy_prior::operator()(t_m);
}

void sorted_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	Eigen::VectorXd t_m = forced_identifiability_transform(p_m);
	cauchy_prior::operator()(t_m, theta_m);
}

using namespace std::placeholders;
//since prior_call_by_dependence_lengths is the only implementation the calling function can use, if one wants to use the same prior for
//all parameters, set the dependence length = n_dims 
inverse_prior::inverse_prior(std::vector<uint> prior_types_, std::vector<double> prior_hyperparams_, std::vector<uint> dependence_lengths_, std::vector<uint> param_prior_types_, uint n_dims_) :
	prior_types(prior_types_),
	prior_hyperparams(prior_hyperparams_),
	dependence_lengths(dependence_lengths_),
	param_prior_types(param_prior_types_),
	n_dims(n_dims_),
	ppf_ptr_v(get_ppf_ptr_vec()) {
	}

inverse_prior::~inverse_prior() {
	for (std::vector<base_prior *>::iterator it = ppf_ptr_v.begin(); it < ppf_ptr_v.end(); it++) {
    	delete *it;
    }
}

//assumes prior_hyperparams is twice length of prior_types, i.e. each prior has two hyperparam values
std::vector<base_prior *> inverse_prior::get_ppf_ptr_vec() {
	std::vector<base_prior *> ppf_ptr_vec;
	ppf_ptr_vec.reserve(prior_types.size());
	double hyperparam1;
	double hyperparam2;
	for (uint i = 0; i < prior_types.size(); ++i) {
		hyperparam1 = prior_hyperparams.at(2 * i);
		hyperparam2 = prior_hyperparams.at(2 * i + 1);
		if (prior_types.at(i) == 0) {
			ppf_ptr_vec.push_back(new uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 1) {
			ppf_ptr_vec.push_back(new pos_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 2) {
			ppf_ptr_vec.push_back(new neg_log_uniform_prior(hyperparam1, hyperparam2));			
		}
		else if (prior_types.at(i) == 3) {
			ppf_ptr_vec.push_back(new log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 4) {
			ppf_ptr_vec.push_back(new gaussian_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 5) {
			ppf_ptr_vec.push_back(new laplace_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 6) {
			ppf_ptr_vec.push_back(new cauchy_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 7) {
			ppf_ptr_vec.push_back(new sorted_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 8) {
			ppf_ptr_vec.push_back(new sorted_pos_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 9) {
			ppf_ptr_vec.push_back(new sorted_neg_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 10) {
			ppf_ptr_vec.push_back(new sorted_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 11) {
			ppf_ptr_vec.push_back(new sorted_gaussian_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 12) {
			ppf_ptr_vec.push_back(new sorted_laplace_prior(hyperparam1, hyperparam2));
		}
		else if (prior_types.at(i) == 13) {
			ppf_ptr_vec.push_back(new sorted_cauchy_prior(hyperparam1, hyperparam2));
		}
	}
	return ppf_ptr_vec;
}

//uses prior function calls which take reference to Eigen::Ref as argument, as may be faster than returning value.
//trade-off between making reference, then assigning the rvalue to the target in-function versus creating return value, converting return value to an rvalue, then assigning this rvalue to the target in the calling body. Which is fastest depends on compiler optimisations, which I need to test.
//n.b. RVO/NRVO only occurs when initialising an object
void inverse_prior::prior_call_by_dependence_lengths(Eigen::Ref<Eigen::VectorXd> & cube_m, Eigen::Ref<Eigen::VectorXd> & theta_m) {
	uint start_ind = 0;
	uint func_count = 0;
	uint dependence_length;
	for (uint i = 0; i < dependence_lengths.size(); ++i){ 
		dependence_length = dependence_lengths.at(i);
		(ppf_ptr_v.at(param_prior_types.at(func_count)))->operator()(cube_m.segment(start_ind, dependence_length), theta_m.segment(start_ind, dependence_length));
		start_ind += dependence_length;
		func_count += 1;
	}
}

void inverse_prior::prior_call_ind_same(Eigen::Ref<Eigen::VectorXd> & cube_m, Eigen::Ref<Eigen::VectorXd> & theta_m) {
	(ppf_ptr_v.at(0))->operator()(cube_m, theta_m);
}

//in future may implement method for all parameters independent but use different priors,
//and most importantly, independent parameters which aren't contiguous (will require truth array as in python case)
//if one makes prior_call_ptr a member variable, program breaks (see declaration in header file for error), 
//could make it global, but I think there's a chance this is dangerous, so I won't.
//so have to have it as local and evaluate conditional upon every call instead, which is a shame.
//could try get it to work as member variable, but cba atm.
void inverse_prior::operator()(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m){
	std::function<void(Eigen::Ref<Eigen::VectorXd> &, Eigen::Ref<Eigen::VectorXd> &)> prior_call_ptr;
	if (dependence_lengths.size() == 1) {
		prior_call_ptr = std::bind(&inverse_prior::prior_call_ind_same, this, _1, _2);
	}
	else {
		prior_call_ptr = std::bind(&inverse_prior::prior_call_by_dependence_lengths, this, _1, _2);
	}
	prior_call_ptr(cube_m, theta_m);
	//alternative, probably more efficient, but less elegant
	// if (dependence_lengths.size() == 1) {
		// prior_call_ind_same(cube_m, theta_m);	
	// }
	// else {
		// prior_call_by_dependence_lengths(cube_m, theta_m);		
	// }
}