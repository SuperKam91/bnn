/* external codebases */
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <gsl/gsl_cdf.h>

/* in-house code */
#include "inverse_priors.hpp"
#include "inverse_stoc_hyper_priors.hpp"
#include "mathematics.hpp"

//code for these inverse prior classes is quite duplicated here from inverse_priors.cpp.
//perhaps instead could have just added stochastic hyperparam methods to classes there,
//but wanted to keep the two sets of classes separate for now

sh_base_prior::~sh_base_prior() {
}

sh_uniform_prior::sh_uniform_prior() {
}

//functions which take hyperparams as eigen vectors may not be the most efficient, but this format
//is required to replicate python implementation where hyperparams are stored in vectors of size n_dims.

Eigen::VectorXd sh_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	return a + ((b - a) * p_m).array();
}

void sh_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	theta_m = a + ((b - a) * p_m).array();
}

void sh_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	theta_m = a + ((b - a).array() * p_m.array()).matrix();
}

sh_pos_log_uniform_prior::sh_pos_log_uniform_prior() {
}

Eigen::VectorXd sh_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	return (log(a) + ((log(b) - log(a)) * p_m).array()).array().exp();
}

//need function which returns eigen vector and takes hyperparams as eigen vecs for 
//void sh_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b)
Eigen::VectorXd sh_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	return (a.array().log() + ((b.array().log() - a.array().log()) * p_m.array()).array()).array().exp();
}

void sh_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	theta_m = (log(a) + ((log(b) - log(a)) * p_m).array()).array().exp();
}

void sh_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	theta_m = (a.array().log() + ((b.array().log() - a.array().log()).array() * p_m.array()).array()).array().exp();
}

sh_neg_log_uniform_prior::sh_neg_log_uniform_prior() {
}

Eigen::VectorXd sh_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	return -1. * sh_pos_log_uniform_prior::operator()(p_m, a, b);
}

//also needed for
//void sh_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) 
Eigen::VectorXd sh_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	return -1. * sh_pos_log_uniform_prior::operator()(p_m, a, b);
}

void sh_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	sh_pos_log_uniform_prior::operator()(p_m, theta_m, a, b);
	theta_m *= -1.; 
}

void sh_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	sh_pos_log_uniform_prior::operator()(p_m, theta_m, a, b);
	theta_m *= -1.; 
}

sh_log_uniform_prior::sh_log_uniform_prior() {
}

Eigen::VectorXd sh_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd p_m_rescaled(p_m.size());
	p_m_rescaled = (p_m.array() < 0.5).select((0.5 - p_m.array()) * 2., (p_m.array() - 0.5) * 2.); 
    return (p_m.array() < 0.5).select(-1. * sh_pos_log_uniform_prior::operator()(p_m_rescaled, a, b), sh_pos_log_uniform_prior::operator()(p_m_rescaled, a, b));
}

void sh_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd p_m_rescaled(p_m.size());
	p_m_rescaled = (p_m.array() < 0.5).select((0.5 - p_m.array()) * 2., (p_m.array() - 0.5) * 2.); 
    theta_m = (p_m.array() < 0.5).select(-1. * sh_pos_log_uniform_prior::operator()(p_m_rescaled, a, b), sh_pos_log_uniform_prior::operator()(p_m_rescaled, a, b));
}

void sh_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd p_m_rescaled(p_m.size());
	p_m_rescaled = (p_m.array() < 0.5).select((0.5 - p_m.array()) * 2., (p_m.array() - 0.5) * 2.); 
    theta_m = (p_m.array() < 0.5).select(-1. * sh_pos_log_uniform_prior::operator()(p_m_rescaled, a, b), sh_pos_log_uniform_prior::operator()(p_m_rescaled, a, b));
}

sh_gaussian_prior::sh_gaussian_prior() {
}

Eigen::VectorXd sh_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double sigma) {
	return mu + sigma * sqrt(2) * ((2 * p_m).array() - 1.).unaryExpr(std::ptr_fun(inv_erf));
}

void sh_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double sigma) {
	theta_m = mu + sigma * sqrt(2) * ((2 * p_m).array() - 1.).unaryExpr(std::ptr_fun(inv_erf));
}

void sh_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> sigma) {
	theta_m = mu + (sigma.array() * sqrt(2) * ((2 * p_m).array() - 1.).unaryExpr(std::ptr_fun(inv_erf))).matrix();
}

sh_laplace_prior::sh_laplace_prior() {
}

Eigen::VectorXd sh_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double b) {
	return mu - b * ((p_m.array() - 0.5).unaryExpr(std::ptr_fun(sgn))) * ((1. - 2. * (p_m.array() - 0.5).abs())).log();
}

void sh_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double b) {
	theta_m = mu - b * ((p_m.array() - 0.5).unaryExpr(std::ptr_fun(sgn))) * ((1. - 2. * (p_m.array() - 0.5).abs())).log();
}

void sh_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> b) {
	theta_m = mu - (b.array() * ((p_m.array() - 0.5).unaryExpr(std::ptr_fun(sgn))) * ((1. - 2. * (p_m.array() - 0.5).abs())).log()).matrix();
}

sh_cauchy_prior::sh_cauchy_prior() {
}

Eigen::VectorXd sh_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double x0, double gamma) {
	return x0 + gamma * (M_PI * (p_m.array() - 0.5)).tan();
}

void sh_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double x0, double gamma) {
	theta_m = x0 + gamma * (M_PI * (p_m.array() - 0.5)).tan();
}

void sh_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> gamma) {
	theta_m = x0 + (gamma.array() * (M_PI * (p_m.array() - 0.5)).tan()).matrix();
}

sh_delta_prior::sh_delta_prior() {
}

Eigen::VectorXd sh_delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double value, double sigma) {
	return Eigen::VectorXd::Constant(p_m.size(), value);
}

void sh_delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double value, double sigma) {
	theta_m = Eigen::VectorXd::Constant(p_m.size(), value);
}

void sh_delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> value, Eigen::Ref<Eigen::VectorXd> sigma) {
	theta_m = value;
}

sh_gamma_prior::sh_gamma_prior() {
}

Eigen::VectorXd sh_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	long int n = p_m.size();
	Eigen::VectorXd theta_m(n);
	for (long int i = 0; i < n; ++i) {
		theta_m(i) = gsl_cdf_gamma_Pinv(p_m(i), a, 1. / b);
	}
	return theta_m;
}

void sh_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	long int n = p_m.size();
	for (long int i = 0; i < n; ++i) {
		theta_m(i) = gsl_cdf_gamma_Pinv(p_m(i), a, 1. / b);
	}
}

void sh_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	long int n = p_m.size();
	for (long int i = 0; i < n; ++i) {
		theta_m(i) = gsl_cdf_gamma_Pinv(p_m(i), a(i), 1. / b(i));
	}
}

Eigen::VectorXd sh_sqrt_recip_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	return (sh_gamma_prior::operator()(p_m, a, b)).array().inverse().sqrt();
}

void sh_sqrt_recip_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	sh_gamma_prior::operator()(p_m, theta_m, a, b);
	theta_m = theta_m.array().inverse().sqrt();
}

void sh_sqrt_recip_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	sh_gamma_prior::operator()(p_m, theta_m, a, b);
	theta_m = theta_m.array().inverse().sqrt();
}

//following are exact replicas from inverse_priors.cpp. just included here for completeness, basically.
//----------------------------------------------------------------------------------------------------------

Eigen::VectorXd sh_forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m) {
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
void sh_forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	for (long int i = n - 2; i >= 0; --i) {
	    t_m(i) = pow(p_m(i), (1. / (static_cast<double>(i) + 1))) * t_m(i + 1);
	}
}

Eigen::VectorXd sh_forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	Eigen::VectorXd i = Eigen::VectorXd::LinSpaced(n - 1, 1., static_cast<double>(n) - 1.);  	
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(i.array().inverse()) * t_m(indices.array() + 1).array();
	return t_m;
}

void sh_forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	Eigen::VectorXd i = Eigen::VectorXd::LinSpaced(n - 1, 1., static_cast<double>(n) - 1.);  	
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(i.array().inverse()) * t_m(indices.array() + 1).array();
}

Eigen::VectorXd sh_forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(((indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() + 1).array();
	return t_m;
}

void sh_forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 0, n - 2);  
	t_m(n - 1) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices).array().pow(((indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() + 1).array();
}

Eigen::VectorXd sh_forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> rev_indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, n - 2, 0);	  
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(rev_indices).array().pow(((rev_indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	return t_m.reverse();
}

void sh_forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> rev_indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, n - 2, 0);	  
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(rev_indices).array().pow(((rev_indices.cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	t_m.reverseInPlace();
}

Eigen::VectorXd sh_forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m) {
	long int n = p_m.size();
	Eigen::VectorXd t_m(n);
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices.reverse().array() - 1).array().pow((((indices.reverse().array() - 1).cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	return t_m.reverse();
}

void sh_forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m) {
	long int n = t_m.size();
	Eigen::Matrix<long int, Eigen::Dynamic, 1> indices = Eigen::Matrix<long int, Eigen::Dynamic, 1>::LinSpaced(n - 1, 1, n - 1);
	t_m(0) = pow(p_m(n - 1), (1. / static_cast<double>(n)));
	t_m(indices) = p_m(indices.reverse().array() - 1).array().pow((((indices.reverse().array() - 1).cast<double>()).array() + 1).inverse()) * t_m(indices.array() - 1).array();
	t_m.reverseInPlace();
}

//----------------------------------------------------------------------------------------------------------

Eigen::VectorXd sh_sorted_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_uniform_prior::operator()(t_m, a, b);
}

void sh_sorted_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_uniform_prior::operator()(t_m, theta_m, a, b);
}

void sh_sorted_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_uniform_prior::operator()(t_m, theta_m, a, b);
}

Eigen::VectorXd sh_sorted_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_pos_log_uniform_prior::operator()(t_m, a, b);
}

void sh_sorted_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_pos_log_uniform_prior::operator()(t_m, theta_m, a, b);
}

void sh_sorted_pos_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_pos_log_uniform_prior::operator()(t_m, theta_m, a, b);
}

Eigen::VectorXd sh_sorted_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_neg_log_uniform_prior::operator()(t_m, a, b);
}

void sh_sorted_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_neg_log_uniform_prior::operator()(t_m, theta_m, a, b);
}

void sh_sorted_neg_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_neg_log_uniform_prior::operator()(t_m, theta_m, a, b);
}

Eigen::VectorXd sh_sorted_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_log_uniform_prior::operator()(t_m, a, b);
}

void sh_sorted_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_log_uniform_prior::operator()(t_m, theta_m, a, b);
}

void sh_sorted_log_uniform_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_log_uniform_prior::operator()(t_m, theta_m, a, b);
}

Eigen::VectorXd sh_sorted_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double sigma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_gaussian_prior::operator()(t_m, mu, sigma);
}

void sh_sorted_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double sigma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_gaussian_prior::operator()(t_m, theta_m, mu, sigma);
}

void sh_sorted_gaussian_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> sigma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_gaussian_prior::operator()(t_m, theta_m, mu, sigma);
}

Eigen::VectorXd sh_sorted_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_laplace_prior::operator()(t_m, mu, b);
}

void sh_sorted_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_laplace_prior::operator()(t_m, theta_m, mu, b);
}

void sh_sorted_laplace_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_laplace_prior::operator()(t_m, theta_m, mu, b);
}

Eigen::VectorXd sh_sorted_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double x0, double gamma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_cauchy_prior::operator()(t_m, x0, gamma);
}

void sh_sorted_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double x0, double gamma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_cauchy_prior::operator()(t_m, theta_m, x0, gamma);
}

void sh_sorted_cauchy_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> gamma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_cauchy_prior::operator()(t_m, theta_m, x0, gamma);
}

Eigen::VectorXd sh_sorted_delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double value, double sigma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_delta_prior::operator()(t_m, value, sigma);
}

void sh_sorted_delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double value, double sigma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_delta_prior::operator()(t_m, theta_m, value, sigma);
}

void sh_sorted_delta_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> value, Eigen::Ref<Eigen::VectorXd> sigma) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_delta_prior::operator()(t_m, theta_m, value, sigma);
}

Eigen::VectorXd sh_sorted_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_gamma_prior::operator()(t_m, a, b);
}

void sh_sorted_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_gamma_prior::operator()(t_m, theta_m, a, b);
}

void sh_sorted_gamma_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_gamma_prior::operator()(t_m, theta_m, a, b);
}

Eigen::VectorXd sh_sorted_sqrt_rec_gam_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	return sh_sqrt_recip_gamma_prior::operator()(t_m, a, b);
}

void sh_sorted_sqrt_rec_gam_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_sqrt_recip_gamma_prior::operator()(t_m, theta_m, a, b);
}

void sh_sorted_sqrt_rec_gam_prior::operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b) {
	Eigen::VectorXd t_m = sh_forced_identifiability_transform(p_m);
	sh_sqrt_recip_gamma_prior::operator()(t_m, theta_m, a, b);
}

using namespace std::placeholders;
//n.b. this class is a little different from the python implementation in the sense that the prior hyperparams don't have length n_dims
//
sh_inverse_prior::sh_inverse_prior(std::vector<uint> hyperprior_types_, std::vector<uint> prior_types_, std::vector<double> hyperprior_params_, std::vector<double> prior_hyperparams_, std::vector<uint> hyper_dependence_lengths_, std::vector<uint> dependence_lengths_, std::vector<uint> param_hyperprior_types_, std::vector<uint> param_prior_types_, uint n_stoc_, uint n_dims_) :
	hyperprior_types(hyperprior_types_),
	prior_types(prior_types_),
	hyperprior_params(hyperprior_params_),
	prior_hyperparams(prior_hyperparams_),
	hyper_dependence_lengths(hyper_dependence_lengths_),
	dependence_lengths(dependence_lengths_),
	param_hyperprior_types(param_hyperprior_types_),
	param_prior_types(param_prior_types_),
	n_stoc(n_stoc_),
	n_dims(n_dims_),
	ppf_hp_ptr_v(get_hp_ppf_ptr_vec()),
	ppf_ptr_v(get_ppf_ptr_vec()),
	hyperparams_m(fill_det_hypers()),
	stoc_hyperparams_m(init_stoc_hypers()) {
}

sh_inverse_prior::~sh_inverse_prior() {
	for (std::vector<sh_base_prior *>::iterator it = ppf_ptr_v.begin(); it < ppf_ptr_v.end(); it++) {
    	delete *it;
    }
    for (std::vector<base_prior *>::iterator it = ppf_hp_ptr_v.begin(); it < ppf_hp_ptr_v.end(); it++) {
    	delete *it;
    }
}

//assumes hyperprior_params is twice length of hyperprior_types, i.e. each hyperprior has two hyperparam values
std::vector<base_prior *> sh_inverse_prior::get_hp_ppf_ptr_vec() {
	std::vector<base_prior *> ppf_hp_ptr_vec;
	ppf_hp_ptr_vec.reserve(hyperprior_types.size());
	double hyperparam1;
	double hyperparam2;
	for (uint i = 0; i < hyperprior_types.size(); ++i) {
		hyperparam1 = hyperprior_params.at(2 * i);
		hyperparam2 = hyperprior_params.at(2 * i + 1);
		if (hyperprior_types.at(i) == 0) {
			ppf_hp_ptr_vec.push_back(new uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 1) {
			ppf_hp_ptr_vec.push_back(new pos_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 2) {
			ppf_hp_ptr_vec.push_back(new neg_log_uniform_prior(hyperparam1, hyperparam2));			
		}
		else if (hyperprior_types.at(i) == 3) {
			ppf_hp_ptr_vec.push_back(new log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 4) {
			ppf_hp_ptr_vec.push_back(new gaussian_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 5) {
			ppf_hp_ptr_vec.push_back(new laplace_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 6) {
			ppf_hp_ptr_vec.push_back(new cauchy_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 7) {
			ppf_hp_ptr_vec.push_back(new delta_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 8) {
			ppf_hp_ptr_vec.push_back(new gamma_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 9) {
			ppf_hp_ptr_vec.push_back(new sqrt_recip_gamma_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 10) {
			ppf_hp_ptr_vec.push_back(new sorted_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 11) {
			ppf_hp_ptr_vec.push_back(new sorted_pos_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 12) {
			ppf_hp_ptr_vec.push_back(new sorted_neg_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 13) {
			ppf_hp_ptr_vec.push_back(new sorted_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 14) {
			ppf_hp_ptr_vec.push_back(new sorted_gaussian_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 15) {
			ppf_hp_ptr_vec.push_back(new sorted_laplace_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 16) {
			ppf_hp_ptr_vec.push_back(new sorted_cauchy_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 17) {
			ppf_hp_ptr_vec.push_back(new sorted_delta_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 18) {
			ppf_hp_ptr_vec.push_back(new sorted_gamma_prior(hyperparam1, hyperparam2));
		}
		else if (hyperprior_types.at(i) == 19) {
			ppf_hp_ptr_vec.push_back(new sorted_sqrt_rec_gam_prior(hyperparam1, hyperparam2));
		}
	}
	return ppf_hp_ptr_vec;
}

std::vector<sh_base_prior *> sh_inverse_prior::get_ppf_ptr_vec() {
	std::vector<sh_base_prior *> ppf_ptr_vec;
	ppf_ptr_vec.reserve(prior_types.size());
	for (uint i = 0; i < prior_types.size(); ++i) {
		if (prior_types.at(i) == 0) {
			ppf_ptr_vec.push_back(new sh_uniform_prior());
		}
		else if (prior_types.at(i) == 1) {
			ppf_ptr_vec.push_back(new sh_pos_log_uniform_prior());
		}
		else if (prior_types.at(i) == 2) {
			ppf_ptr_vec.push_back(new sh_neg_log_uniform_prior());			
		}
		else if (prior_types.at(i) == 3) {
			ppf_ptr_vec.push_back(new sh_log_uniform_prior());
		}
		else if (prior_types.at(i) == 4) {
			ppf_ptr_vec.push_back(new sh_gaussian_prior());
		}
		else if (prior_types.at(i) == 5) {
			ppf_ptr_vec.push_back(new sh_laplace_prior());
		}
		else if (prior_types.at(i) == 6) {
			ppf_ptr_vec.push_back(new sh_cauchy_prior());
		}
		else if (prior_types.at(i) == 7) {
			ppf_ptr_vec.push_back(new sh_delta_prior());
		}
		else if (prior_types.at(i) == 8) {
			ppf_ptr_vec.push_back(new sh_gamma_prior());
		}
		else if (prior_types.at(i) == 9) {
			ppf_ptr_vec.push_back(new sh_sqrt_recip_gamma_prior());
		}
		else if (prior_types.at(i) == 10) {
			ppf_ptr_vec.push_back(new sh_sorted_uniform_prior());
		}
		else if (prior_types.at(i) == 11) {
			ppf_ptr_vec.push_back(new sh_sorted_pos_log_uniform_prior());
		}
		else if (prior_types.at(i) == 12) {
			ppf_ptr_vec.push_back(new sh_sorted_neg_log_uniform_prior());
		}
		else if (prior_types.at(i) == 13) {
			ppf_ptr_vec.push_back(new sh_sorted_log_uniform_prior());
		}
		else if (prior_types.at(i) == 14) {
			ppf_ptr_vec.push_back(new sh_sorted_gaussian_prior());
		}
		else if (prior_types.at(i) == 15) {
			ppf_ptr_vec.push_back(new sh_sorted_laplace_prior());
		}
		else if (prior_types.at(i) == 16) {
			ppf_ptr_vec.push_back(new sh_sorted_cauchy_prior());
		}
		else if (prior_types.at(i) == 17) {
			ppf_ptr_vec.push_back(new sh_sorted_delta_prior());
		}
		else if (prior_types.at(i) == 18) {
			ppf_ptr_vec.push_back(new sh_sorted_gamma_prior());
		}
		else if (prior_types.at(i) == 19) {
			ppf_ptr_vec.push_back(new sh_sorted_sqrt_rec_gam_prior());
		}
	}
	return ppf_ptr_vec;
}

//small function to return empty eigen vec with size n_dims.
Eigen::VectorXd sh_inverse_prior::init_stoc_hypers() {
	Eigen::VectorXd stoc_hyperparams(n_dims);
	return stoc_hyperparams;
}

//probably won't be vectorised so not very efficient. 
//not sure if you can do slices as lvalues using eigen, so don't know if it can be improved easily.
//only called once upon creation of object, so not a big deal. 
Eigen::VectorXd sh_inverse_prior::fill_det_hypers() {
	uint start_ind = 0;
	uint dependence_length;
	Eigen::VectorXd hyperparams(n_dims);
	if (dependence_lengths.size() == 1) {
		hyperparams = Eigen::VectorXd::Constant(hyperparams.size(), prior_hyperparams.at(0));
	}
	for (uint i = 0; i < dependence_lengths.size(); ++i) {
		dependence_length = dependence_lengths.at(i);
		// for (uint j = 0; j < dependence_length; ++j) {
		// 	hyperparams(start_ind + j) = prior_hyperparams.at(param_prior_types.at(i)); 
		// }
		hyperparams.segment(start_ind, dependence_length) = Eigen::VectorXd::Constant(dependence_length, prior_hyperparams.at(param_prior_types.at(i)));
		start_ind += dependence_length;
	}
	return hyperparams;
}

void sh_inverse_prior::hyperprior_call_ind(Eigen::Ref<Eigen::VectorXd> cube_m) {
	Eigen::VectorXd temp(1);
	(ppf_hp_ptr_v.at(0))->operator()(cube_m, temp);
	stoc_hyperparams_m = Eigen::VectorXd::Constant(stoc_hyperparams_m.size(), temp(0));
}

//again, need to test for loop and slicing method to see if latter works.
void sh_inverse_prior::hyperprior_call_by_hyper_dependence_lengths(Eigen::Ref<Eigen::VectorXd> cube_m) {
	uint start_ind = 0;
	uint dependence_length;
	Eigen::VectorXd temp(1);
	for (uint i = 0; i < hyper_dependence_lengths.size(); ++i) { 
		dependence_length = hyper_dependence_lengths.at(i);
		(ppf_hp_ptr_v.at(param_hyperprior_types.at(i)))->operator()(cube_m.segment(i, 1), temp);
		// for (uint j = 0; j < dependence_length; ++j) {
		// 	stoc_hyperparams_m(start_ind + j) = temp(0);
		// }
		stoc_hyperparams_m.segment(start_ind, dependence_length) = Eigen::VectorXd::Constant(dependence_length, temp(0));
		start_ind += dependence_length;
	}
}

void sh_inverse_prior::prior_call_by_dependence_lengths(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	uint start_ind = 0;
	uint dependence_length;
	for (uint i = 0; i < dependence_lengths.size(); ++i) { 
		dependence_length = dependence_lengths.at(i);
		(ppf_ptr_v.at(param_prior_types.at(i)))->operator()(cube_m.segment(start_ind, dependence_length), theta_m.segment(start_ind, dependence_length), hyperparams_m.segment(start_ind, dependence_length), stoc_hyperparams_m.segment(start_ind, dependence_length));
		start_ind += dependence_length;
	}
}

void sh_inverse_prior::prior_call_ind_same(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	(ppf_ptr_v.at(0))->operator()(cube_m, theta_m, hyperparams_m, stoc_hyperparams_m);
}

//using & references in this function (i.e. Eigen::Ref<Eigen::VectorXd> &) 
//don't work unlike in the inverse_priors.cpp equivalent function.
//seems to be because here cube_m.segment() is used instead of cube_m.
//same goes for signatures of functions called by this function
void sh_inverse_prior::operator()(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	std::cout << "cube" << std::endl;
	std::cout << cube_m << std::endl;

	std::cout << "det hyperparams" << std::endl;
	std::cout << hyperparams_m << std::endl;
	std::function<void(Eigen::Ref<Eigen::VectorXd>)> hyperprior_call_ptr;
	if (hyper_dependence_lengths.size() == 1) {
		hyperprior_call_ptr = std::bind(&sh_inverse_prior::hyperprior_call_ind, this, _1);
	}
	else {
		hyperprior_call_ptr = std::bind(&sh_inverse_prior::hyperprior_call_by_hyper_dependence_lengths, this, _1);
	}
	std::function<void(Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> prior_call_ptr;
	hyperprior_call_ptr(cube_m.segment(0, n_stoc));
	std::cout << "stoc hyperparams" << std::endl;
	std::cout << stoc_hyperparams_m << std::endl;
	if (dependence_lengths.size() == 1) {
		prior_call_ptr = std::bind(&sh_inverse_prior::prior_call_ind_same, this, _1, _2);
	}
	else {
		prior_call_ptr = std::bind(&sh_inverse_prior::prior_call_by_dependence_lengths, this, _1, _2);
	}
	prior_call_ptr(cube_m.segment(n_stoc, n_dims), theta_m);
	std::cout << "theta" << std::endl;
	std::cout << theta_m << std::endl;

}