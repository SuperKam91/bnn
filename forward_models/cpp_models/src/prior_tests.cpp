/* external codebases */
#include <iostream>
#include <Eigen/Dense>
#include <string>

/* in-house code */
#include "inverse_priors.hpp"
#include "inverse_stoc_hyper_priors.hpp"
#include "inverse_stoc_var_hyper_priors.hpp"
#include "externs.hpp"
#include "prior_tests.hpp"
#include "tools.hpp"

//test prior set up (externally) for forward propagation of nn. cube should be manually set to have same dimensionality as nn
void nn_prior_test(bool print_out) {
	double cube[] = {0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7};
	double theta[60];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_weights);
    e_ip(cube_m, theta_m);
	if (print_out) {
		std::cout << "for the hypercube = " << cube_m << std::endl;    
		std::cout << "theta = " << theta_m << std::endl;
	}
}

void nn_sh_prior_test(bool print_out) {
	double cube[] = {0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7};
	double theta[60];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_stoc + e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_stoc + e_n_weights);
    e_sh_ip(cube_m, theta_m);
	if (print_out) {
		std::cout << "for the hypercube = " << cube_m << std::endl;    
		std::cout << "theta = " << theta_m << std::endl;
	}
}

void nn_sv_sh_prior_test(bool print_out) {
	double cube[] = {0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7};
	double theta[60];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_stoc + e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_stoc + e_n_weights);
    e_svh_ip(cube_m, theta_m);
	if (print_out) {
		std::cout << "for the hypercube = " << cube_m << std::endl;    
		std::cout << "theta = " << theta_m << std::endl;
	}
}

//test prior from setting all prior parameters manually
void prior_test(bool print_out) {
	uint n_weights = 2;
	std::vector<uint> dependence_lengths = {2};
	std::vector<uint> prior_types = {7};
	std::vector<double> prior_hyperparams = {-2., 2.};
	std::vector<uint> param_prior_types = {0};
	inverse_prior ip(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, n_weights);
	double cube[] = {0.1, 0.6};
	double theta[2];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, n_weights);
    ip(cube_m, theta_m);
	if (print_out) {
		std::cout << "for the hypercube = " << cube_m << std::endl;    
		std::cout << "theta = " << theta_m << std::endl;
	}
}

void prior_functions_test(bool print_out) {
	double cube[] = {0.1, 0.6, 0.8};
	double theta[3];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, 3);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, 3);
    double hyperparam1 = 1.;
    double hyperparam2 = 4.;
	uniform_prior u(hyperparam1, hyperparam2);
	pos_log_uniform_prior plu(hyperparam1, hyperparam2);
	neg_log_uniform_prior nlu(hyperparam1, hyperparam2);			
	log_uniform_prior lu(hyperparam1, hyperparam2);
	gaussian_prior g(hyperparam1, hyperparam2);
	laplace_prior l(hyperparam1, hyperparam2);
	cauchy_prior c(hyperparam1, hyperparam2);
	delta_prior d(hyperparam1, hyperparam2);
	gamma_prior ga(hyperparam1, hyperparam2);
	sqrt_recip_gamma_prior srga(hyperparam1, hyperparam2);
	sorted_uniform_prior su(hyperparam1, hyperparam2);
	sorted_pos_log_uniform_prior splu(hyperparam1, hyperparam2);
	sorted_neg_log_uniform_prior snlu(hyperparam1, hyperparam2);
	sorted_log_uniform_prior slu(hyperparam1, hyperparam2);
	sorted_gaussian_prior sg(hyperparam1, hyperparam2);
	sorted_laplace_prior sl(hyperparam1, hyperparam2);
	sorted_cauchy_prior sc(hyperparam1, hyperparam2);
	sorted_delta_prior sd(hyperparam1, hyperparam2);
	sorted_gamma_prior sga(hyperparam1, hyperparam2);
	sorted_sqrt_rec_gam_prior ssrga(hyperparam1, hyperparam2);
	if (print_out) {
		std::cout << "using call operators which return a theta eigen vec" << std::endl;   
		std::cout << u(cube_m) << std::endl;
		std::cout << plu(cube_m) << std::endl;
		std::cout << nlu(cube_m) << std::endl;
		std::cout << lu(cube_m) << std::endl;
		std::cout << g(cube_m) << std::endl;
		std::cout << l(cube_m) << std::endl;
		std::cout << c(cube_m) << std::endl;
		std::cout << d(cube_m) << std::endl;
		std::cout << ga(cube_m) << std::endl;
		std::cout << srga(cube_m) << std::endl;
		std::cout << su(cube_m) << std::endl;
		std::cout << splu(cube_m) << std::endl;
		std::cout << snlu(cube_m) << std::endl;
		std::cout << slu(cube_m) << std::endl;
		std::cout << sg(cube_m) << std::endl;
		std::cout << sl(cube_m) << std::endl;
		std::cout << sc(cube_m) << std::endl;
		std::cout << sd(cube_m) << std::endl;
		std::cout << sga(cube_m) << std::endl;
		std::cout << ssrga(cube_m) << std::endl;
		std::cout << "using call operators which take theta eigen vec as arg" << std::endl;
		u(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		plu(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		nlu(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		lu(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		g(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		l(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		c(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		d(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		ga(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		srga(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		su(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		splu(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		snlu(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		slu(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		sg(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		sl(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		sc(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		sd(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		sga(cube_m, theta_m);
		std::cout << theta_m << std::endl;
		ssrga(cube_m, theta_m);
		std::cout << theta_m << std::endl;
	}
}

void stoc_prior_functions_test(bool print_out) {
	double cube[] = {0.1, 0.6, 0.8};
	double theta[3];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, 3);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, 3);
    double hyperparam1 = 1.;
    double hyperparam2 = 4.;
    Eigen::VectorXd hyperparam1_m(3);
    hyperparam1_m << hyperparam1, hyperparam1, hyperparam1;
    Eigen::VectorXd hyperparam2_m(3);
    hyperparam2_m << hyperparam2, hyperparam2, hyperparam2;
    sh_uniform_prior u;
	sh_pos_log_uniform_prior plu;
	sh_neg_log_uniform_prior nlu;			
	sh_log_uniform_prior lu;
	sh_gaussian_prior g;
	sh_laplace_prior l;
	sh_cauchy_prior c;
	sh_delta_prior d;
	sh_gamma_prior ga;
	sh_sqrt_recip_gamma_prior srga;
	sh_sorted_uniform_prior su;
	sh_sorted_pos_log_uniform_prior splu;
	sh_sorted_neg_log_uniform_prior snlu;
	sh_sorted_log_uniform_prior slu;
	sh_sorted_gaussian_prior sg;
	sh_sorted_laplace_prior sl;
	sh_sorted_cauchy_prior sc;
	sh_sorted_delta_prior sd;
	sh_sorted_gamma_prior sga;
	sh_sorted_sqrt_rec_gam_prior ssrga;
	if (print_out) {
		std::cout << "using call operators which return a theta eigen vec" << std::endl;   
		std::cout << u(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << plu(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << nlu(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << lu(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << g(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << l(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << c(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << d(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << ga(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << srga(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << su(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << splu(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << snlu(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << slu(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << sg(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << sl(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << sc(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << sd(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << sga(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << ssrga(cube_m, hyperparam1, hyperparam2) << std::endl;
		std::cout << "using call operators which take theta eigen vec as arg" << std::endl;
		u(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		plu(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		nlu(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		lu(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		g(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		l(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		c(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		d(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		ga(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		srga(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		su(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		splu(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		snlu(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		slu(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		sg(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		sl(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		sc(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		sd(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		sga(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		ssrga(cube_m, theta_m, hyperparam1, hyperparam2);
		std::cout << theta_m << std::endl;
		std::cout << "using call operators which take theta eigen vec as arg and hyperparams as eigen vecs" << std::endl;
		u(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		plu(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		nlu(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		lu(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		g(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		l(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		c(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		d(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		ga(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		srga(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		su(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		splu(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		snlu(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		slu(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		sg(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		sl(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		sc(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		sd(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		sga(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
		ssrga(cube_m, theta_m, hyperparam1_m, hyperparam2_m);
		std::cout << theta_m << std::endl;
	}
}

void inverse_stoc_hyper_priors_test1() {
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0.};
	std::vector<uint> hyper_dependence_lengths = {1};
	std::vector<uint> dependence_lengths = {1};
	std::vector<uint> param_hyperprior_types = {0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = 1;
	uint n_dims = 1;
	Eigen::VectorXd cube_m(2);
    cube_m << 0.1, 0.6;
    Eigen::VectorXd theta_m(2);
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
}

void inverse_stoc_hyper_priors_test2() {
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0.};
	std::vector<uint> hyper_dependence_lengths = {1};
	std::vector<uint> dependence_lengths = {1};
	std::vector<uint> param_hyperprior_types = {0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = 1;
	uint n_dims = 2;
	Eigen::VectorXd cube_m(3);
    cube_m << 0.1, 0.6, 0.7;
    Eigen::VectorXd theta_m(3);
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
}

void inverse_stoc_hyper_priors_test3() {
	std::vector<uint> hyperprior_types = {9, 7};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2., 1., 1.};
	std::vector<double> prior_hyperparams = {0.};
	std::vector<uint> hyper_dependence_lengths = {1, 1};
	std::vector<uint> dependence_lengths = {1};
	std::vector<uint> param_hyperprior_types = {0, 1};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = 2;
	uint n_dims = 2;
	Eigen::VectorXd cube_m(4);
    cube_m << 0.1, 0.5, 0.6, 0.7;
    Eigen::VectorXd theta_m(4);
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
}

void inverse_stoc_hyper_priors_test4() {
	std::vector<uint> hyperprior_types = {9, 7, 9};
	std::vector<uint> prior_types = {4, 5};
	std::vector<double> hyperprior_params = {1., 2., 1., 1., 1., 5.};
	std::vector<double> prior_hyperparams = {0., 1.};
	std::vector<uint> hyper_dependence_lengths = {1, 2, 1};
	std::vector<uint> dependence_lengths = {1, 3};
	std::vector<uint> param_hyperprior_types = {0, 1, 2};
	std::vector<uint> param_prior_types = {0, 1};
	uint n_stoc = 3;
	uint n_dims = 4;
	Eigen::VectorXd cube_m(7);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4;
    Eigen::VectorXd theta_m(7);
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
}

void inverse_stoc_hyper_priors_test5() {
	std::vector<uint> hyperprior_types = {9, 7, 9, 7};
	std::vector<uint> prior_types = {4, 5, 4};
	std::vector<double> hyperprior_params = {1., 2., 1., 1., 1., 5., 3., 1.};
	std::vector<double> prior_hyperparams = {0., 1., 2.};
	std::vector<uint> hyper_dependence_lengths = {1, 3, 1, 1};
	std::vector<uint> dependence_lengths = {1, 3, 2};
	std::vector<uint> param_hyperprior_types = {1, 0, 3, 2};
	std::vector<uint> param_prior_types = {1, 2, 0};
	uint n_stoc = 4;
	uint n_dims = 6;
	Eigen::VectorXd cube_m(10);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5;
    Eigen::VectorXd theta_m(10);
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
}

void inverse_stoc_hyper_priors_test6() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0.};
	std::string granularity = "single";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = true;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test7() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4, 5};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0., 1};
	std::string granularity = "single";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = false;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0};
	std::vector<uint> param_prior_types = {0, 1, 0, 1, 0, 0, 1, 1, 0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test8() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0.};
	std::string granularity = "layer";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = true;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0, 0, 0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test9() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9, 7};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2., 2., 0.};
	std::vector<double> prior_hyperparams = {0.};
	std::string granularity = "layer";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = true;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {1, 0, 1};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test10() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9, 7};
	std::vector<uint> prior_types = {4, 5};
	std::vector<double> hyperprior_params = {1., 2., 2., 0.};
	std::vector<double> prior_hyperparams = {0., 1.};
	std::string granularity = "layer";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = false;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {1, 0, 1};
	std::vector<uint> param_prior_types = {0, 1, 0, 1, 0, 0, 1, 1, 0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test11() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0.};
	std::string granularity = "input_size";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = true;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test12() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9, 7};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2., 2., 0.};
	std::vector<double> prior_hyperparams = {0.};
	std::string granularity = "input_size";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = true;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0, 1, 0, 0, 1, 1, 1, 0, 0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test13() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> prior_types = {4, 5};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0., 1.};
	std::string granularity = "input_size";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = false;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<uint> param_prior_types = {0, 1, 0, 1, 0, 0, 1, 1, 0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_hyper_priors_test14() {
	const uint n_inputs = 1;
	const uint n_outputs = 1;
	const std::vector<uint> l_sizes = {3,2};
	const uint n_weights = calc_num_weights(n_inputs, l_sizes, n_outputs);
	std::vector<uint> weight_s = get_weight_shapes(n_inputs, l_sizes, n_outputs);
	std::vector<uint> num_weights_layers = calc_num_weights_layers(weight_s);
	std::vector<uint> hyperprior_types = {9, 7};
	std::vector<uint> prior_types = {4, 5};
	std::vector<double> hyperprior_params = {1., 2., 2., 0.};
	std::vector<double> prior_hyperparams = {0., 1.};
	std::string granularity = "input_size";
	std::vector<uint> hyper_dependence_lengths = get_hyper_dependence_lengths(weight_s, granularity);
	bool independent = false;
	std::vector<uint> dependence_lengths = get_degen_dependence_lengths(weight_s, independent);
	std::vector<uint> param_hyperprior_types = {0, 1, 0, 0, 1, 1, 1, 0, 0};
	std::vector<uint> param_prior_types = {0, 1, 0, 1, 0, 0, 1, 1, 0};
	uint n_stoc = static_cast<uint>(hyper_dependence_lengths.size());
	uint n_dims = n_weights;
	Eigen::VectorXd cube_m(n_stoc + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.6, 0.8;
    Eigen::VectorXd theta_m(n_stoc + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	sh_inverse_prior prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_var_hyper_priors_test1() {
	std::vector<uint> hyperprior_types = {9};
	std::vector<uint> var_prior_types = {10};
	std::vector<uint> prior_types = {4};
	std::vector<double> hyperprior_params = {1., 2.};
	std::vector<double> var_prior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0.};
	std::vector<uint> hyper_dependence_lengths = {1};
	std::vector<uint> var_dependence_lengths = {1};
	std::vector<uint> dependence_lengths = {1};
	std::vector<uint> param_hyperprior_types = {0};
	std::vector<uint> var_param_prior_types = {0};
	std::vector<uint> param_prior_types = {0};
	uint n_stoc = 1;
	uint n_stoc_var = 1;
	uint n_dims = 1;
	Eigen::VectorXd cube_m(n_stoc + n_stoc_var + n_dims);
    cube_m << 0.1, 0.1, 0.6;
    Eigen::VectorXd theta_m(n_stoc + n_stoc_var + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	svh_inverse_prior prior(hyperprior_types, var_prior_types, prior_types, hyperprior_params, var_prior_params, prior_hyperparams, hyper_dependence_lengths, var_dependence_lengths, dependence_lengths, param_hyperprior_types, var_param_prior_types, param_prior_types, n_stoc, n_stoc_var, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_var_hyper_priors_test2() {
	std::vector<uint> hyperprior_types = {9, 7, 9, 7};
	std::vector<uint> var_prior_types = {10};
	std::vector<uint> prior_types = {4, 5, 4};
	std::vector<double> hyperprior_params = {1., 2., 1., 1., 1., 5., 3., 1.};
	std::vector<double> var_prior_params = {1., 2.};
	std::vector<double> prior_hyperparams = {0., 1., 2.};
	std::vector<uint> hyper_dependence_lengths = {1, 3, 1, 1};
	std::vector<uint> var_dependence_lengths = {1};
	std::vector<uint> dependence_lengths = {1, 3, 2};
	std::vector<uint> param_hyperprior_types = {1, 0, 3, 2};
	std::vector<uint> var_param_prior_types = {0};
	std::vector<uint> param_prior_types = {1, 2, 0};
	uint n_stoc = 4;
	uint n_stoc_var = 1;
	uint n_dims = 6;
	Eigen::VectorXd cube_m(n_stoc + n_stoc_var + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.5, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5;
    Eigen::VectorXd theta_m(n_stoc + n_stoc_var + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	svh_inverse_prior prior(hyperprior_types, var_prior_types, prior_types, hyperprior_params, var_prior_params, prior_hyperparams, hyper_dependence_lengths, var_dependence_lengths, dependence_lengths, param_hyperprior_types, var_param_prior_types, param_prior_types, n_stoc, n_stoc_var, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}

void inverse_stoc_var_hyper_priors_test3() {
	std::vector<uint> hyperprior_types = {9, 7, 9, 7};
	std::vector<uint> var_prior_types = {10, 7};
	std::vector<uint> prior_types = {4, 5, 4};
	std::vector<double> hyperprior_params = {1., 2., 1., 1., 1., 5., 3., 1.};
	std::vector<double> var_prior_params = {1., 2., 3., 3.};
	std::vector<double> prior_hyperparams = {0., 1., 2.};
	std::vector<uint> hyper_dependence_lengths = {1, 3, 1, 1};
	std::vector<uint> var_dependence_lengths = {1, 1};
	std::vector<uint> dependence_lengths = {1, 3, 2};
	std::vector<uint> param_hyperprior_types = {1, 0, 3, 2};
	std::vector<uint> var_param_prior_types = {1, 0};
	std::vector<uint> param_prior_types = {1, 2, 0};
	uint n_stoc = 4;
	uint n_stoc_var = 2;
	uint n_dims = 6;
	Eigen::VectorXd cube_m(n_stoc + n_stoc_var + n_dims);
    cube_m << 0.1, 0.5, 0.6, 0.7, 0.9, 0.5, 0.8, 0.9, 0.4, 0.2, 0.1, 0.5;
    Eigen::VectorXd theta_m(n_stoc + n_stoc_var + n_dims);
    std::cout << "addresses before call" << cube_m.data() << " and " << theta_m.data() << std::endl;
	svh_inverse_prior prior(hyperprior_types, var_prior_types, prior_types, hyperprior_params, var_prior_params, prior_hyperparams, hyper_dependence_lengths, var_dependence_lengths, dependence_lengths, param_hyperprior_types, var_param_prior_types, param_prior_types, n_stoc, n_stoc_var, n_dims);
    prior(cube_m, theta_m);
    std::cout << "addresses after call" << cube_m.data() << " and " << theta_m.data() << std::endl;
}