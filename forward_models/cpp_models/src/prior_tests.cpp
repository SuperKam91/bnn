/* external codebases */
#include <iostream>
#include <Eigen/Dense>

/* in-house code */
#include "inverse_priors.hpp"
#include "externs.hpp"
#include "prior_tests.hpp"

//test prior set up (externally) for forward propagation of nn. cube should be manually set to have same dimensionality as nn
void nn_prior_test(bool print_out) {
	double cube[] = {0.1, 0.9};
	double theta[2];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_weights);
    e_ip(cube_m, theta_m);
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
	double cube[] = {0.1, 0.9};
	double theta[2];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, n_weights);
    ip(cube_m, theta_m);
	if (print_out) {
		std::cout << "for the hypercube = " << cube_m << std::endl;    
		std::cout << "theta = " << theta_m << std::endl;
	}
}