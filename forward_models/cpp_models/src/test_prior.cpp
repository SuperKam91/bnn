/* external codebases */
#include <iostream>
#include <Eigen/Dense>

/* in-house code */
#include "inverse_priors.hpp"
#include "externs.hpp"
#include "test_prior.hpp"

void test_prior ()
{
	double cube[] = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3};
	double theta[14];
	Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_weights);
    e_ip(cube_m, theta_m);
	std::cout << "theta = " << theta << std::endl;
	for (int i = 0; i < e_n_weights; ++i) {
		std::cout << theta[i] << std::endl;
	}

}