/* external codebases */
#include <Eigen/Dense>
#include <iostream>

/* in-house code */
#include "inverse_priors.hpp"
#include "inverse_stoc_hyper_priors.hpp"
#include "inverse_stoc_var_hyper_priors.hpp"

using namespace std::placeholders;
svh_inverse_prior::svh_inverse_prior(std::vector<uint> hyperprior_types_, std::vector<uint> var_prior_types_, std::vector<uint> prior_types_, std::vector<double> hyperprior_params_, std::vector<double> var_prior_params_, std::vector<double> prior_hyperparams_, std::vector<uint> hyper_dependence_lengths_, std::vector<uint> var_dependence_lengths_, std::vector<uint> dependence_lengths_, std::vector<uint> param_hyperprior_types_, std::vector<uint> var_param_prior_types_, std::vector<uint> param_prior_types_, uint n_stoc_, uint n_stoc_var_, uint n_dims_) :
	sh_inverse_prior::sh_inverse_prior(hyperprior_types_, prior_types_, hyperprior_params_, prior_hyperparams_, hyper_dependence_lengths_, dependence_lengths_, param_hyperprior_types_, param_prior_types_, n_stoc_, n_dims_),
	var_prior_types(var_prior_types_),
	var_prior_params(var_prior_params_),
	var_dependence_lengths(var_dependence_lengths_),
	var_param_prior_types(var_param_prior_types_),
	n_stoc_var(n_stoc_var_),
	ppf_var_ptr_v(get_var_ppf_ptr_vec()) {
}

svh_inverse_prior::~svh_inverse_prior() {
	// these aren't needed here, as they are called automatically in sh destructor
	// for (std::vector<sh_base_prior *>::iterator it = ppf_ptr_v.begin(); it < ppf_ptr_v.end(); it++) {
    	// delete *it;
    // }
    // for (std::vector<base_prior *>::iterator it = ppf_hp_ptr_v.begin(); it < ppf_hp_ptr_v.end(); it++) {
    	// delete *it;
    // }
    for (std::vector<base_prior *>::iterator it = ppf_var_ptr_v.begin(); it < ppf_var_ptr_v.end(); it++) {
    	delete *it;
    }
}

std::vector<base_prior *> svh_inverse_prior::get_var_ppf_ptr_vec() {
	std::vector<base_prior *> ppf_var_ptr_vec;
	ppf_var_ptr_vec.reserve(var_prior_types.size());
	double hyperparam1;
	double hyperparam2;
	for (uint i = 0; i < var_prior_types.size(); ++i) {
		hyperparam1 = var_prior_params.at(2 * i);
		hyperparam2 = var_prior_params.at(2 * i + 1);
		if (var_prior_types.at(i) == 0) {
			ppf_var_ptr_vec.push_back(new uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 1) {
			ppf_var_ptr_vec.push_back(new pos_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 2) {
			ppf_var_ptr_vec.push_back(new neg_log_uniform_prior(hyperparam1, hyperparam2));			
		}
		else if (var_prior_types.at(i) == 3) {
			ppf_var_ptr_vec.push_back(new log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 4) {
			ppf_var_ptr_vec.push_back(new gaussian_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 5) {
			ppf_var_ptr_vec.push_back(new laplace_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 6) {
			ppf_var_ptr_vec.push_back(new cauchy_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 7) {
			ppf_var_ptr_vec.push_back(new delta_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 8) {
			ppf_var_ptr_vec.push_back(new gamma_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 9) {
			ppf_var_ptr_vec.push_back(new sqrt_recip_gamma_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 10) {
			ppf_var_ptr_vec.push_back(new recip_gamma_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 11) {
			ppf_var_ptr_vec.push_back(new sorted_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 12) {
			ppf_var_ptr_vec.push_back(new sorted_pos_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 13) {
			ppf_var_ptr_vec.push_back(new sorted_neg_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 14) {
			ppf_var_ptr_vec.push_back(new sorted_log_uniform_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 15) {
			ppf_var_ptr_vec.push_back(new sorted_gaussian_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 16) {
			ppf_var_ptr_vec.push_back(new sorted_laplace_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 17) {
			ppf_var_ptr_vec.push_back(new sorted_cauchy_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 18) {
			ppf_var_ptr_vec.push_back(new sorted_delta_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 19) {
			ppf_var_ptr_vec.push_back(new sorted_gamma_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 20) {
			ppf_var_ptr_vec.push_back(new sorted_sqrt_rec_gam_prior(hyperparam1, hyperparam2));
		}
		else if (var_prior_types.at(i) == 21) {
			ppf_var_ptr_vec.push_back(new sorted_rec_gam_prior(hyperparam1, hyperparam2));
		}
	}
	return ppf_var_ptr_vec;
}

void svh_inverse_prior::var_prior_call_by_dependence_lengths(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	uint start_ind = 0;
	uint dependence_length;
	for (uint i = 0; i < var_dependence_lengths.size(); ++i) { 
		dependence_length = var_dependence_lengths.at(i);
		(ppf_var_ptr_v.at(var_param_prior_types.at(i)))->operator()(cube_m.segment(start_ind, dependence_length), theta_m.segment(start_ind, dependence_length));
		start_ind += dependence_length;
	}
}

void svh_inverse_prior::var_prior_call_ind_same(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	(ppf_var_ptr_v.at(0))->operator()(cube_m, theta_m);
}

void svh_inverse_prior::operator()(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m) {
	// std::cout << "cube" << std::endl;
	// std::cout << cube_m << std::endl;
	// std::cout << "det hyperparams" << std::endl;
	// std::cout << hyperparams_m << std::endl;
	std::function<void(Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> hyperprior_call_ptr;
	if (hyper_dependence_lengths.size() == 1) {
		hyperprior_call_ptr = std::bind(&sh_inverse_prior::hyperprior_call_ind, this, _1, _2);
	}
	else {
		hyperprior_call_ptr = std::bind(&sh_inverse_prior::hyperprior_call_by_hyper_dependence_lengths, this, _1, _2);
	}
	hyperprior_call_ptr(cube_m.segment(0, n_stoc), theta_m.segment(0, n_stoc));
	// std::cout << "stoc hyperparams" << std::endl;
	// std::cout << stoc_hyperparams_m << std::endl;
	std::function<void(Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> var_prior_call_ptr;
	if (var_dependence_lengths.size() == 1) {
		var_prior_call_ptr = std::bind(&svh_inverse_prior::var_prior_call_ind_same, this, _1, _2);
	}
	else {
		var_prior_call_ptr = std::bind(&svh_inverse_prior::var_prior_call_by_dependence_lengths, this, _1, _2);
	}
	var_prior_call_ptr(cube_m.segment(n_stoc, n_stoc_var), theta_m.segment(n_stoc, n_stoc_var));
	// std::cout << "var params" << std::endl;
	// std::cout << theta_m.segment(n_stoc, n_stoc_var) << std::endl;	
	std::function<void(Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> prior_call_ptr;
	if (dependence_lengths.size() == 1) {
		prior_call_ptr = std::bind(&sh_inverse_prior::prior_call_ind_same, this, _1, _2);
	}
	else {
		prior_call_ptr = std::bind(&sh_inverse_prior::prior_call_by_dependence_lengths, this, _1, _2);
	}
	prior_call_ptr(cube_m.segment(n_stoc + n_stoc_var, n_dims), theta_m.segment(n_stoc + n_stoc_var, n_dims));
	// std::cout << "full params" << std::endl;
	// std::cout << theta_m << std::endl;
}