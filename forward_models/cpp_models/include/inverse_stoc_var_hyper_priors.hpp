#pragma once

/* external codebases */
#include <vector>
#include <Eigen/Dense>

/* in-house code */
#include "inverse_stoc_hyper_priors.hpp"

class svh_inverse_prior: public sh_inverse_prior {
public:
	using sh_inverse_prior::sh_inverse_prior;
	svh_inverse_prior(std::vector<uint>, std::vector<uint>, std::vector<uint>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<uint>, std::vector<uint>, std::vector<uint>, std::vector<uint>, std::vector<uint>, std::vector<uint>, uint, uint, uint);
	~svh_inverse_prior();
	void var_prior_call_by_dependence_lengths(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
	void var_prior_call_ind_same(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
	void operator()(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
protected:
	//member variables
	std::vector<uint> var_prior_types;
	std::vector<double> var_prior_params;
	std::vector<uint> var_dependence_lengths;
	std::vector<uint> var_param_prior_types;
	uint n_stoc_var;
	std::vector<base_prior *> ppf_var_ptr_v;
	//member methods
	std::vector<base_prior *> get_var_ppf_ptr_vec();
};