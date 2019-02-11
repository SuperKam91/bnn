#pragma once

/* external codebases */
#include <vector>
#include <Eigen/Dense>

class sh_base_prior {
public:
	virtual Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double hyperparam_1, double hyperparam_2) = 0;
	virtual void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double hyperparam_1, double hyperparam_2) = 0;
	virtual void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> hyperparam_1,Eigen::Ref<Eigen::VectorXd> hyperparam_2) = 0;
	//if we consider stochastic hyperparams, will need operator() which accepts these as arguments
	//for derived classes, gets rid of warning: base class has a non-virtual destructor c++.
	//consequences of warning are probably unimportant for my uses
	virtual ~sh_base_prior(); 
};

class sh_uniform_prior: public sh_base_prior {
public:
	sh_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_pos_log_uniform_prior: public sh_base_prior {
public:
	sh_pos_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_neg_log_uniform_prior: public sh_pos_log_uniform_prior {
public:
	sh_neg_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_log_uniform_prior: public sh_pos_log_uniform_prior {
public:
	sh_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);	
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_gaussian_prior: public sh_base_prior {
public:
	sh_gaussian_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> sigma);
};

class sh_laplace_prior: public sh_base_prior {
public:
	sh_laplace_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_cauchy_prior: public sh_base_prior {
public:
	sh_cauchy_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double x0, double gamma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double x0, double gamma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> gamma);
};

class sh_delta_prior: public sh_base_prior {
public:
	sh_delta_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double value, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double value, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> value, Eigen::Ref<Eigen::VectorXd> sigma);
};

class sh_gamma_prior: public sh_base_prior {
public:
	sh_gamma_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sqrt_recip_gamma_prior: public sh_gamma_prior {
public:
	using sh_gamma_prior::sh_gamma_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

//n.b. following are same as ones inplemented in inverse_priors.cpp, just included for completeness
//------------------------------------------------------------------------------------------------------------

Eigen::VectorXd sh_forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m);

void sh_forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd sh_forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m);

void sh_forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd sh_forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m);

void sh_forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd sh_forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m);

void sh_forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd sh_forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m);

void sh_forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

//------------------------------------------------------------------------------------------------------------

class sh_sorted_uniform_prior: public sh_uniform_prior {
public:
	using sh_uniform_prior::sh_uniform_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sorted_pos_log_uniform_prior: public sh_pos_log_uniform_prior {
public:
	using sh_pos_log_uniform_prior::sh_pos_log_uniform_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sorted_neg_log_uniform_prior: public sh_neg_log_uniform_prior {
public:
	using sh_neg_log_uniform_prior::sh_neg_log_uniform_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sorted_log_uniform_prior: public sh_log_uniform_prior {
public:
	using sh_log_uniform_prior::sh_log_uniform_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sorted_gaussian_prior: public sh_gaussian_prior {
public:
	using sh_gaussian_prior::sh_gaussian_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> sigma);
};

class sh_sorted_laplace_prior: public sh_laplace_prior {
public:
	using sh_laplace_prior::sh_laplace_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> mu, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sorted_cauchy_prior: public sh_cauchy_prior {
public:
	using sh_cauchy_prior::sh_cauchy_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double x0, double gamma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double x0, double gamma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> gamma);
};

//pointless but included for consistency as in python implementation
class sh_sorted_delta_prior: public sh_delta_prior {
public:
	using sh_delta_prior::sh_delta_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double value, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double value, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> value, Eigen::Ref<Eigen::VectorXd> sigma);
};

class sh_sorted_gamma_prior: public sh_gamma_prior {
public:
	using sh_gamma_prior::sh_gamma_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_sorted_sqrt_rec_gam_prior: public sh_sqrt_recip_gamma_prior {
public:
	using sh_sqrt_recip_gamma_prior::sh_sqrt_recip_gamma_prior;
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, Eigen::Ref<Eigen::VectorXd> a, Eigen::Ref<Eigen::VectorXd> b);
};

class sh_inverse_prior {
public:
	sh_inverse_prior(std::vector<uint>, std::vector<uint>, std::vector<double>, std::vector<double>, std::vector<uint>, std::vector<uint>, std::vector<uint>, std::vector<uint>, uint, uint);
	~sh_inverse_prior();
	Eigen::VectorXd init_stoc_hypers();
	Eigen::VectorXd fill_det_hypers();
	void hyperprior_call_ind(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
	void hyperprior_call_by_hyper_dependence_lengths(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
	void prior_call_by_dependence_lengths(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
	void prior_call_ind_same(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
	void operator()(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
protected:
	//member variables
	std::vector<uint> hyperprior_types;
	std::vector<uint> prior_types;
	std::vector<double> hyperprior_params;
	std::vector<double> prior_hyperparams;
	std::vector<uint> hyper_dependence_lengths;
	std::vector<uint> dependence_lengths;
	std::vector<uint> param_hyperprior_types;
	std::vector<uint> param_prior_types;
	uint n_stoc;
	uint n_dims;
	std::vector<base_prior *> ppf_hp_ptr_v;
	std::vector<sh_base_prior *> ppf_ptr_v;
	Eigen::VectorXd hyperparams_m; //deterministic hyperparams
	Eigen::VectorXd stoc_hyperparams_m; //stochastic hyperparams
	//member methods
	std::vector<base_prior *> get_hp_ppf_ptr_vec();
	std::vector<sh_base_prior *> get_ppf_ptr_vec();
};
