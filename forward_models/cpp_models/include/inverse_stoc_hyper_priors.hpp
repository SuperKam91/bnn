#pragma once

/* external codebases */
#include <vector>
#include <Eigen/Dense>

class base_prior {
public:
	virtual Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double hyperparam_1, double hyperparam_2) = 0;
	virtual void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double hyperparam_1, double hyperparam_2) = 0;
	//if we consider stochastic hyperparams, will need operator() which accepts these as arguments
	//for derived classes, gets rid of warning: base class has a non-virtual destructor c++.
	//consequences of warning are probably unimportant for my uses
	virtual ~base_prior(); 
};

class uniform_prior: public base_prior {
public:
	uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class pos_log_uniform_prior: public base_prior {
public:
	pos_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class neg_log_uniform_prior: public pos_log_uniform_prior {
public:
	neg_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class log_uniform_prior: public pos_log_uniform_prior {
public:
	log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);	
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class gaussian_prior: public base_prior {
public:
	gaussian_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double sigma);
};

class laplace_prior: public base_prior {
public:
	laplace_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double b);
};

class cauchy_prior: public base_prior {
public:
	cauchy_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double x0, double gamma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double x0, double gamma);
};

class delta_prior: public base_prior {
public:
	delta_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double value, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double value, double sigma);
};

class gamma_prior: public base_prior {
public:
	gamma_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class sqrt_recip_gamma_prior: public gamma_prior {
public:
	sqrt_recip_gamma_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

Eigen::VectorXd forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m);

void forced_identifiability_transform(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m);

void forced_identifiability_transform2(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m);

void forced_identifiability_transform2p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m);

void forced_identifiability_transform3(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

Eigen::VectorXd forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m);

void forced_identifiability_transform3p5(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> t_m);

class sorted_uniform_prior: public uniform_prior {
public:
	sorted_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class sorted_pos_log_uniform_prior: public pos_log_uniform_prior {
public:
	sorted_pos_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class sorted_neg_log_uniform_prior: public neg_log_uniform_prior {
public:
	sorted_neg_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class sorted_log_uniform_prior: public log_uniform_prior {
public:
	sorted_log_uniform_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class sorted_gaussian_prior: public gaussian_prior {
public:
	sorted_gaussian_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double sigma);
};

class sorted_laplace_prior: public laplace_prior {
public:
	sorted_laplace_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double mu, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double mu, double b);
};

class sorted_cauchy_prior: public cauchy_prior {
public:
	sorted_cauchy_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double x0, double gamma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double x0, double gamma);
};

//pointless but included for consistency as in python implementation
class sorted_delta_prior: public delta_prior {
public:
	sorted_delta_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double value, double sigma);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double value, double sigma);
};

class sorted_gamma_prior: public gamma_prior {
public:
	sorted_gamma_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class sorted_sqrt_rec_gam_prior: public sqrt_recip_gamma_prior {
public:
	sorted_sqrt_rec_gam_prior();
	Eigen::VectorXd operator()(Eigen::Ref<Eigen::VectorXd> p_m, double a, double b);
	void operator()(Eigen::Ref<Eigen::VectorXd> p_m, Eigen::Ref<Eigen::VectorXd> theta_m, double a, double b);
};

class inverse_prior {
public:
	inverse_prior(std::vector<uint>, std::vector<double>, std::vector<uint>, std::vector<uint>, uint);
	~inverse_prior();
	void prior_call_by_dependence_lengths(Eigen::Ref<Eigen::VectorXd> & cube_m, Eigen::Ref<Eigen::VectorXd> & theta_m);
	void prior_call_ind_same(Eigen::Ref<Eigen::VectorXd> & cube_m, Eigen::Ref<Eigen::VectorXd> & theta_m);
	void operator()(Eigen::Ref<Eigen::VectorXd> cube_m, Eigen::Ref<Eigen::VectorXd> theta_m);
protected:
	//member variables
	std::vector<uint> prior_types;
	std::vector<double> prior_hyperparams;
	std::vector<uint> dependence_lengths;
	std::vector<uint> param_prior_types;
	uint n_dims;
	std::vector<base_prior *> ppf_ptr_v;
	// unfortunately when I make prior_call_ptr a member variable, get Error in `bin/main': munmap_chunk(): invalid pointer
	// when program exits 
	//member methods
	std::vector<base_prior *> get_ppf_ptr_vec();
};