#pragma once

/* external codebases */
#include <string>

/* in-house code */
#include "forward.hpp"
#include "inverse_priors.hpp"
#include "inverse_stoc_hyper_priors.hpp"
#include "inverse_stoc_var_hyper_priors.hpp"

extern const std::string e_batch_type;
extern forward_prop e_nn;
extern const uint e_n_weights;
extern const uint e_n_stoc;
extern const uint e_n_stoc_var;
extern const std::string e_data_dir;
extern const std::string e_data;
extern const std::string e_chains_dir;
extern const std::string e_weights_dir;
extern const std::string e_data_suffix;
extern const std::string e_hyper_type;
extern const std::string e_var_type;
extern inverse_prior e_ip;
extern sh_inverse_prior e_sh_ip;
extern svh_inverse_prior e_svh_ip;