/* external codebases */
//#include <string>
#include <Eigen/Dense>
#include <string>

/* in-house code */
#include "polychord_interfaces.hpp"
#include "externs.hpp"


void run_polychord_wrap(bool profiling) {
    //some of default settings changed to sensible values by kj
    int nDims, nDerived;
    nDims = static_cast<int>(e_n_weights + e_n_stoc + e_n_stoc_var);
    nDerived = 0;

    Settings settings(nDims,nDerived);

    //default nlive is 50
    settings.nlive         = 1000;
    settings.num_repeats   = settings.nDims*5;
    //set to true by kj
    settings.do_clustering = true;    
    //settings.do_clustering = false;

    settings.precision_criterion = 1e-3;
    settings.logzero = -1e30;

    settings.base_dir      = e_chains_dir;
    settings.file_root     = e_data + e_data_suffix;
    //just write .txt output for now, kj
    settings.write_resume  = true;
    settings.read_resume   = true;
    settings.write_live    = false;
    settings.write_dead    = false;
    //added by kj
    settings.write_prior = false;
    //changed by kj
    if (profiling) {
        settings.write_stats   = false;
    }
    else {
        settings.write_stats   = true;        
    }

    settings.equals        = false;
    //changed by kj
    if (profiling) {
        settings.posteriors    = false;
    }
    else {
        settings.posteriors    = true;
    }
    // settings.posteriors    = false;
    settings.cluster_posteriors = false;
    if (profiling) {
        settings.feedback     = 0; //suppress output for profiling    
    }
    else {
        settings.feedback      = 1;        
    }
    settings.compression_factor  = 0.36787944117144233;

    //don't boost posterior kj
    settings.boost_posterior= 0.0;
    //settings.boost_posterior= 5.0;

    if ((e_hyper_type == "deterministic") && (e_var_type == "deterministic")) {
        run_polychord(loglikelihood,prior,dumper,settings);
    }
    else if ((e_hyper_type == "stochastic") && (e_var_type == "deterministic")) {
        run_polychord(loglikelihood,sh_prior,dumper,settings);
    }
    else if ((e_hyper_type == "stochastic") && (e_var_type == "stochastic")) {
        run_polychord(loglikelihood,svh_prior,dumper,settings);
    }
}

double loglikelihood (double theta[], int nDims, double phi[], int nDerived)
{
    Eigen::Map<Eigen::VectorXd> w_m(theta, e_n_stoc + e_n_stoc_var + e_n_weights);
    return e_nn(w_m.segment(e_n_stoc, e_n_stoc_var + e_n_weights));
}

void prior (double cube[], double theta[], int nDims)
{
    Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_weights);
    e_ip(cube_m, theta_m);
}

void sh_prior (double cube[], double theta[], int nDims)
{
    Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_stoc + e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_stoc + e_n_weights);
    e_sh_ip(cube_m, theta_m);
}

void svh_prior (double cube[], double theta[], int nDims)
{
    Eigen::Map<Eigen::VectorXd> cube_m(cube, e_n_stoc + e_n_stoc_var + e_n_weights);
    Eigen::Map<Eigen::VectorXd> theta_m(theta, e_n_stoc + e_n_stoc_var + e_n_weights);
    e_svh_ip(cube_m, theta_m);
}


void setup_loglikelihood()
{
    //============================================================
    // insert likelihood setup here
    //
    //
    //============================================================
}

void dumper(int ndead,int nlive,int npars,double* live,double* dead,double* logweights,double logZ, double logZerr)
{
}