/* external codebases */
//#include <string>
#include <Eigen/Dense>

/* in-house code */
#include "polychord_interfaces.hpp"
#include "externs.hpp"


void run_polychord_wrap() {
    //some of default settings changed to sensible values by kj
    int nDims, nDerived;
    nDims = static_cast<int>(e_n_weights);
    nDerived = 0;

    Settings settings(nDims,nDerived);

    //default nlive is 50
    settings.nlive         = 200;
    settings.num_repeats   = settings.nDims*5;
    //set to true by kj
    settings.do_clustering = true;    
    //settings.do_clustering = false;

    settings.precision_criterion = 1e-3;
    settings.logzero = -1e30;

    settings.base_dir      = e_chains_dir;
    settings.file_root     = e_data;
    //just write .txt output for now, kj
    settings.write_resume  = false;
    settings.read_resume   = false;
    settings.write_live    = false;
    settings.write_dead    = false;
    //added by kj
    settings.write_prior = false;
    //changed by kj
    settings.write_stats   = true;
    // settings.write_stats   = false;

    settings.equals        = false;
    //changed by kj
    settings.posteriors    = true;
    // settings.posteriors    = false;
    settings.cluster_posteriors = false;

    settings.feedback      = 1;
    settings.compression_factor  = 0.36787944117144233;

    //don't boost posterior kj
    settings.boost_posterior= 0.0;
    //settings.boost_posterior= 5.0;


    run_polychord(loglikelihood,prior,dumper,settings);
}

double loglikelihood (double theta[], int nDims, double phi[], int nDerived)
{

    //============================================================
    // insert likelihood code here
    //
    //
    //============================================================
    Eigen::Map<Eigen::VectorXd> w_m(theta, e_n_weights);
    return e_slp_nn(w_m);
}


void prior (double cube[], double theta[], int nDims)
{
    //============================================================
    // insert prior code here
    //
    //
    //============================================================
    for(int i=0;i<nDims;i++)
        theta[i] = 2 * (cube[i] - 0.5);

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