/* external codebases */
#include <string>
#include <iostream>

/* in-house code */
#include "polychord_interfaces.hpp"
#include "externs.hpp"


void run_polychord_wrap() {
    // PolyChord 1 related stuff
    int nDims, nDerived;
    nDims = 27;
    nDerived = 0;

    Settings settings(nDims,nDerived);

    settings.nlive         = 50;
    settings.num_repeats   = settings.nDims*5;
    settings.do_clustering = false;

    settings.precision_criterion = 1e-3;
    settings.logzero = -1e30;

    settings.base_dir      = "chains";
    settings.file_root     = "test";

    settings.write_resume  = false;
    settings.read_resume   = false;
    settings.write_live    = true;
    settings.write_dead    = false;
    settings.write_stats   = false;

    settings.equals        = false;
    settings.posteriors    = false;
    settings.cluster_posteriors = false;

    settings.feedback      = 1;
    settings.compression_factor  = 0.36787944117144233;

    settings.boost_posterior= 5.0;


    run_polychord(loglikelihood,prior,dumper,settings);
}

double loglikelihood (double theta[], int nDims, double phi[], int nDerived)
{

    //============================================================
    // insert likelihood code here
    //
    //
    //============================================================
    //for (int i=0;i<nDims;i++)
    //    logL += theta[i]*theta[i];
    std::cout << theta << std::endl;

    //creates eigen vector object from array without making a copy
    Eigen::VectorXd w;
    new (&w) Eigen::Map<Eigen::VectorXd> (theta, nDims);
    std::cout << g_slp_nn(w) << std::endl;
    exit(0);
    return g_slp_nn(w);;
}


void prior (double cube[], double theta[], int nDims)
{
    //============================================================
    // insert prior code here
    //
    //
    //============================================================
    for(int i=0;i<nDims;i++)
        theta[i] = cube[i];

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