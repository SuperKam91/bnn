/* external codebases */
#include <vector>
#include <iostream>
#include <Eigen/Dense>

/* in-house code */
#include "forward.hpp"
#include "interfaces.hpp"

double loglikelihood (double theta[], int nDims, double phi[], int nDerived);
void prior (double cube[], double theta[], int nDims);
void dumper(int,int,int,double*,double*,double*,double, double);
void setup_loglikelihood();

const uint num_inputs = 2;
const uint num_outputs = 2;
const std::vector<uint> layer_sizes = std::vector<uint>{5};
const uint num_weights = calc_num_weights(num_inputs, layer_sizes, num_outputs);
forward_prop slp_nn(2, 2, 6, 6, std::vector<uint>{5}, "./data/scratch_gauss_slp_x.txt", "./data/scratch_gauss_slp_y.txt");

int main() {

  // generate weights manually for test. not needed in production
  //---------------------------------------------------------------------------
  //---------------------------------------------------------------------------
  slp_nn.calc_LL_norm("gauss");


  // PolyChord 1 related stuff
  int nDims, nDerived;
  nDims = 3;
  nDerived = 1;

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

  setup_loglikelihood();
  run_polychord(loglikelihood,prior,dumper,settings) ;


  return 0;
}

double loglikelihood (double theta[], int nDims, double phi[], int nDerived)
{
    double logL=0.0;

    //============================================================
    // insert likelihood code here
    //
    //
    //============================================================
    for (int i=0;i<nDims;i++)
        logL += theta[i]*theta[i];


    Eigen::VectorXd w = Eigen::VectorXd::LinSpaced(num_weights, 0, num_weights - 1);
    slp_nn(w);
    
    return logL;
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
