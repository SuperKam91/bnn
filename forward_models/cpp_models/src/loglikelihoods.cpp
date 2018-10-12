/* external codebase */
#include <Eigen/Dense>

/* in-house code */
#include "loglikelihoods.hpp"

double calc_gauss_ll(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm) {
    const double chi_sq = -1. / (2. * LL_var) * (pred - y).squaredNorm();
    return chi_sq + LL_norm;
}

double calc_ce_ll(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm) {
    return -1. * (pred.array().log() * y.array()).sum();
}