/* external codebase */
#include <Eigen/Dense>

/* in-house code */
#include "loglikelihoods.hpp"

double calc_gauss_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size) {
    const double chi_sq = -1. / (2. * LL_var) * (pred - y).squaredNorm();
    return chi_sq + LL_norm;
}

double calc_ce_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size) {
    return (pred.array().log() * y.array()).sum();
}

double calc_av_gauss_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size) {
    const double chi_sq = -1. / (2. * LL_var * LL_dim) * (pred - y).squaredNorm();
    return chi_sq + LL_norm;
}

double calc_av_ce_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size) {
    LL_norm = -1. * std::log(pred.array().pow(1. / batch_size).colwise().prod().sum());
    return (pred.array().log() * y.array()).sum() / batch_size + LL_norm;
}

//simply returns sum of predictions (nn outputs). for testing purposes only
double calc_d_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size) {
	return pred.sum();
}