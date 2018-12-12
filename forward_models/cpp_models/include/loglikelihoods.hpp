#pragma once

/* external codebases */
#include <Eigen/Dense>

double calc_gauss_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size);

double calc_ce_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size);

double calc_av_gauss_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size);

double calc_av_ce_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size);

double calc_d_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm, const uint & LL_dim, const uint & batch_size);