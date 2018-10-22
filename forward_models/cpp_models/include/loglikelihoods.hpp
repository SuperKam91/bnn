#pragma once

/* external codebases */
#include <Eigen/Dense>

double calc_gauss_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , Eigen::MatrixXd & pred, double & LL_var, double & LL_norm);

double calc_ce_ll(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm);

