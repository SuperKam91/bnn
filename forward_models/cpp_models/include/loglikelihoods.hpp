#pragma once

#include <Eigen/Dense>

double calc_gauss_ll(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm);

double calc_ce_ll(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred, double & LL_var, double & LL_norm);

