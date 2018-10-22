#pragma once

/* external codebases */
#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXd slp_nn_1(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);

Eigen::MatrixXd slp_nn_old(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);
