#pragma once

#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXd slp_nn_1(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);
