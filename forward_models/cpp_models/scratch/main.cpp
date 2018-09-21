#include <iostream>
#include <Eigen/Dense>

#include "example.hpp"

int main(int argc, char *argv[])
{
    double p = 0;
    std::cout << example_function(p) << std::endl << std::endl;

    Eigen::VectorXd x(3);
    x << 1, 2, 3;

    Eigen::MatrixXd M(3,3);
    M << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    std::cout << M   << std::endl 
              << "x" << std::endl 
              << x   << std::endl 
              << "=" << std::endl
              << M*x << std::endl;
}
