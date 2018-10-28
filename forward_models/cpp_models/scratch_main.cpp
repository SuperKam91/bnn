/* external codebases */
#include <cmath>
#include <iostream>
#include <Eigen/Core>

/* in-house code */

class Exp_c {
public:
	double operator()(double);
};

double Exp_c::operator()(double x) { 
    return std::exp(x);
}


double Exp(double);
	
double Exp(double x) // the functor we want to apply
{
    return std::exp(x);
}

int main()
{
    Eigen::MatrixXd m(2, 2);
    m << 0, 1, 2, 3;
    std::function<double(double)> exp_wrap = Exp;
    std::cout << m << std::endl << "becomes: ";
    std::cout << std::endl << m.unaryExpr(exp_wrap) << std::endl;
}
