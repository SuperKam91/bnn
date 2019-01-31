/* external codebases */
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <gsl/gsl_sf_bessel.h>

#include <algorithm> // for copy
#include <iterator> // for ostream_iterator

/* in-house code */

//super useful line of code to print out vector in one line:
//std::copy(*vec*.begin(), *vec*.end(), std::ostream_iterator<*vec type*>(std::cout, *delimiter*));


class Exp_c {
public:
	double operator()(double);
	// void operator()(double);
};

double Exp_c::operator()(double x) { 
    return std::exp(x);
}

// void Exp_c::operator()(double x) { 
//      std::cout << std::exp(x) << std::endl;
// }



double Exp(double);
	
double Exp(double x) // the functor we want to apply
{
    return std::exp(x);
}

int main() {
	double x = 5.0;
    double y = gsl_sf_bessel_J0 (x);
    printf ("J0(%g) = %.18e\n", x, y);
	return 0;
}
