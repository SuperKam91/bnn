/* external codebases */
#include <cmath>
#include <limits>
#include <Eigen/Dense>

/* in-house code */
#include "mathematics.hpp"

//the following is taken from:
//http://libit.sourceforge.net/math_8c-source.html
//i.e. the libit library.
//see webpage for copyright and acknowledgements
//presumably this is an extension of cmath
//------------------------------------------------------------------------------------

#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899
#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1
#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454
#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1

double erfinv (double x) {
	double x2, r, y;
    double sign_x = sgn(x);
    if (sign_x == -1.) {
    	x = -x;
    }
	if ((x < -1.) || (x > 1.)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	else if (x == 0.) {
		return 0.;
	}
    else if (x <= 0.7) {
	    x2 = x * x;
	    r =
	    x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
	    r /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 +
	    erfinv_b1) * x2 + erfinv_b0;
    }
    else {
	    y = sqrt (-log((1 - x) / 2));
	    r = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
	    r /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
    }
	r *= sign_x;
	x *= sign_x;
	r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));
	r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r)); 
   	return r;
}

//------------------------------------------------------------------------------------

double sgn(double x) {
	if (x > 0.) {
        return 1.;
    }
    else if (x == 0.) {
    	return 0.;
    }
    else { 
        return -1.;
    }
} 

double inv_erf(double x){
	return erfinv(x);
}

double relu(double x) {
    if (x > 0.) {
        return x;
    }
    else { 
        return 0.;
    }
}

double const_pi() { 
    return atan(1)*4; 
}

Eigen::MatrixXd softmax(Eigen::MatrixXd x) {
    return x.array().exp().colwise() / x.array().exp().rowwise().sum();
}