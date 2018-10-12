#include <cmath>

/* in-house code */
#include "mathematics.hpp"

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