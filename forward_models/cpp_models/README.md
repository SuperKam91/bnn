Example C++ repository
======================

* Headers for in-house software are in `./include`, and should be appended `.hpp`
* Sources are in `./src`, and should be appended `.cpp`
* The [Eigen](http://eigen.tuxfamily.org/dox/) and [Catch2](https://github.com/catchorg/Catch2) header-only libraries are included in `./external/Eigen and .external/Catch2`. Also, the headers for the [GNU scientific library](https://www.gnu.org/software/gsl/doc/html/usage.html) are included at ./external/gsl, but also requires the library (as well as relevant blas libraries to be included in the /lib dir).
* the polychord library should also be built and included in /lib.
* Tests are included in `./src/test`
* An example source file and it's corresponding header and test has been included

You can make and run the program with
```bash
make -j
./bin/main
```
where `-j` indicates that you can compile in parallel (which is possible due to the [Non-recursive nature](http://aegis.sourceforge.net/auug97.pdf) of the makefile. `./main.cpp` includes some example code.

You can make and run the tests with
```bash
make test
./bin/test
```
