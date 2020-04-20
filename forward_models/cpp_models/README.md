C++ BNN implementation
======================

* Headers for in-house software are in `./include`, and should be appended `.hpp`
* Sources are in `./src`, and should be appended `.cpp`
* The [Eigen](http://eigen.tuxfamily.org/dox/) and [Catch2](https://github.com/catchorg/Catch2) header-only libraries are included in `./external/Eigen and .external/Catch2`
* The headers for the [GNU scientific library](https://www.gnu.org/software/gsl/doc/html/usage.html) are included at ./external/gsl/, but one also requires the library (as well as relevant blas libraries to be included in the /lib dir). To obtain these, download the gsl install from: ftp://ftp.gnu.org/gnu/gsl/. Once extracted, go to the file directory and do something along the lines of ./configure --prefix=/home/dir_to_save_gsl_install && make && make install. n.b. without the prefix, I often get permission errors for the default install location. The libraries can then be found in home/dir_to_save_gsl_install/lib/ (n.b. this includes the blas libraries etc.). These should then be copied to the bnn/.../lib/ directory. Note also the gsl headers can be found in home/dir_to_save_gsl_install/include/
* the polychord library should also be built and included in bnn/.../lib
* Tests are included in `./src/test`
* An example source file and it's corresponding header and test has been included

You can `make` i.e. compile the code with:
```bash
make all
```
