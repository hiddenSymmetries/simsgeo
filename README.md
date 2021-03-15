# DEPRECATION WARNING: This repository has been deprecated. All functionality has been moved into the [SIMSOPT](https://github.com/hiddenSymmetries/simsopt/) package.

# SIMSGEO

![build-and-test-status](https://github.com/hiddenSymmetries/simsgeo/workflows/Build%20&%20Test/badge.svg)
[![codecov](https://codecov.io/gh/hiddenSymmetries/simsgeo/branch/master/graph/badge.svg?token=3HW4S2HW69)](https://codecov.io/gh/hiddenSymmetries/simsgeo)



A collection of mostly geometric objects that are commonly used in Stellarator codes.


## Installation

It should be as simple as 

    git clone --recursive https://github.com/hiddenSymmetries/simsgeo.git
    cd simsgeo
    pip3 install -e .

and then you can run

    cd driver
    python3 driver.py

## Trouble shooting

On a mac you may want to point the install at the GCC compiler (e.g. because you have an old version of clang installed). Assuming you have `gcc-10` installed and in the `PATH`, simply do

    env CC=gcc-10 CXX=g++-10 pip3 install -vvv -e .
