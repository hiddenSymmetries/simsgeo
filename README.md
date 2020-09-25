# SIMSGEO

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
