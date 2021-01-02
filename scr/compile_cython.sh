#!/bin/bash

pushd src
python pyx/cython_setup.py build_ext --inplace
popd
