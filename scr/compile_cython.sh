#!/bin/bash

pushd src
.venv/bin/python3 pyx/cython_setup.py build_ext --inplace
popd
