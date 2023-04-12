#!/bin/bash
#
# ~~~
# This file is part of the paper:
#
#           "A Relaxed Localized Trust-Region Reduced Basis Approach for
#                      Optimization of Multiscale Problems"
#
# by: Tim Keil and Mario Ohlberger
#
#   https://github.com/TiKeil/Trust-region-TSRBLOD-code
#
# Copyright all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Author: Tim Keil 
# ~~~

set -e

git submodule update --init --recursive

# initialize the virtualenv
export BASEDIR="${PWD}"
python3 -m venv venv
source venv/bin/activate

# install python dependencies into the virtualenv
cd "${BASEDIR}"
pip install --upgrade pip
pip install $(grep Cython requirements.txt)
pip install -r requirements.txt

# install local pymor, gridlod and pdeopt version
cd "${BASEDIR}"
cd pymor && pip install -e .
cd "${BASEDIR}"
cd pdeopt && pip install -e .
cd "${BASEDIR}"
cd gridlod && pip install -e .
cd "${BASEDIR}"
cd TSRBLOD/rblod && pip install -e .
cd "${BASEDIR}"
cd TSRBLOD/perturbations-for-2d-data && pip install -e .
cd "${BASEDIR}"
cd scripts && mkdir tmp

cd "${BASEDIR}"
echo
echo "All done! From now on run"
echo "  source venv/bin/activate"
echo "to activate the virtualenv!"
