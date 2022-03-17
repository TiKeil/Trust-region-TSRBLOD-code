# ~~~
# This file is part of the paper:
#
#           "A relaxed localized trust-region reduced basis approach for
#                      optimization of multiscale problems"
#
# by: Tim Keil and Mario Ohlberger
#
#   https://github.com/TiKeil/Trust-region-TSRBLOD-code
#
# Copyright 2019-2022 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil (2022)
# ~~~

from setuptools import setup

setup(name='pdeopt',
      version='2020.1',
      description='Pymor support for PDE-constrained optimization',
      author='Tim Keil, Luca Mechelli',
      author_email='tim.keil@wwu.de',
      license='MIT',
      packages=['pdeopt'])
