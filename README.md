```
# ~~~
# This file is part of the paper:
#
#           " An Online Efficient Two-Scale Reduced Basis Approach
#                for the Localized Orthogonal Decomposition "
#
#   https://github.com/TiKeil/Two-scale-RBLOD.git
#
# Copyright 2021 all developers. All rights reserved.
# License: licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil
#   Stephan Rave
# ~~~
```

In this repository, we provide the code for the numerical experiments in Section 5
of the paper **"tba"** by Tim Keil and Mario Ohlberger.
The preprint is available [here](https://arxiv.org/).

For just taking a look at the experiment outputs and data, you do not need to
install the software. Just go to `scripts/test_outputs/`,
where we have stored printouts of our numerical experiments.
In order to relate this data to the paper, we provide further information in the next section.

If you want to have a closer look at the implementation or generate the results by
yourself, we provide simple setup instructions for configuring your own Python environment.
We note that our setup instructions are written for Ubuntu Linux only and we do not provide
setup instructions for MacOS and Windows.
Our setup instructions have successfully been tested on a fresh Ubuntu 20.04.2.0 LTS system.
The actual experiments have been computed on the
[PALMA II HPC cluster](<https://www.uni-muenster.de/IT/en/services/unterstuetzungsleistung/hpc/index.shtml>).
For the concrete configurations we refer to the scripts in `submit_to_cluster`.

# How to quickly find the data from the paper

We provide information on how to relate the output files to the figures and tables in the paper.
All output files and figures are stored in `scripts/test_outputs`.
Note that the outputs are verbose outputs compared to the ones that we present in the paper,
which is also the reason why we do not provide scripts for constructing the error plots and
tables from the paper.

# Organization of the repository

We used several external software packages:

- [pyMOR](https://pymor.org) is a software library for Model Order Reduction.
- [gridlod](https://github.com/fredrikhellman/gridlod) is a discretization toolkit for the
Localized Orthogonal Decompostion (LOD) method. 
- [perturbations-for-2d-data](https://github.com/TiKeil/perturbations-for-2d-data) contains
a coefficient generator for constructing randomized and highly oscillating coefficients.
- [TSRBLOD](https://github.com/TiKeil/Two-scale-RBLOD) this code has been used for the [paper](https://arxiv.org/abs/2111.08643) where the TSRBLOD has been introduced. The essential code for our project can be found the respective `rblod` module.

We added the external software as editable submodules with a fixed commit hash.
For the TR-TSRBLOD, we have developed a Python module `pdeopt` that provides all other code that was required for the project. We also note that parts of this code have already been developed in the respective TR-RB software.
The rest of the code is contained in `scripts`, where you find the main scripts for the numerical experiments.

# Setup

On a standard Ubuntu system (with Python and C compilers installed) it will most likely be enough
to just run our setup script. For that, please clone the repository

```
git clone https://github.com/TiKeil/Trust-region-TSRBLOD-code.git
```

and execute the provided setup script via 

```
cd Trust-region-TSRBLOD-code
./setup.sh
```

If this does not work for you, and you don't know how to help yourself,
please follow the extended setup instructions below.

## Installation on a fresh system

We also provide setup instructions for a fresh Ubuntu system (20.04.2.0 LTS).
The following steps need to be taken:

```
sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install build-essential
sudo apt install python3-dev
sudo apt install python3-venv
sudo apt install libopenmpi-dev
sudo apt install libsuitesparse-dev
```

Now you are ready to clone the repository and run the setup script:

```
git clone https://github.com/TiKeil/Trust-region-TSRBLOD-code.git
cd Trust-region-TSRBLOD-code
./setup.sh
```

# Running the experiments

You can make sure your that setup is complete by running the minimal test script
```
cd scripts/test_scripts
mpirun python minimal_test.py
```

If this works fine (with a summary of some methods in the end), your setup is working well.

Moreover, we provide further information
on how to reconstruct the figures and tables from the paper.
Please note that these shell scripts will produce verbose outputs.
The above mentioned output files in `scripts/test_outputs` are a minimal version of this.
For executing Python scripts, you need to activate the virtual environment by

```
source venv/bin/activate
```

Note that the jobs require a HPC computing system.
In particular, starting the shell scripts with only a few parallel cores (or even without `mpirun`)
on your local computer may take days to weeks.

Please have a look at the description of the main scripts to try different configurations of the given problem classes.
Note that it is also possible to solve your own parameterized problems with our code since the problem definitions that are used in
`pdeopt/problems.py` are very general. 

# Questions

If there are any questions of any kind, please contact us via <tim.keil@wwu.de>.
