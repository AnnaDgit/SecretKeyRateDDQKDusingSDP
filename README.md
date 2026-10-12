# SecretKeyRateDDQKDusingSDP

[https://github.com/AnnaDgit/SecretKeyRateDDQKDusingSDP.git](https://github.com/AnnaDgit/SecretKeyRateDDQKDusingSDP.git)

The repository contains the code files to generate all results of the research paper "Secret Key Rate of Device Dependent Quantum Key Distribution using Semidefinite Programming". The project code is implemented in both MATLAB and Python and can be found in the respective folders:

- [MATLAB](./MATLAB/)
- [Python](./Python/)

The code for this project uses the SDP developed by Araujo et al. for the research paper “Quantum key distribution rates from semidefinite
programming, whose code is publicly available at https://github.com/araujoms/qkd.

## MATLAB

To use the MATLAB code, install **YALMIP**, a toolbox that lets you formulate complex optimization problems using a simple syntax and acts as a bridge to external solvers such as **MOSEK**, which is used in this project.

1. Download the [YALMPI Zip](https://github.com/yalmip/yalmip/archive/master.zip).
2. Unzip the archive and add all its directories to your MATLAB path.
3. Follow the [YALMIP Installation](https://yalmip.github.io/tutorial/installation/) instructions for more details.
4. Install [MOSEK](https://docs.mosek.com/latest/install/installation.html) by following its installation instructions.
5. Obtain a MOSEK license. A free academic license is available [here](https://www.mosek.com/products/academic-licenses/).

## Python

To use the Python code, install **CVXPY** and the **MOSEK** solver:

```bash
pip install cvxpy
pip install Mosek
```

A MOSEK license is also required. A free academic license is available [here](https://www.mosek.com/products/academic-licenses/).
