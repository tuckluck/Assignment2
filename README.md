# Assignment2

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)


[![codecov](https://codecov.io/gh/tuckluck/Assignment2/graph/badge.svg?token=TKF4CLV1G5)](https://codecov.io/gh/tuckluck/Assignment2)
[![tests](https://github.com/tuckluck/Assignment2/actions/workflows/testsDS.yml/badge.svg)](https://github.com/tuckluck/Assignment2/actions)


Welcome to the Assignment 2 repo. Please follow the directions below to download and run the code for the direct stiffness solver

To install this package, please begin by setting up a conda environment (mamba also works):
```bash
conda create --name tl_Assignment2-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate tl_Assignment2-env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```

Create an editable install of the Newtonian method code (note: you must be in the correct directory):
```bash
pip install -e .
```
Test that the code is working with pytest:
```bash
pytest -v --cov=direct_stiffness --cov-report term-missing
```
Code coverage should be nearly 100%. Now you are prepared to write your own code based on this method and/or run the tutorial. 

Setup juypter notebook for tutorials

```bash
pip install jupyter
```

Run Tutorial
```bash
jupyter notebook Assign2_tutorial.ipynb
```


