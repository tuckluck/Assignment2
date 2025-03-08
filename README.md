# Assignment2
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
Test that the code is working with pytest:
```bash
pytest -v --cov=direct_stiffness --cov-report term-missing
```
Code coverage should be nearly 100%. Now you are prepared to write your own code based on this method and/or run the tutorial. 

Create an editable install of the Newtonian method code (note: you must be in the correct directory):
```bash
pip install -e .
```

Setup juypter notebook for tutorials

```bash
pip install jupyter
```

Run Tutorial
```bash
jupyter notebook Assign2_tutorial.ipynb
```


