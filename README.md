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
jupyter notebook ds_tutorial1.ipynb
```


