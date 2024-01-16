#!/bin/bash

#create environment
conda create -n train_llm python==3.11

#install jupyter notebook
Pip install Jupyter
conda install -c conda-forge ipywidgets
Sudo python -m ipykernel install --name train_llm
