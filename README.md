# GroningenML

This is the repository for the Kaggle competiton 'Titanic: Machine Learning from Disaster'
https://www.kaggle.com/c/titanic

## Overview

- wiki: overview of data and tips for the processing of the data 
- data: copy of Kaggle Titanic data


## Getting started

### Get the code

All command are relative to the code tree.

### (ana)conda)

Run `conda install --yes --file requirements.txt` or configure the environment based on requirements. See #2 when in trouble.

### pipenv

`pipenv install -r requirements.txt` 

### virtual env

- `virtualenv ktitanic` (once)
- `source ./bin/activate` (every time you revisit the project)
- `deactivate` (when tempory done)

See https://virtualenv.pypa.io/en/stable/userguide/#usage

### pip

As pip installs globally this is not recommended.

`pip install -r requirements.txt`

## Data

You can download the files manually or use [Kaggle API](https://github.com/Kaggle/kaggle-api)

### Kaggle API

Make sure you are in the `Data` directory.

Run `kaggle competitions download -c titanic`
