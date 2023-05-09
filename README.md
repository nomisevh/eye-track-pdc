# Multivariate Time-Series Classification for Parkinson’s Diagnosis

## Table of Contents

1. [Development Setup](#setup)
2. [How To Run](#run)

## Development Setup <a href="#setup"></a>

### Setting up environment

#### conda

To set up using conda, run the following:

```bash
cd <project_root>
conda env create -f conda/[OS].yaml
```

To update an existing environment to use the latest dependencies, run the following:

```bash
cd <project_root>
conda activate eyetrackpdc
conda env update --file conda/[OS].yaml --prune`
```

### Datasets

We use a private dataset from the Karolinska Institute for our experiments. For details, see our paper **todo**.
For access to the data, contact the authors.

### Logging

Logging is done via Neptune. For privacy reasons the neptune config is not tracked in version control. In order to use
with neptune, create a config on the form:

```yaml
project: <str>
api_key: <str>
capture_hardware_metrics: True
# async (default), offline, debug
mode: async
```

### Project Structure

📂`conda`: required package versions for different operating systems

📂`config`: config files

📂`data` datasets

&nbsp;&nbsp;┗ 📂`ki`: Karolinska Institute data

&nbsp;&nbsp;&nbsp;&nbsp;┗ 📂`tmp`: temp folder for cached dataset instances

📂`src`: Sources root

┣ 📂`models`: model classes

┣ 📂`processor`: data preprocessor

┣ 📂`utils`: utility methods and classes

┃📜...

📜`styleguide.md`: code style guide

## How To Run <a href="#run"></a>

After making sure that the environment is set up and the data is in the `data/ki` directory, you can
run the various experiments by their respective entrypoints in the `src/entrypoints` directory.

### Experiments

- Classification with ROCKET and classic ML classifiers `src/entrypoints/rocket.py`
- Classification with InceptionTime `src/entrypoints/inception.py`

 
