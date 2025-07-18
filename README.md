# Jitterbug: Framework for Jitter-Based Congestion Inference

Welcome to the Jitterbug repository, where you can access the source code and installation instructions for Jitterbug.

Jitterbug is the outcome of research presented in the paper _"Jitterbug: A new framework for jitter-based congestion inference"_, published in the Proceedings of the Passive and Active Measurement Conference (PAM) 2022, held virtually in March 2022.

## Table of Contents

1. [Installation Guide](#installation)
   1. [Setting Up a Virtual Environment](#virtual-environment)
   2. [Cloning Third-Party Dependencies](#dependencies)
   3. [Installation Process](#install)
2. [Using Jitterbug](#usage)
   1. [Usage Example](#example)
   2. [Input File Format](#input-format)
   3. [Real-World Data Examples](#real-data)
3. [Command-Line Interface](#cli)
4. [Citing Jitterbug](#citing)
5. [Repository Structure](#structure)

## <a name="installation"></a> 1. Installation Guide

### <a name="virtual-environment"></a> 1.1. Setting Up a Virtual Environment

It is recommended to use a Python virtual environment for running Jitterbug. This repository contains a `requirements.txt` file for installing all necessary Python packages.

Execute the following commands to set up the virtual environment:

```
$ python3 -m venv .jitterbug
$ source .jitterbug/bin/activate
$ pip3 install ipykernel
$ ipython kernel install --user --name=.jitterbug
$ pip3 install -r requirements.txt
```

### <a name="dependencies"></a> 1.2. Cloning Third-Party Dependencies

```
$ git clone https://github.com/hildensia/bayesian_changepoint_detection.git
$ cd bayesian_changepoint_detection
$ python setup.py install
$ cd ..
```

### <a name="install"></a> 1.3. Installation Process

```
$ python setup.py sdist bdist_wheel build_ext
$ pip install -e .
```

## <a name="usage"></a> 2. Using Jitterbug

### <a name="example"></a> 2.1. Usage Example

Ensure the virtual environment is activated before executing Jitterbug:

```
$ jitterbug -r rtts.csv -i jd -c bcp
```

Example `rtts.csv` file is provided in the repository.

### <a name="input-format"></a> 2.2. Input File Format

#### RTT Measurements

Format of `rtts.csv`:

```
epoch,values
1512144010.0,63.86
1512144010.0,66.52
...
```


### <a name="real-data"></a> 2.3. Real-World Data Examples

Explore Jitterbug using two Jupyter Notebooks provided in this repository:

1. [KS-test Congestion Inference](example/jitter-kstest-example.ipynb)
2. [Jitter Dispersion Congestion Inference](example/jitter-dispersion-example.ipynb)

## <a name="cli"></a> 3. Command-Line Interface

For detailed command usage, run:

```
$ jitterbug --help
```

## <a name="citing"></a> 4. Citing Jitterbug

Please cite Jitterbug using the following reference:

```
@InProceedings{carisimo2022jitterbug,
  author="Carisimo, Esteban and Mok, Ricky K. P. and Clark, David D. and Claffy, K. C.",
  title="Jitterbug: A New Framework for Jitter-Based Congestion Inference",
  booktitle="Passive and Active Measurement",
  year="2022",
  publisher="Springer International Publishing",
  address="Cham",
  pages="155--179",
  isbn="978-3-030-98785-5"
}
```

## <a name="structure"></a> 5. Repository Structure

```
.
├── LICENSE
├── README.md
├── example
│   ├── data
│   │   ├── congestion-inferences
│   │   │   ├── jd_inferences.csv
│   │   │   └── kstest_inferences.csv
│   │   └── congestion-measurements
│   │       ├── mins

.csv
│   │       └── raw.csv
│   ├── jitter-dispersion-example.ipynb
│   └── jitter-kstest-example.ipynb
├── jitterbug
│   ├── __init__.py
│   ├── _jitter.py
│   ├── _jitterbug.py
│   ├── _latency_jump.py
│   ├── bcp.py
│   ├── cong_inference.py
│   ├── filters.py
│   ├── kstest.py
│   ├── preprocessing.py
│   └── signal_energy.py
├── requirements.txt
├── setup.py
└── tools
    └── jitterbug.py
```

TBD