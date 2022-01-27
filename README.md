# Jitterbug: A new framework for jitter-based congestion inference

Welcome to the repository of _Jitterbug_. Here you can find a the source code and the instructions to install and use _Jitterbug_.

Jitterbug is a product of the research published in the homonymous paper _"Jitterbug: A new framework for jitter-based congestion inference"_ appearing in the Proceedings of the [Passive and Active Measurement Conference (PAM) 2022](https://pam2022.nl/), March 2022, Virtual Event.

# Table of Contents

1. [Install Jitterbug](#setup)
   1. [Set up a virtual environment](#venv)
   2. [Clone additional third-party dependencies](#dependencies)
   3. [Install](#install)
2. [Run Jitterbug](#run)
   1. [Example of use](#use)
   2. [Data format of the input files](#format)
   3. [Examples with real-world data](#format)
3. [Jitterbug --help](#help)
4. [Citation](#citation)
5. [Strcuture of the repository](#tree)


# <a name="setup"></a> 1. Install Jitterbug


## <a name="venv"></a> 1.1. Set up a virtual environment

We highly recommend you to use a Python virtual environment to run these examples. In this repository, we also include a requirements.txt to install all python packages needed to run the examples.

To install this virtual environment, you have to run the following commands.

**This repo includes addition requirements to run the example.**

```
$ python3 -m venv .jitterbug
$ source .jitterbug/bin/activate
$ pip3 install ipykernel
$ ipython kernel install --user --name=.jitterbug
$ pip3 install -r requirements.txt
```

## <a name="dependencies"></a> 1.2. Clone additional third-party dependencies

```
$ git clone https://github.com/hildensia/bayesian_changepoint_detection.git
$ cd bayesian_changepoint_detection
$ python setup.py install
```

## <a name="dependencies"></a> 1.3. Install

```
$ python setup.py sdist bdist_wheel build_ext
$ pip install -e .
```

# <a name="run"></a> 2. Run Jitterbug

## <a name="use"></a> 2.1 Example of use

If you're using a virtual environment, please activate it before running this command

```
$ jitterbug -m mins.csv -r rtts.csv -i jd -c bcp
```

You can find real-world examples of ```rtts.csv``` [(here!)](example/data/congestion-measurements/raw.csv) and ```mins.csv``` [(here!)](example/data/congestion-measurements/mins.csv) files in this repository.

## <a name="format"></a> 2.2 Data format of the input files

Here we show some example of the structure of the input files to run Jitterbug

### 2.2.1 RTT measurements

Data structure of the ```rtts.csv``` file

```
epoch,values
1512144010.0,63.86
1512144010.0,66.52
1512144010.0,72.09
1512144020.0,85.2
```
### 2.2.2 minimun RTT

Data structure of the ```mins.csv``` file

```
epoch,values
1512144000.0,42.85
1512144900.0,18.82
1512145800.0,28.53
1512146700.0,38.89
```

# <a name="notebooks"></a> 2.3 Examples with real-world data

To get familiar with Jitter, we include on this repo **two** Jupyter Notebooks presenting the input time series and the congestion inference using _KS-test_ and _Jitter dispersion_ methods:

 1. [KS-test congestion inference](examples/jitter-kstest-example.ipynb)
 2. [Jitter dispersion congestion inference](examples/jitter-dispersion-example.ipynb)


# <a name="help"></a> 3. Jitterbug --help

```
$ jitterbug --help
usage: jitterbug [-h] -m MINS -r RTT -i {ks,jd} [-c [{bcp,hmm}]]
                 [-j [JITTER_DISPERSION_THREHOLD]]
                 [-l [LATENCY_JUMP_THRESHOLD]] [-ma [MOVING_AVERAGE_ORDER]]
                 [-iqr [MOVING_IQR_ORDER]] [-o [OUTPUT]]

optional arguments:
  -h, --help            show this help message and exit
  -m MINS, --mins MINS  path to minimum RTT file.
  -r RTT, --rtt RTT     path to raw RTT file.
  -i {ks,jd}, --inference_method {ks,jd}
                        select the inference method (1) Jitter Dispersion (jd)
                        (2) KS test (ks).
  -c [{bcp,hmm}], --cdp [{bcp,hmm}]
                        Select the Change Point Detection (CPD) algorithm: (1)
                        Bayesian Chenge Point Detection (bcp, default) (2)
                        Hidden Markov Model (hmm, not implemented yet)
  -j [JITTER_DISPERSION_THREHOLD], --jitter_dispersion_threhold [JITTER_DISPERSION_THREHOLD]
                        Configure the sensitivity of the increase of the
                        variability of the jitter dispersion time series to be
                        considered as a period of congestion. This parameter
                        is only used in the Jitter Dispersion Method. Default
                        value 0.25.
  -l [LATENCY_JUMP_THRESHOLD], --latency_jump_threshold [LATENCY_JUMP_THRESHOLD]
                        Configure the sensitivity of the increase of latency
                        baseline to be considered as a period of congestion.
                        Default value 0.5.
  -ma [MOVING_AVERAGE_ORDER], --moving_average_order [MOVING_AVERAGE_ORDER]
                        Define the order of the Moving Average filter. Use
                        only POSITIVE EVEN INTEGER values. Default values is
                        6.
  -iqr [MOVING_IQR_ORDER], --moving_iqr_order [MOVING_IQR_ORDER]
                        Define the order of the Moving IQR filter. Use only
                        POSITIVE INTEGER values. Default values is 4.
  -o [OUTPUT], --output [OUTPUT]
                        Specify the output file
```


# <a name="citation"></a>3. Citation

If you use _Jitterbug_, please cite it as:

```
```

# <a name="tree"></a>3. Repo structure

```
.
├── LICENSE
├── README.md
├── example
│   ├── data
│   │   ├── congestion-inferences
│   │   │   ├── jd_inferences.csv
│   │   │   └── kstest_inferences.csv
│   │   └── congestion-measurements
│   │       ├── mins.csv
│   │       └── raw.csv
│   ├── jitter-dispersion-example.ipynb
│   └── jitter-kstest-example.ipynb
├── jitterbug
│   ├── __init__.py
│   ├── _jitter.py
│   ├── _jitterbug.py
│   ├── _latency_jump.py
│   ├── bcp.py
│   ├── cong_inference.py
│   ├── filters.py
│   ├── kstest.py
│   ├── preprocessing.py
│   └── signal_energy.py
├── requirements.txt
├── setup.py
└── tools
    └── jitterbug.py

6 directories, 21 files
```
