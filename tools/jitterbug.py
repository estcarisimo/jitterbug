#!/usr/bin/env python
import os.path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
import numpy as np

from jitterbug.bcp import bcp
from jitterbug._jitter import compute_jiiter_dispersion, compute_ks_test
from jitterbug._latency_jump import latencyJumps
from jitterbug.cong_inference import congestionInference


DEFAULT_MOVING_IQR_ORDER = 4
DEFAULT_MOVING_AVERAGE_ORDER = 6

DEFAULT_JITTER_DISPERSION_THESHOLD = 0.25
DEFAULT_LATENCY_JUMP_THESHOLD = 0.5

CONGESTION_INFERENCE_METHODS = ["ks", "jd"]
CDP_ALGORITHMS = ["bcp", "hmm"]


def check_positive_value(x):

    _x = int(x)

    if _x < 0:
        argparse.ArgumentTypeError("The order of the filter MUST BE a positive INTEGER number")

    return _x

def check_even_value(x):

    _x = check_positive_value(x)

    if _x % 2 != 0:
        argparse.ArgumentTypeError("Jitterbug only admits even values (x % 2 == 0) for the order of the Moving Average filter")

    return _x

def jitterbug(epoch_rtt, rtt, epoch_mins, mins, inference_method, cdp_algorithm, 
              latency_jump_threshold, jitter_dispresion_threshold, 
              moving_average_order, moving_iqr_order):

    # cdp
    if cdp_algorithm =="bcp":
        cpDetector = bcp(epoch_mins, mins)
        cpDetector.fit()
        change_points = cpDetector.getChangePoints()
    elif cdp_algorithm == "hmm":
        raise Exception("We are sorry! This CDP algorithm has not been implemented yet.")
    else:
        raise Exception("There is no such CDP algorithm supported by Jitterbug")


    # check whether there are change point in the input signal
    if len(change_points) > 0:

        # Latency jumps
        lj = latencyJumps(epoch_mins, mins, change_points)
        lj.fit(latency_jump_threshold)
        latency_jumps = lj.getLatencyJumps()

        # Jitter analysis
        if inference_method == "jd":
            jitter_analysis = compute_jiiter_dispersion(change_points, epoch_mins, mins, 
                                                        moving_average_order, moving_iqr_order,
                                                        jitter_dispresion_threshold)
        elif inference_method == "ks":
            jitter_analysis = compute_ks_test(change_points, epoch_rtt, rtt)
        else:
            raise Exception("There is no such congestion inference method")
        
        # Congestion inference
        inference = congestionInference(latency_jumps, jitter_analysis)
        inference.fit()
        results = inference.getInferences()

    else:
        print("No change point was detected!")
        results = []
    
    return pd.DataFrame(results, columns=["starts", "ends", "congestion"])

def open_files(rtt_file, min_file):

    if not os.path.exists(rtt_file) or not os.path.exists(min_file):
        raise Exception("Input file(s) do(es) not exist")

    cols = ["epoch", "values"]
    rtts_df = (pd.read_csv(rtt_file, names=cols) 
              [lambda x: np.ones(len(x)).astype(bool)
                         if (x.iloc[0] != cols).all()
                         else np.concatenate([[False], np.ones(len(x)-1).astype(bool)])])
    mins_df = (pd.read_csv(min_file, names=cols) 
              [lambda x: np.ones(len(x)).astype(bool)
                         if (x.iloc[0] != cols).all()
                         else np.concatenate([[False], np.ones(len(x)-1).astype(bool)])])

    rtts_df = rtts_df.astype({"epoch": float, "values": float})
    mins_df = mins_df.astype({"epoch": float, "values": float})

    return rtts_df, mins_df


def main():


    parser = ArgumentParser()
    parser.add_argument("-m",
                        "--mins",
                        help="path to minimum RTT file.",
                        required=True)
    parser.add_argument("-r",
                        "--rtt",
                        help="path to raw RTT file.",
                        required=True)
    parser.add_argument("-i",
                        "--inference_method",
                        help="select the inference method (1) Jitter Dispersion (jd) (2) KS test (ks).",
                        required=True,
                        choices=CONGESTION_INFERENCE_METHODS
                        )
    parser.add_argument("-c",
                        "--cdp",
                        help="Select the Change Point Detection (CPD) algorithm: \
                              (1) Bayesian Chenge Point Detection (bcp, default) \
                              (2) Hidden Markov Model (hmm, not implemented yet)",
                        nargs='?', 
                        const="bcp",
                        default="bcp",
                        choices=CDP_ALGORITHMS,
                        type=str)
    parser.add_argument("-j",
                        "--jitter_dispersion_threhold",
                        help=f"Configure the sensitivity of the increase of the variability of the \
                               jitter dispersion time series to be considered as a period of congestion. \
                               This parameter is only used in the Jitter Dispersion Method. \
                               Default value {DEFAULT_JITTER_DISPERSION_THESHOLD}.",
                        nargs='?', 
                        const=DEFAULT_JITTER_DISPERSION_THESHOLD,
                        default=DEFAULT_JITTER_DISPERSION_THESHOLD,
                        type=float)
    parser.add_argument("-l",
                        "--latency_jump_threshold",
                        help=f"Configure the sensitivity of the increase of latency baseline \
                               to be considered as a period of congestion. \
                               Default value {DEFAULT_LATENCY_JUMP_THESHOLD}.",
                        nargs='?', 
                        const=DEFAULT_LATENCY_JUMP_THESHOLD,
                        default=DEFAULT_LATENCY_JUMP_THESHOLD,
                        type=float)
    parser.add_argument("-ma",
                        "--moving_average_order",
                        help=f"Define the order of the Moving Average filter. Use only POSITIVE EVEN INTEGER values. \
                               Default values is {DEFAULT_MOVING_AVERAGE_ORDER}.",
                        nargs='?', 
                        const=DEFAULT_MOVING_AVERAGE_ORDER,
                        default=DEFAULT_MOVING_AVERAGE_ORDER,
                        type=check_even_value)
    parser.add_argument("-iqr",
                        "--moving_iqr_order",
                        help=f"Define the order of the Moving IQR filter. Use only POSITIVE INTEGER values. \
                               Default values is {DEFAULT_MOVING_IQR_ORDER}.",
                        nargs='?', 
                        const=DEFAULT_MOVING_IQR_ORDER,
                        default=DEFAULT_MOVING_IQR_ORDER,
                        type=check_positive_value)
    parser.add_argument("-o",
                        "--output",
                        help="Specify the output file",
                        nargs='?', 
                        const="",
                        default="",
                        type=str)


    args = parser.parse_args()

    rtts_df, mins_df = open_files(args.rtt, args.mins)

    inferences = jitterbug(
        rtts_df["epoch"].values, 
        rtts_df["values"].values, 
        mins_df["epoch"].values, 
        mins_df["values"].values,
        args.inference_method,
        args.cdp,
        args.latency_jump_threshold,
        args.jitter_dispersion_threhold,
        args.moving_average_order,
        args.moving_iqr_order
    )

    if len(args.output) > 0:
        inferences.to_csv(
            args.output,
            header=True,
            index=False, 
        )
    else:
        print(inferences.values)

if __name__ == "__main__":
    # execute only if run as a script
    main()
