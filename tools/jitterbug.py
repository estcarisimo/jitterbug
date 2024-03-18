#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from jitterbug.bcp import BCP
from jitterbug._jitter import compute_jitter_dispersion, compute_ks_test
from jitterbug._latency_jump import LatencyJumps
from jitterbug.cong_inference import CongestionInference

# Default parameters
DEFAULT_MOVING_IQR_ORDER = 4
DEFAULT_MOVING_AVERAGE_ORDER = 6
DEFAULT_CPD_THRESHOLD = 0.25
DEFAULT_JITTER_DISPERSION_THRESHOLD = 0.25
DEFAULT_LATENCY_JUMP_THRESHOLD = 0.5
CONGESTION_INFERENCE_METHODS = ["ks", "jd"]
CDP_ALGORITHMS = ["bcp", "hmm"]

def check_positive_value(x):
    """
    Ensures that a value is a positive integer.

    Parameters:
    x (str): Input value as a string.

    Returns:
    int: Positive integer value.

    Raises:
    ValueError: If the value is negative or not an integer.
    """
    _x = int(x)
    if _x < 0:
        raise ValueError("The order of the filter MUST BE a positive INTEGER number")
    return _x

def check_even_value(x):
    """
    Ensures that a value is a positive, even integer.

    Parameters:
    x (str): Input value as a string.

    Returns:
    int: Positive, even integer value.

    Raises:
    ValueError: If the value is not even or not a positive integer.
    """
    _x = check_positive_value(x)
    if _x % 2 != 0:
        raise ValueError("Jitterbug only admits even values (x % 2 == 0) for the order of the Moving Average filter")
    return _x

def jitterbug_analysis(epoch_rtt, rtt, epoch_mins, mins, inference_method, cdp_algorithm, 
                       latency_jump_threshold, jitter_dispersion_threshold, 
                       moving_average_order, moving_iqr_order, cpd_threshold):
    """
    Main function to execute jitter and latency jump analysis based on the specified parameters.

    Parameters are as described in the command line arguments.
    """
    if cdp_algorithm == "bcp":
        cp_detector = BCP(epoch_mins, mins, cpd_threshold=cpd_threshold)
        cp_detector.fit()
        change_points = cp_detector.getChangePoints()
    elif cdp_algorithm == "hmm":
        raise NotImplementedError("This CDP algorithm has not been implemented yet.")
    else:
        raise ValueError("Unsupported CDP algorithm specified.")

    if len(change_points) > 0:
        latency_jumps_detector = LatencyJumps(epoch_mins, mins, change_points)
        latency_jumps_detector.fit(latency_jump_threshold)
        latency_jumps = latency_jumps_detector.getLatencyJumps()

        if inference_method == "jd":
            jitter_analysis = compute_jitter_dispersion(change_points, epoch_mins, mins, 
                                                        moving_average_order, moving_iqr_order,
                                                        jitter_dispersion_threshold)
        elif inference_method == "ks":
            jitter_analysis = compute_ks_test(change_points, epoch_rtt, rtt)
        else:
            raise ValueError("Unsupported congestion inference method.")
        
        congestion_inference = CongestionInference(latency_jumps, jitter_analysis)
        congestion_inference.fit()
        results = congestion_inference.getInferences()
    else:
        print("No change point was detected!")
        results = []

    return pd.DataFrame(results, columns=["starts", "ends", "congestion"])


def load_and_process_rtt_data(rtt_file):
    """
    Loads the RTT data from a specified file and processes it to extract minimum RTT values over intervals.

    Parameters
    ----------
    rtt_file : str
        Path to the file containing RTT data.

    Returns
    -------
    tuple of pd.DataFrame
        - The first DataFrame contains the original RTT data with an added datetime column.
        - The second DataFrame contains the minimum RTT values computed over 15-minute intervals.

    Raises
    ------
    FileNotFoundError
        If the specified RTT file does not exist.

    Notes
    -----
    The function reads Round-Trip Time (RTT) data from a CSV file, converts epoch timestamps to datetime,
    and computes the minimum RTT values within 15-minute intervals. This processed data can be useful for
    analyzing network performance and identifying patterns or anomalies in RTT over time.
    """
    # Check if the RTT file exists and raise an error if it does not
    if not os.path.exists(rtt_file):
        raise FileNotFoundError("Input file does not exist")

    # Read the RTT data from the file
    rtts_df = pd.read_csv(rtt_file)

    # Convert the 'epoch' column to datetime format for easier analysis
    rtts_df['datetime'] = pd.to_datetime(rtts_df['epoch'], unit='s')

    # Group the data by 15-minute intervals and compute the minimum RTT for each interval
    mins_df = rtts_df.groupby(pd.Grouper(key='datetime', freq="15min"))[["epoch", "values"]].min().reset_index()

    return rtts_df, mins_df



def main():
    """
    Main function to parse arguments and execute the analysis.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--rtt", required=True, help="path to raw RTT file.")
    parser.add_argument("-i", "--inference_method", required=True, choices=CONGESTION_INFERENCE_METHODS,
                        help="select the inference method: Jitter Dispersion (jd) or KS test (ks).")
    parser.add_argument("-c", "--cdp", nargs='?', const="bcp", default="bcp", choices=CDP_ALGORITHMS,
                        help="select the Change Point Detection (CPD) algorithm: Bayesian Change Point Detection (bcp, default) or Hidden Markov Model (hmm, not implemented yet).")
    parser.add_argument("-cpdth", "--cpd_threshold", nargs='?', const=DEFAULT_CPD_THRESHOLD, default=DEFAULT_CPD_THRESHOLD, type=float,
                        help="sensitivity of the Change Point Detection algorithm.")
    parser.add_argument("-j", "--jitter_dispersion_threshold", nargs='?', const=DEFAULT_JITTER_DISPERSION_THRESHOLD, default=DEFAULT_JITTER_DISPERSION_THRESHOLD, type=float,
                        help="sensitivity of the increase of the variability of the jitter dispersion time series to be considered as a period of congestion (used in Jitter Dispersion Method).")
    parser.add_argument("-l", "--latency_jump_threshold", nargs='?', const=DEFAULT_LATENCY_JUMP_THRESHOLD, default=DEFAULT_LATENCY_JUMP_THRESHOLD, type=float,
                        help="sensitivity of the increase of latency baseline to be considered as a period of congestion.")
    parser.add_argument("-ma", "--moving_average_order", nargs='?', const=DEFAULT_MOVING_AVERAGE_ORDER, default=DEFAULT_MOVING_AVERAGE_ORDER, type=check_even_value,
                        help="order of the Moving Average filter (POSITIVE EVEN INTEGER).")
    parser.add_argument("-iqr", "--moving_iqr_order", nargs='?', const=DEFAULT_MOVING_IQR_ORDER, default=DEFAULT_MOVING_IQR_ORDER, type=check_positive_value,
                        help="order of the Moving IQR filter (POSITIVE INTEGER).")
    parser.add_argument("-o", "--output", nargs='?', const="", default="", type=str,
                        help="specify the output file (results are printed to stdout if not specified).")

    args = parser.parse_args()

    rtts_df, mins_df = load_and_process_rtt_data(args.rtt)

    inferences = jitterbug_analysis(
        rtts_df["epoch"].values, 
        rtts_df["values"].values, 
        mins_df["epoch"].values, 
        mins_df["values"].values,
        args.inference_method,
        args.cdp,
        args.latency_jump_threshold,
        args.jitter_dispersion_threshold,
        args.moving_average_order,
        args.moving_iqr_order,
        args.cpd_threshold
    )

    if args.output:
        inferences.to_csv(args.output, header=True, index=False)
    else:
        print(inferences)

if __name__ == "__main__":
    main()
