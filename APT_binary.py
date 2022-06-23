#!/usr/bin/env python
# -W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.utils
import pint.models.model_builder as mb
import pint.random_models
from pint.phase import Phase
from copy import deepcopy
from collections import OrderedDict
from astropy import log
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import operator
import time
from pathlib import Path
import socket


def starting_points(toas, args=None):
    """
    Choose which cluster to NOT jump, i.e. where to start

    toas : TOA object
    args : command line arguments
    """
    t = deepcopy(toas)
    mjd_values = t.get_mjds().value
    dts = np.fabs(mjd_values - mjd_values[:, np.newaxis]) + np.eye(len(mjd_values))

    score_list = (1.0 / dts).sum(axis=1)

    mask_list = []
    starting_cluster_list = []
    # f = pint.fitter.WLSFitter(t, m)
    # f.fit_toas()
    i = -1
    while score_list.any():
        # break
        i += 1
        # print(i)
        hsi = np.argmax(score_list)
        score_list[hsi] = 0
        cluster = t.table["clusters"][hsi]
        # mask = np.zeros(len(mjd_values), dtype=bool)
        mask = t.table["clusters"] == cluster

        if i == 0 or not np.any(
            np.all(mask == mask_list, axis=1)
        ):  # equivalent to the intended effect of checking if not mask in mask_list
            mask_list.append(mask)
            starting_cluster_list.append(cluster)
    if args is not None:
        max_starts = args.max_starts
    else:
        max_starts = 5
    return (mask_list[:max_starts], starting_cluster_list[:max_starts])


def JUMP_adder_begginning(
    mask: np.ndarray, toas, model, output_timfile, output_parfile
):
    """
    Adds JUMPs to a timfile as the begginning of analysis.

    mask : a mask to select which toas will not be jumped
    toas : TOA object
    output_timfile : name for the tim file to be written
    output_parfile : name for par file to be written
    """
    t = deepcopy(toas)
    flag_name = "jump_tim"

    former_cluster = t.table[mask]["clusters"][0]
    j = 0
    for i, table in enumerate(t.table[~mask]):
        if table["clusters"] != former_cluster:
            former_cluster = table["clusters"]
            j += 1
        table["flags"][flag_name] = str(j)
    t.write_TOA_file(output_timfile)

    # model.jump_flags_to_params(t) doesn't currently work (need flag name to be "tim_jump" and even then it still won't work),
    # so the following is a workaround. This is likely related to issue 1294.
    ### (workaround surrounded in ###)
    with open(output_parfile, "w") as parfile:
        parfile.write(model.as_parfile())
        for i in range(1, j + 1):
            parfile.write(f"JUMP\t\t-{flag_name} {i}\t0 1 0\n")
    model = mb.get_model(output_parfile)
    ###

    return t, model


def set_F1_lim(args, parfile):
    # if F1_lim not specified in command line, calculate the minimum span based on general F0-F1 relations from P-Pdot diagram

    if args.F1_lim == None:
        # for slow pulsars, allow F1 to be up to 1e-12 Hz/s, for medium pulsars, 1e-13 Hz/s, otherwise, 1e-14 Hz/s (recycled pulsars)
        F0 = mb.get_model(parfile).F0.value

        if F0 < 10:
            F1 = 10**-12

        elif 10 < F0 < 100:
            F1 = 10**-13

        else:
            F1 = 10**-14

        # rearranged equation [delta-phase = (F1*span^2)/2], span in seconds.
        # calculates span (in days) for delta-phase to reach 0.35 due to F1
        args.F1_lim = np.sqrt(0.35 * 2 / F1) / 86400.0


def APT_argument_parse(parser, argv):
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="tim file to read toas from")
    parser.add_argument(
        "binary_model",
        help="which binary pulsar model to use.",
        choices=["ELL1", "ell1"],
    )
    parser.add_argument(
        "--starting_points",
        help="mask array to apply to chose the starting points, clusters or mjds",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--maskfile",
        help="csv file of bool array for fit points",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_pred",
        help="Number of predictive models that should be calculated",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--ledge_multiplier",
        help="scale factor for how far to plot predictive models to the left of fit points",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--redge_multiplier",
        help="scale factor for how far to plot predictive models to the right of fit points",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--RAJ_lim",
        help="minimum time span before Right Ascension (RAJ) can be fit for",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--DECJ_lim",
        help="minimum time span before Declination (DECJ) can be fit for",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--F1_lim",
        help="minimum time span before Spindown (F1) can be fit for (default = time for F1 to change residuals by 0.35phase)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--Ftest_lim",
        help="Upper limit for successful Ftest values",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--check_bad_points",
        help="whether the algorithm should attempt to identify and ignore bad data",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--plot_bad_points",
        help="Whether to actively plot the polynomial fit on a bad point. This will interrupt the program and require manual closing",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--check_bp_min_diff",
        help="minimum residual difference to count as a questionable point to check",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--check_bp_max_resid",
        help="maximum polynomial fit residual to exclude a bad data point",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--check_bp_n_clusters",
        help="how many clusters ahead of the questionable group to fit to confirm a bad data point",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--try_poly_extrap",
        help="whether to try to speed up the process by fitting ahead where polyfit confirms a clear trend",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--plot_poly_extrap",
        help="Whether to plot the polynomial fits during the extrapolation attempts. This will interrupt the program and require manual closing",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--pe_min_span",
        help="minimum span (days) before allowing polynomial extrapolation attempts",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--pe_max_resid",
        help="maximum acceptable goodness of fit for polyfit to allow the polynomial extrapolation to succeed",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--span1_c",
        help="coefficient for first polynomial extrapolation span (i.e. try polyfit on current span * span1_c)",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--span2_c",
        help="coefficient for second polynomial extrapolation span (i.e. try polyfit on current span * span2_c)",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--span3_c",
        help="coefficient for third polynomial extrapolation span (i.e. try polyfit on current span * span3_c)",
        type=float,
        default=2.4,
    )
    parser.add_argument(
        "--max_wrap",
        help="how many phase wraps in each direction to try",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--plot_final",
        help="whether to plot the final residuals at the end of each attempt",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--data_path",
        help="where to store data",
        type=str,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--parfile_compare",
        help="par file to compare solution to",
        type=str,
        default="",
    )
    parser.add_argument(
        "--chisq_cutoff",
        help="The minimum reduced chisq to be admitted as a potential solution.",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--max_starts",
        help="maximum number of initial JUMP configurations",
        type=float,
        default=5,
    )

    args = parser.parse_args(argv)
    # interpret strings as booleans
    args.check_bad_points = args.check_bad_points.lower()[0] == "t"
    args.try_poly_extrap = args.try_poly_extrap.lower()[0] == "t"
    args.plot_poly_extrap = args.plot_poly_extrap.lower()[0] == "t"
    args.plot_bad_points = args.plot_bad_points.lower()[0] == "t"
    args.plot_final = args.plot_final.lower()[0] == "t"

    return args, parser


def main(argv=None):
    import argparse
    import sys

    # read in arguments from the command line

    """required = parfile, timfile"""
    """optional = starting points, param ranges"""
    parser = argparse.ArgumentParser(
        description="PINT tool for agorithmically timing pulsars."
    )

    args, parser = APT_argument_parse(parser, argv)

    # if given starting points from command line, check if ints (group numbers) or floats (mjd values)
    start_type = None
    start = None
    if args.starting_points != None:
        start = args.starting_points.split(",")
        try:
            start = [int(i) for i in start]
            start_type = "clusters"
        except:
            start = [float(i) for i in start]
            start_type = "mjds"

    """start main program"""
    # construct the filenames
    # datadir = os.path.dirname(os.path.abspath(str(__file__))) # replacing these lines with the following lines
    # allows APT to be run in the directory that the command was ran on
    # parfile = os.path.join(datadir, args.parfile)
    # timfile = os.path.join(datadir, args.timfile)
    parfile = Path(args.parfile)
    timfile = Path(args.timfile)
    original_path = Path.cwd()
    data_path = Path(args.data_path)

    #### FIXME When fulled implemented, DELETE the following line
    if socket.gethostname()[0] == "J":
        data_path = Path.cwd()
    else:
        data_path = Path("/data1/people/jdtaylor")
    ####
    os.chdir(data_path)

    toas = pint.toa.get_TOAs(timfile)
    toas.table["clusters"] = toas.get_clusters

    # every TOA, should never be edited
    all_toas_beggining = deepcopy(toas)

    pulsar_name = str(mb.get_model(parfile).PSR.value)
    if not Path(f"alg_saves/{pulsar_name}").exists():
        Path(f"alg_saves/{pulsar_name}").mkdir(parents=True)
    os.chdir(Path(f"alg_saves/{pulsar_name}"))

    set_F1_lim(args, parfile)

    mask_list, starting_cluster_list = starting_points(t)
    for mask_number, mask in enumerate(mask_list):

        print(
            f"Mask number {mask_number} has started. Starting cluster = {starting_cluster_list[mask_number]}"
        )

        m = mb.get_model(parfile)
        t, m = JUMP_adder_begginning(
            mask,
            toas,
            m,
            output_timfile=Path(f"{pulsar_name}_mask{mask_number}_{start}.tim"),
            output_parfile=Path(f"{pulsar_name}_mask{mask_number}_{start}.par"),
        )


if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print(
        f"Final Runtime (including plots): {end_time - start_time} seconds, or {(end_time - start_time) / 60.0} minutes"
    )
