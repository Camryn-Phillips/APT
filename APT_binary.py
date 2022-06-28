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
from pint.fitter import WLSFitter
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


class StartingJumpError(Exception):
    pass


def starting_points(toas, args=None):
    """
    Choose which cluster to NOT jump, i.e. where to start

    toas : TOA object
    args : command line arguments

    Returns
    tuple : (mask_list[:max_starts], starting_cluster_list[:max_starts])
    """
    t = deepcopy(toas)
    if "clusters" not in t.table.columns:
        t.table["clusters"] = t.get_clusters()
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
    mask: np.ndarray, toas, model, output_parfile, output_timfile
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

    return model, t


def JUMP_adder_begginning_cluster(
    mask: np.ndarray, toas, model, output_parfile, output_timfile
):
    """
    Adds JUMPs to a timfile as the begginning of analysis.
    This differs from JUMP_adder_begginning in that the jump flags
    are named based on the cluster number, not sequenitally from 0.

    mask : a mask to select which toas will not be jumped
    toas : TOA object
    model : model object
    output_parfile : name for par file to be written
    output_timfile : name for the tim file to be written

    Returns
    model, t
    """
    t = deepcopy(toas)
    if "clusters" not in t.table.columns:
        t.table["clusters"] = t.get_clusters()
    flag_name = "jump_tim"

    former_cluster = t.table[mask]["clusters"][0]
    j = 0
    for i, table in enumerate(t.table[~mask]):
        # if table["clusters"] != former_cluster:
        #     former_cluster = table["clusters"]
        #     j += 1
        table["flags"][flag_name] = str(table["clusters"])
    t.write_TOA_file(output_timfile)

    # model.jump_flags_to_params(t) doesn't currently work (need flag name to be "tim_jump" and even then it still won't work),
    # so the following is a workaround. This is likely related to issue 1294.
    ### (workaround surrounded in ###)
    with open(output_parfile, "w") as parfile:
        parfile.write(model.as_parfile())
        for i in set(t.table[~mask]["clusters"]):
            parfile.write(f"JUMP\t\t-{flag_name} {i}\t0 1 0\n")
    model = mb.get_model(output_parfile)
    ###

    return model, t


def phase_connector(
    toas: pint.toa.TOAs,
    model: pint.models.timing_model.TimingModel,
    connection_filter: str = "linear",
    cluster: int = "all",
    mjds_total: np.ndarray = None,
    **kwargs,
):
    """
    Makes sure each cluster is phase connected with itself.
    toas : TOAs object
    model : model object
    connection_filter1 : the basic filter for determing what is and what is not phase connected
        options: 'linear', 'polynomial'
    kwargs : an additional constraint on phase connection, can use any number of these
        options: 'wrap', 'degree'
    mjds_total : all mjds of TOAs, optional (may decrease runtime to include)

    Returns
    True
    """
    # print(f"cluster {cluster}")
    if cluster == "all":
        for cluster_number in set(toas["clusters"]):
            phase_connector(
                toas,
                model,
                connection_filter,
                cluster_number,
                mjds_total,
                **kwargs,
            )
        return True

    if mjds_total is None:
        mjds_total = toas.get_mjds().value
    if "clusters" not in toas.table.columns:
        toas.table["clusters"] = toas.get_clusters()
    if "pulse_number" not in toas.table.colnames:
        toas.compute_pulse_numbers(model)
    if "delta_pulse_number" not in toas.table.colnames:
        toas.table["delta_pulse_number"] = np.zeros(len(toas.get_mjds()))

    # this must be recalculated evertime because the residuals change
    # if a delta_pulse_number is added
    residuals_total = pint.residuals.Residuals(toas, model).calc_phase_resids()

    t = deepcopy(toas)
    cluster_mask = t.table["clusters"] == cluster
    t.select(cluster_mask)
    if len(t) == 1:  # no phase wrapping possible
        return True

    residuals = residuals_total[cluster_mask]
    mjds = mjds_total[cluster_mask]
    # if True: # for debugging
    #     print("#" * 100)
    #     plt.plot(mjds, residuals)
    #     plt.show()
    # print(cluster_mask)
    # print(residuals)

    if connection_filter == "linear":
        # residuals_dif = np.concatenate((residuals, np.zeros(1))) - np.concatenate(
        #     (np.zeros(1), residuals)
        # )
        residuals_dif = residuals[1:] - residuals[:-1]
        # "normalize" to speed up computation (it is slower on larger numbers)
        residuals_dif_normalized = residuals_dif / max(residuals_dif)

        # This is the filter for phase connected within a cluster
        if np.all(residuals_dif_normalized >= 0) or np.all(
            residuals_dif_normalized <= 0
        ):
            return True

        # another condition that means phase connection
        if kwargs.get("wraps") is True and max(np.abs(residuals_dif)) < 0.4:
            return True
        # print(max(np.abs(residuals_dif)))
        # attempt to fix the phase wraps, then run recursion to either fix it again
        # or verify it is fixed

        biggest = 0
        location_of_biggest = -1
        biggest_is_pos = True
        was_pos = True
        count = 0
        for i, is_pos in enumerate(residuals_dif_normalized >= 0):
            if is_pos and was_pos or (not is_pos and not was_pos):
                count += 1
                was_pos = is_pos
            elif is_pos or was_pos:  # slope flipped
                if count > biggest:
                    biggest = count
                    location_of_biggest = i - 1
                    biggest_is_pos = is_pos
                count = 1
                was_pos = is_pos
            # do this check one last time in case the last point is part of the
            # longest series of adjacent slopes
            if count >= biggest and np.abs(residuals_dif_normalized[i]) < np.abs(
                residuals_dif_normalized[i - 1]
            ):
                biggest = count
                # not i - 1 because that last point is like the (i + 1)th element
                location_of_biggest = i
                # flipping slopes
                biggest_is_pos = is_pos
        if biggest_is_pos:
            sign = 1
        else:
            sign = -1
        # everything to the left move down if positive
        t.table["delta_pulse_number"][: location_of_biggest - biggest + 1] = -1 * sign
        # everything to the right move up if positive
        t.table["delta_pulse_number"][location_of_biggest + 2 :] = sign

        # finally, apply these to the original set
        toas.table[cluster_mask] = t.table

    if connection_filter == "np.unwrap":
        # use the np.unwrap function somehow
        residuals_unwrapped = np.unwrap(np.array(residuals), period=1)
        t.table["delta_pulse_number"] = residuals_unwrapped - residuals
        # print(t.table["delta_pulse_number"])
        toas.table[cluster_mask] = t.table

        # this filter currently does not check itself
        return True

    # run it again, will return true and end the recursion if nothing needs to be fixed
    phase_connector(toas, model, connection_filter, cluster, mjds_total, **kwargs)


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
    toas.table["clusters"] = toas.get_clusters()
    mjds_total = toas.get_mjds().value

    # every TOA, should never be edited
    all_toas_beggining = deepcopy(toas)

    pulsar_name = str(mb.get_model(parfile).PSR.value)
    alg_saves_Path = Path(f"alg_saves/{pulsar_name}")
    if not alg_saves_Path.exists():
        alg_saves_Path.mkdir(parents=True)

    set_F1_lim(args, parfile)

    mask_list, starting_cluster_list = starting_points(toas)
    for mask_number, mask in enumerate(mask_list):
        starting_cluster = starting_cluster_list[mask_number]
        print(
            f"\nMask number {mask_number} has started. Starting cluster: {starting_cluster}\n"
        )

        mask_Path = Path(f"mask{mask_number}")
        alg_saves_mask_Path = alg_saves_Path / mask_Path
        if not alg_saves_mask_Path.exists():
            alg_saves_mask_Path.mkdir()

        m = mb.get_model(parfile)
        m, t = JUMP_adder_begginning_cluster(
            mask,
            toas,
            m,
            output_timfile=alg_saves_mask_Path / Path(f"{pulsar_name}_start.tim"),
            output_parfile=alg_saves_mask_Path / Path(f"{pulsar_name}_start.par"),
        )

        # start fitting for main binary parameters immediately
        if args.binary_model.lower() == "ell1":
            for param in ["PB", "TASC", "A1"]:
                getattr(m, param).frozen = False

        # a copy, with the flags included
        base_toas = deepcopy(t)

        # the following before the while loop is the very first fit with only one cluster not JUMPed

        phase_connector(
            t,
            m,
            "linear",
            cluster="all",
            mjds_total=mjds_total,
            wraps=True,
        )

        print("Fitting...")
        f = WLSFitter(t, m)
        print("BEFORE:", f.get_fitparams())
        print(f.fit_toas())

        print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
        print("RMS in phase is", f.resids.phase_resids.std())
        print("RMS in time is", f.resids.time_resids.std().to(u.us))
        print("\n Best model is:")
        print(f.model.as_parfile())

        t.table["delta_pulse_number"] = 0
        t.compute_pulse_numbers(f.model)
        # update the model
        m = f.model

        fig, ax = plt.subplots()
        ax.plot(mjds_total, pint.residuals.Residuals(t, m).calc_phase_resids())
        ax.set_xlabel("MJD")
        ax.set_ylabel("Residual (phase)")
        plt.savefig(
            alg_saves_mask_Path / Path(f"{pulsar_name}_iteration{iteration}.png")
        )

        # something is wrong if the reduced chisq is greater that 3 at this stage,
        # so APTB should not waste time attempting to coerce a solution
        if pint.residuals.Residuals(t, m).chi2_reduced > 3:
            plt.plot(
                mjds_total, pint.residuals.Residuals(t, m).calc_phase_resids(), "o"
            )
            plt.show()
            response = input(
                f"The reduced chisq is {pint.residuals.Residuals(t, m).chi2_reduced}.\n"
                + "This is adnormally high, should APTB continue anyway? (y/n)"
            )
            if response.lower() != "y":
                raise StartingJumpError(
                    "Reduced chisq adnormally high, quitting program."
                )

        iteration = 0
        while True:
            # the main loop of the algorithm
            iteration += 1
            # print(f"iteration: {iteration}", end="#" * 200 + "\n")

            residuals_total = pint.residuals.Residuals(t, m).calc_phase_resids()
            # temp_mask = t.table["clusters"] == 9
            # plt.plot(mjds_total[temp_mask], residuals_total[temp_mask], 'o')
            # plt.show()
            # want to phase connect toas within a cluster first:
            phase_connector(
                t,
                m,
                "linear",
                cluster="all",
                mjds_total=mjds_total,
                wraps=True,
            )

            fig, ax = plt.subplots(3, 1, figsize=(15, 10))
            y1 = pint.residuals.Residuals(t, m).calc_phase_resids()

            print("Fitting...")
            f = WLSFitter(t, m)
            print("BEFORE:", f.get_fitparams())
            print(f.fit_toas())

            print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
            print("RMS in phase is", f.resids.phase_resids.std())
            print("RMS in time is", f.resids.time_resids.std().to(u.us))
            print("\n Best model is:")
            print(f.model.as_parfile())

            t.table["delta_pulse_number"] = 0
            t.compute_pulse_numbers(f.model)
            # residuals = pint.residuals.Residuals(t, f.model)

            ###
            residuals = pint.residuals.Residuals(t, f.model)
            # y0 = residuals_total
            # ms = 2
            # ax[0].plot(mjds_total, y0, "o", markersize=ms)
            # ax[1].plot(mjds_total, y1, "o", markersize=ms)
            # ax[2].plot(mjds_total, y2, "o", markersize=ms)
            plt.plot()
            plt.savefig(
                alg_saves_mask_Path / Path(f"{pulsar_name}_iteration{iteration}.par")
            )

            break
        break


if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print(
        f"Final Runtime (including plots): {end_time - start_time} seconds, or {(end_time - start_time) / 60.0} minutes"
    )
