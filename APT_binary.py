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
from APT import get_closest_cluster, solution_compare, bad_points
import APT_binary_extension
import argparse
from loguru import logger as log


class StartingJumpError(Exception):
    pass


def starting_points(toas, args=None, score_function="original", **kwargs):
    """
    Choose which cluster to NOT jump, i.e. where to start

    Parameters
    ----------
    toas : TOAs object
    args : command line arguments
    score_function : which function to use to rank TOAs

    Returns
    -------
    tuple : (mask_list[:max_starts], starting_cluster_list[:max_starts])
    """
    t = deepcopy(toas)
    if "clusters" not in t.table.columns:
        t.table["clusters"] = t.get_clusters()
    mjd_values = t.get_mjds().value
    dts = np.fabs(mjd_values - mjd_values[:, np.newaxis]) + np.eye(len(mjd_values))

    if score_function == "original":
        score_list = (1.0 / dts).sum(axis=1)
    # different powers give different starting masks
    elif score_function == "original_different_power":
        power = kwargs.get("power", 0.3)
        score_list = (1.0 / dts**power).sum(axis=1)

    mask_list = []
    starting_cluster_list = []
    # f = pint.fitter.WLSFitter(t, m)
    # f.fit_toas()
    i = -1
    while score_list.any():
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


def JUMP_adder_begginning_cluster(
    mask: np.ndarray, t, model, output_parfile, output_timfile
):
    """
    Adds JUMPs to a timfile as the begginning of analysis.
    This differs from JUMP_adder_begginning in that the jump flags
    are named based on the cluster number, not sequenitally from 0.

    Parameters
    ----------
    mask : a mask to select which toas will not be jumped
    t : TOAs object
    model : model object
    output_parfile : name for par file to be written
    output_timfile : name for the tim file to be written

    Returns
    -------
    model, t
    """
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
    residuals=None,
    **kwargs,
):
    """
    Makes sure each cluster is phase connected with itself.

    Parameters
    ----------
    toas : TOAs object
    model : model object
    connection_filter : the basic filter for determing what is and what is not phase connected
        options: 'linear', 'polynomial'
    kwargs : an additional constraint on phase connection, can use any number of these
        options: 'wrap', 'degree'
    mjds_total : all mjds of TOAs, optional (may decrease runtime to include)

    Returns
    -------
    None
        The only intention of the return statements is to end the function
    """
    # print(f"cluster {cluster}")

    # this function can perform the neccesary actions on all clusters using either
    # np.unwrap or recursion

    # these need to be reset before unwrapping occurs
    toas.table["delta_pulse_number"] = np.zeros(len(toas.get_mjds()))
    toas.compute_pulse_numbers(model)

    if cluster == "all":
        if connection_filter == "np.unwrap":
            t = deepcopy(toas)
            mask_with_closest = kwargs.get(
                "mask_with_closest", np.zeros(len(toas), dtype=bool)
            )
            t.select(~mask_with_closest)
            residuals_unwrapped = np.unwrap(
                np.array(residuals[~mask_with_closest]), period=1
            )
            t.table["delta_pulse_number"] = (
                residuals_unwrapped - residuals[~mask_with_closest]
            )
            toas.table[~mask_with_closest] = t.table
            return
        for cluster_number in set(toas["clusters"]):
            phase_connector(
                toas,
                model,
                connection_filter,
                cluster_number,
                mjds_total,
                residuals,
                **kwargs,
            )
        return

    if mjds_total is None:
        mjds_total = toas.get_mjds().value
    if "clusters" not in toas.table.columns:
        toas.table["clusters"] = toas.get_clusters()
    # if "pulse_number" not in toas.table.colnames:  ######
    # toas.compute_pulse_numbers(model)
    # if "delta_pulse_number" not in toas.table.colnames:
    # toas.table["delta_pulse_number"] = np.zeros(len(toas.get_mjds()))

    # this must be recalculated evertime because the residuals change
    # if a delta_pulse_number is added
    residuals_total = pint.residuals.Residuals(toas, model).calc_phase_resids()

    t = deepcopy(toas)
    cluster_mask = t.table["clusters"] == cluster
    t.select(cluster_mask)
    if len(t) == 1:  # no phase wrapping possible
        return

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
            return

        # another condition that means phase connection
        if kwargs.get("wraps") and max(np.abs(residuals_dif)) < 0.4:
            return
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
        return

    # run it again, will return None and end the recursion if nothing needs to be fixed
    phase_connector(
        toas, model, connection_filter, cluster, mjds_total, residuals, **kwargs
    )


def save_state(
    f,
    m,
    t,
    mjds,
    pulsar_name,
    iteration,
    folder,
    args,
    save_plot=False,
    show_plot=False,
    mask_with_closest=None,
    **kwargs,
):
    """
    Records the par and tim files of the current state and graphs a figure.
    It also checks if A1 is negative and if it is, it will ask the user if APTB
    should attempt to fix it. When other binary models are implemented, other
    types of checks for their respective models would need to be implemented here,
    provided easy fixes are available.

    Parameters
    ----------
    f : fitter object
    m : model object
    t : TOAs object
    mjds : all mjds of TOAS
    pulsar_name : name of pulsar ('fake_#' in the test cases)
    iteration : how many times the while True loop logic has repeated + 1
    folder : path to directory for saving the state
    args : command line arguments
    save_plot : whether to save the plot or not (defaults to False)
    show_plot : whether to display the figure immediately after generation, iterupting the program (defaults to False)
    mask_with_closest : the mask with the non-JUMPed TOAs
    kwargs : additional keyword arguments

    Returns
    -------
    bool
        whether or not a managed error occured (True if no issues). Nonmanaged errors will completely hault the program
    """
    m_copy = deepcopy(m)
    t = deepcopy(t)

    t.write_TOA_file(folder / Path(f"{pulsar_name}_{iteration}.tim"))
    try:
        with open(folder / Path(f"{pulsar_name}_{iteration}.par"), "w") as file:
            file.write(m_copy.as_parfile())
    except ValueError as e:
        if str(e)[:47] != "Projected semi-major axis A1 cannot be negative":
            raise e
        response = input(
            f"\nA1 detected to be negative! Should APTB attempt to correct it? (y/n)"
        )
        if response.lower() != "y":
            print(e)
            print("Moving onto next mask")
            return False
        print(f"(A1 = {m_copy.A1.value} and TASC = {m_copy.TASC.value})\n")
        if args.binary_model.lower() == "ell1":
            # the fitter cannot distinguish between a sinusoid and the negative
            # sinusoid shifted by a half period
            m.A1.value = -m.A1.value
            m.TASC.value = m.TASC.value + m.PB.value / 2
            m_copy = deepcopy(m)
            with open(folder / Path(f"{pulsar_name}_{iteration}.par"), "w") as file:
                file.write(m_copy.as_parfile())

    if save_plot or show_plot:
        fig, ax = plt.subplots(figsize=(12, 7))

        fig, ax = plot_plain(
            f,
            mjds,
            t,
            m,
            iteration,
            fig,
            ax,
            mask_with_closest=mask_with_closest,
        )

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(folder / Path(f"{pulsar_name}_{iteration}.png"))
        plt.close()
    # if the function has gotten to this point, (likely) no issues have occured
    return True


def plot_plain(f, mjds, t, m, iteration, fig, ax, mask_with_closest=None):
    """
    A helper function for save_state. Graphs the time & phase residuals.
    Including mask_with_closests colors red the TOAs not JUMPed.

    This function is largely inherited from APT with some small, but crucial, changes

    Parameters
    ----------
    f, mjds, t, m, iteration, mask)wtih_closest : identical to save_state
    fig : matplotlib.pyplot.subplots figure object
    ax : matplotlib.pyplot.subplots axis object

    Returns
    -------
    fig, ax
        These then will be handled, and likely saved, by the rest of the save_state function
    """
    # plot post fit residuals with error bars
    model0 = deepcopy(m)
    r = pint.residuals.Residuals(t, model0).time_resids.to(u.us).value

    xt = mjds
    ax.errorbar(
        mjds,
        r,
        t.get_errors().to(u.us).value,
        fmt=".b",
        label="post-fit",
    )

    if mask_with_closest is not None:
        ax.errorbar(
            mjds[mask_with_closest],
            r[mask_with_closest],
            t.get_errors().to(u.us).value[mask_with_closest],
            fmt=".r",
        )

    # string of fit parameters for plot title
    fitparams = ""
    if f:
        for param in f.get_fitparams().keys():
            if "JUMP" in str(param):
                fitparams += f"J{str(param)[4:]} "
            else:
                fitparams += str(param) + " "

    # notate the pulsar name, iteration, and fit parameters
    plt.title(f"{m.PSR.value} Post-Fit Residuals {iteration} | fit params: {fitparams}")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (us)")

    # set the y limit to be just above and below the max and min points
    yrange = abs(max(r) - min(r))
    ax.set_ylim(max(r) + 0.1 * yrange, min(r) - 0.1 * yrange)

    # scale to the edges of the points or the edges of the random models, whichever is smaller
    width = max(mjds) - min(mjds)
    if (min(mjds) - 0.1 * width) < (min(xt) - 20) or (max(mjds) + 0.1 * width) > (
        max(xt) + 20
    ):
        ax.set_xlim(min(xt) - 20, max(xt) + 20)

    else:
        ax.set_xlim(min(mjds) - 0.1 * width, max(mjds) + 0.1 * width)

    plt.grid()

    def us_to_phase(x):
        return (x / (10**6)) * m.F0.value

    def phase_to_us(y):
        return (y / m.F0.value) * (10**6)

    # include secondary axis to show phase
    secaxy = ax.secondary_yaxis("right", functions=(us_to_phase, phase_to_us))
    secaxy.set_ylabel("residuals (phase)")

    return fig, ax


def Ftest_param(r_model, fitter, param_name):
    """
    do an F-test comparing a model with and without a particular parameter added

    Note: this is NOT a general use function - it is specific to this code and cannot be easily adapted to other scripts

    Parameters
    ----------
    r_model : timing model to be compared
    fitter : fitter object containing the toas to compare on
    param_name : name of the timing model parameter to be compared

    Returns
    -------
    float
        the value of the F-test
    """
    # read in model and toas
    m_plus_p = deepcopy(r_model)
    toas = deepcopy(fitter.toas)

    # set given parameter to unfrozen
    getattr(m_plus_p, param_name).frozen = False

    # make a fitter object with the chosen parameter unfrozen and fit the toas using the model with the extra parameter
    f_plus_p = pint.fitter.WLSFitter(toas, m_plus_p)
    f_plus_p.fit_toas()

    # calculate the residuals for the fit with (m_plus_p_rs) and without (m_rs) the extra parameter
    m_rs = pint.residuals.Residuals(toas, fitter.model)
    m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)

    # calculate the Ftest, comparing the chi2 and degrees of freedom of the two models
    Ftest_p = pint.utils.FTest(
        float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
    )
    # The Ftest determines how likely (from 0. to 1.) that improvement due to the new parameter is due to chance and not necessity
    # Ftests close to zero mean the parameter addition is necessary, close to 1 the addition is unnecessary,
    # and NaN means the fit got worse when the parameter was added

    # if the Ftest returns NaN (fit got worse), iterate the fit until it improves to a max of 3 iterations.
    # It may have gotten stuck in a local minima
    counter = 0

    while not Ftest_p and counter < 3:
        counter += 1

        f_plus_p.fit_toas()
        m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)

        # recalculate the Ftest
        Ftest_p = pint.utils.FTest(
            float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
        )

    # print the Ftest for the parameter and return the value of the Ftest
    print("Ftest" + param_name + ":", Ftest_p)
    return Ftest_p


def do_Ftests(t, m, mask_with_closest, args):
    """
    Does the Ftest on the neccesarry parameters

    Parameters
    ----------
    t : TOAs object
    m : model object
    mask_with_closest : the clusters
    args : command line arguments

    Returns
    m (with particular parameters now potentially unfrozen)
    """

    # fit toas with new model
    f = pint.fitter.WLSFitter(t, m)
    f.fit_toas()

    t_copy = deepcopy(t)
    t_copy.select(mask_with_closest)

    # calculate the span of fit toas for comparison to minimum parameter spans
    span = t_copy.get_mjds().max() - t_copy.get_mjds().min()
    print("Current fit TOAs span:", span)

    Ftests = dict()
    f_params = []
    # TODO: need to take into account if param isn't setup in model yet

    # make list of already fit parameters
    for param in m.params:
        if getattr(m, param).frozen == False:
            f_params.append(param)

    # if span is longer than minimum parameter span and parameter hasn't been added yet, do Ftest to see if parameter should be added
    if "RAJ" not in f_params and span > args.RAJ_lim * u.d:
        Ftest_R = Ftest_param(m, f, "RAJ")
        Ftests[Ftest_R] = "RAJ"

    if "DECJ" not in f_params and span > args.DECJ_lim * u.d:
        Ftest_D = Ftest_param(m, f, "DECJ")
        Ftests[Ftest_D] = "DECJ"

    if "F1" not in f_params and span > args.F1_lim * u.d:
        Ftest_F = Ftest_param(m, f, "F1")
        Ftests[Ftest_F] = "F1"

    m, t, f, f_params, span, Ftests, args = APT_binary_extension.do_Ftests_binary(
        m, t, f, f_params, span, Ftests, args
    )

    # if args.binary_model.lower() == "ell1":
    #     if "EPS1" not in f_params and span > args.EPS1_lim * u.d:
    #         Ftest_F = Ftest_param(m, f, "EPS1")
    #         Ftests[Ftest_F] = "EPS1"
    #     if "EPS2" not in f_params and span > args.EPS2_lim * u.d:
    #         Ftest_F = Ftest_param(m, f, "EPS2")
    #         Ftests[Ftest_F] = "EPS2"

    # remove possible boolean elements from Ftest returning False if chi2 increases
    Ftests_keys = [key for key in Ftests.keys() if type(key) != bool]

    # if no Ftests performed, continue on without change
    if not bool(Ftests_keys):
        if span > 100 * u.d:
            print("F0, RAJ, DECJ, and F1 have all been added")

    # if smallest Ftest of those calculated is less than the given limit, add that parameter to the model. Otherwise add no parameters
    elif min(Ftests_keys) < args.Ftest_lim:
        add_param = Ftests[min(Ftests_keys)]
        print("adding param ", add_param, " with Ftest ", min(Ftests_keys))
        getattr(m, add_param).frozen = False

    return m


def set_F1_lim(args, parfile):
    """
    if F1_lim not specified in command line, calculate the minimum span based on general F0-F1 relations from P-Pdot diagram

    Parameters
    ----------
    args : command line arguments
    parfile : parfile

    Returns
    -------
    None
    """

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


def set_binary_pars_lim(args):
    if args.binary_model.lower() == "ell1":
        args.EPS1_lim = 75
        args.EPS2_lim = 75

    return args


def quadratic_phase_wrap_checker(
    m,
    t,
    mask_with_closest,
    closest_cluster_mask,
    b,
    maxiter_while,
    closest_cluster,
):
    # TODO may need to use the F-test in here
    """
    Checks for phase wraps using the Freire and Ridolfi method.

    Parameters
    ----------
    m : model
    t : TOAs object
    mask_with_closest : the mask with the non-JUMPed TOAs
    closest_cluster_mask : the mask with only the closest cluster
    b : which phase wraps to use as the sample
    maxiter_while : maxiter for WLS fitting
    closest_cluster : the cluster number of the closest cluster

    Returns
    -------
    f, t
    """
    # run from highest to lowest b until no error is raised
    for b_i in range(b, -1, -1):
        chisq_samples = {}
        f = WLSFitter(t, m)
        try:
            for wrap in [-b_i, 0, b_i]:
                # m_copy = deepcopy(m)
                t.table["delta_pulse_number"][closest_cluster_mask] = wrap
                f.fit_toas(maxiter=maxiter_while)
                # m_copy = do_Ftests(t, m_copy, mask_with_closest, args)
                chisq_samples[wrap] = f.resids.chi2_reduced
                # if the loop did not encounter an error, then reassign b and end the loop
                b = b_i
                break

        except Exception as e:
            # try running it again with a lower b
            # sometimes if b is too high, APTB wants to fit the samples phase
            # wraps by allowing A1 to be negative, which should be avoided
            if b < 1:
                log.warning(f"QuadraticPhaseWrapCheckerError: b_i is {b_i}")
                print(e)
                response = input(
                    "Should APTB continue anyway, assuming no phase wraps for the next cluster? (y/n)"
                )
                if response == "y":
                    return WLSFitter(t, m), t
                else:
                    raise e

    min_wrap = round(
        (b / 2)
        * (chisq_samples[-b] - chisq_samples[b])
        / (chisq_samples[b] + chisq_samples[-b] - 2 * chisq_samples[0])
    )
    # check +1, 0, and -1 wrap from min_wrap just to be safe
    # TODO: an improvement would be for APTB to continue down each path
    # of +1, 0, and -1. However, knowing when to cut off a branch
    # would have to implemented.
    t_wrap_dict = {}
    f_wrap_dict = {}
    chisq_wrap = {}
    for wrap in range(-1, 2):
        t.table["delta_pulse_number"][closest_cluster_mask] = min_wrap + wrap
        f.fit_toas(maxiter=maxiter_while)

        # t_plus_minus["delta_pulse_number"] = 0
        # t_plus_minus.compute_pulse_numbers(f_plus_minus.model)

        chisq_wrap[f.resids.chi2_reduced] = min_wrap + wrap
        t_wrap_dict[min_wrap + wrap] = deepcopy(t)
        f_wrap_dict[min_wrap + wrap] = deepcopy(f)

    min_chisq = min(chisq_wrap.keys())

    min_wrap_number_total = chisq_wrap[min_chisq]

    t = t_wrap_dict[min_wrap_number_total]
    f = f_wrap_dict[min_wrap_number_total]

    print(
        f"Attemping a phase wrap of {min_wrap_number_total} on closest cluster (cluster {closest_cluster})."
    )

    return f, t  # , chisq_samples, min_wrap_number_total

    # t_closest_cluster.table["delta_pulse_number"] = min_wrap_number_total
    # t.table[closest_cluster_mask] = t_closest_cluster.table


def APT_argument_parse(parser, argv):
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="tim file to read toas from")
    parser.add_argument(
        "--binary_model",
        help="which binary pulsar model to use.",
        choices=["ELL1", "ell1"],
        default=None,
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
    parser.add_argument(
        "--multiprocessing",
        help="whether to include multiprocessing or not (t/f)",
        type=str,
        default="f",
    )
    parser.add_argument(
        "--check_phase_wraps",
        help="whether to check for phase wraps or not (t/f)",
        type=str,
        default="t",
    )
    parser.add_argument(
        "--maxiter_while",
        help="sets the maxiter argument for f.fit_toas for fittings done within the while loop",
        type=int,
        default=1,
    )

    args = parser.parse_args(argv)
    # interpret strings as booleans
    args.check_bad_points = args.check_bad_points.lower()[0] == "t"
    args.try_poly_extrap = args.try_poly_extrap.lower()[0] == "t"
    args.plot_poly_extrap = args.plot_poly_extrap.lower()[0] == "t"
    args.plot_bad_points = args.plot_bad_points.lower()[0] == "t"
    args.plot_final = args.plot_final.lower()[0] == "t"
    args.multiprocessing = args.multiprocessing.lower()[0] == "t"
    args.check_phase_wraps = args.check_phase_wraps.lower()[0] == "t"

    return args, parser


def main(args, parser, mask_selector=None):

    if args is None:
        # need to run this again if not previously specified
        parser = argparse.ArgumentParser(
            description="PINT tool for agorithmically timing binary pulsars."
        )
        args, parser = APT_argument_parse(parser, argv=None)
        print(args)
        raise Exception("test stop")

    # import argparse
    # import sys

    # args, parser, mask_selector = main_args
    flag_name = "jump_tim"

    # read in arguments from the command line

    """required = parfile, timfile"""
    """optional = starting points, param ranges"""
    # parser = argparse.ArgumentParser(
    #     description="PINT tool for agorithmically timing binary pulsars."
    # )

    # args, parser = APT_argument_parse(parser, argv=None)

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

    # this sets the maxiter argument for f.fit_toas for fittings done within the while loop
    # (default to 2)
    maxiter_while = args.maxiter_while

    mask_list, starting_cluster_list = starting_points(toas)
    for mask_number, mask in enumerate(mask_list):
        starting_cluster = starting_cluster_list[mask_number]

        # for multiprocessing, the mask_selector tells each iteration of main to skip
        # all but one of the masks
        if mask_selector is not None and mask_number != mask_selector:
            continue
        if starting_cluster != 22:
            continue
        print(
            f"\nMask number {mask_number} has started. Starting cluster: {starting_cluster}\n"
        )

        mask_Path = Path(f"mask{mask_number}_cluster{starting_cluster}")
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
        t.compute_pulse_numbers(m)
        args.binary_model = m.BINARY.value
        args = set_binary_pars_lim(args)

        # start fitting for main binary parameters immediately
        if args.binary_model.lower() == "ell1":
            for param in ["PB", "TASC", "A1"]:
                getattr(m, param).frozen = False

        # a copy, with the flags included
        base_toas = deepcopy(t)

        # the following before the while loop is the very first fit with only one cluster not JUMPed

        # want to phase connect toas within a cluster first:
        residuals_start = pint.residuals.Residuals(t, m).calc_phase_resids()

        phase_connector(
            t,
            m,
            "np.unwrap",
            cluster="all",
            mjds_total=mjds_total,
            residuals=residuals_start,
            wraps=True,
        )

        if not save_state(
            m=m,
            t=t,
            mjds=mjds_total,
            pulsar_name=pulsar_name,
            f=None,
            args=args,
            folder=alg_saves_mask_Path,
            iteration="start_right_after_phase",
            save_plot=True,
        ):
            # try next mask
            continue

        print("Fitting...")
        f = WLSFitter(t, m)
        print("BEFORE:", f.get_fitparams())
        # changing maxiter here may have some effects
        print(f.fit_toas(maxiter=2))

        print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
        print("RMS in phase is", f.resids.phase_resids.std())
        print("RMS in time is", f.resids.time_resids.std().to(u.us))
        print("\n Best model is:")
        print(f.model.as_parfile())

        # new model so need to update table
        t.table["delta_pulse_number"] = 0
        t.compute_pulse_numbers(f.model)

        # update the model
        m = f.model

        if not save_state(
            f,
            m,
            t,
            mjds_total,
            pulsar_name,
            args=args,
            folder=alg_saves_mask_Path,
            iteration="start",
            save_plot=True,
        ):
            continue

        # something is certaintly wrong if the reduced chisq is greater that 3 at this stage,
        # so APTB should not waste time attempting to coerce a solution
        chisq_start = pint.residuals.Residuals(t, m).chi2_reduced
        log.info(f"The reduced chisq after the initial fit is {round(chisq_start, 3)}")
        if chisq_start > 3:
            log.warning(
                f"The reduced chisq after the initial fit is {round(chisq_start, 3)}"
            )
            plt.plot(
                mjds_total, pint.residuals.Residuals(t, m).calc_phase_resids(), "o"
            )
            plt.show()
            response = input(
                f"The reduced chisq is {pint.residuals.Residuals(t, m).chi2_reduced}.\n"
                + "This is adnormally high, should APTB continue anyway? (y/n)"
            )
            if response.lower() != "y":
                log.warning("StartingJumpError")
                raise StartingJumpError(
                    "Reduced chisq adnormally high, quitting program."
                )

        # mask_with_closest will be everything not JUMPed, as well as the next clusters
        # to be de JUMPed

        # the following list comprehension allows a JUMP number to be found
        # by indexing this list with its cluster number. The wallrus operator
        # is used for brevity.
        j = 0
        cluster_to_JUMPs = [
            f"JUMP{(j:=j+1)}" if i != starting_cluster else ""
            for i in range(t.table["clusters"][-1] + 1)
        ]
        skip_mask = False
        bad_mjds = []
        mask_with_closest = deepcopy(mask)
        iteration = 0
        while True:
            # the main loop of the algorithm
            iteration += 1

            # find the closest cluster
            closest_cluster, dist = get_closest_cluster(
                deepcopy(t), deepcopy(t[mask_with_closest]), deepcopy(t)
            )
            print("closest cluster:", closest_cluster)
            if closest_cluster is None:
                # end the program
                break

            closest_cluster_mask = t.table["clusters"] == closest_cluster

            # TODO add polyfit here
            # random models can cover this instead
            # do slopes match from next few clusters, or does a quadratic fit

            mask_with_closest = np.logical_or(mask_with_closest, closest_cluster_mask)

            closest_cluster_JUMP = cluster_to_JUMPs[closest_cluster]
            getattr(m, closest_cluster_JUMP).frozen = True
            getattr(m, closest_cluster_JUMP).value = 0
            getattr(m, closest_cluster_JUMP).uncertainty = 0
            # seeing if removing the value of every JUMP helps
            # for JUMP in cluster_to_JUMPs:
            #     if JUMP and not getattr(m, JUMP).frozen:
            #         getattr(m, JUMP).value = 0
            #         getattr(m, JUMP).uncertainty = 0

            t.table["delta_pulse_number"] = 0
            t.compute_pulse_numbers(m)
            residuals = pint.residuals.Residuals(t, m).calc_phase_resids()

            # save_state(
            #     f,
            #     m,
            #     t,
            #     mjds_total,
            #     pulsar_name,
            #     args=args,
            #     folder=alg_saves_mask_Path,
            #     iteration=f"prepre{iteration}",
            #     save_plot=True,
            # )

            phase_connector(
                t,
                m,
                "np.unwrap",
                cluster="all",
                mjds_total=mjds_total,
                residuals=residuals,
                mask_with_closest=mask_with_closest,
                wraps=True,
            )
            if not save_state(
                f,
                m,
                t,
                mjds_total,
                pulsar_name,
                args=args,
                folder=alg_saves_mask_Path,
                iteration=f"prefit{iteration}",
                save_plot=True,
                mask_with_closest=mask_with_closest,
            ):
                skip_mask = True
                break

            # TODO add pontential phase wraps here
            if args.check_phase_wraps:
                f, t = quadratic_phase_wrap_checker(
                    m,
                    t,
                    mask_with_closest,
                    closest_cluster_mask,
                    b=5,
                    maxiter_while=maxiter_while,
                    closest_cluster=closest_cluster,
                )
            else:
                f = pint.fitter.WLSFitter(t, m)
                f.fit_toas(maxiter=maxiter_while)
                t.table["delta_pulse_number"] = 0
                t.compute_pulse_numbers(f.model)
            m = f.model

            # use random models, or design matrix method to determine if next
            # cluster is within the error space. If the next cluster is not
            # within the error space, check for phase wraps.

            # TODO use random models or design matrix here

            # # TODO add pontential phase wraps here
            # m, t = quadratic_phase_wrap_checker(m, t, mask_with_closest, closest_cluster_mask)

            # TODO add check_bad_points here-ish

            # use the F-test to determine if another parameter should be fit

            m = do_Ftests(t, m, mask_with_closest, args)

            # fit
            f = pint.fitter.WLSFitter(t, m)
            f.fit_toas(maxiter=maxiter_while)

            m = f.model

            if not save_state(
                f,
                m,
                t,
                mjds_total,
                pulsar_name,
                args=args,
                folder=alg_saves_mask_Path,
                iteration=iteration,
                save_plot=True,
                mask_with_closest=mask_with_closest,
            ):
                skip_mask = True
                break

            # repeat

        if skip_mask:
            continue

        # try fitting with any remaining unfit parameters included and see if the fit is better for it
        m_plus = deepcopy(m)
        getattr(m_plus, "RAJ").frozen = False
        getattr(m_plus, "DECJ").frozen = False
        getattr(m_plus, "F1").frozen = False

        if args.binary_model.lower() == "ell1":
            getattr(m_plus, "EPS1").frozen = False
            getattr(m_plus, "EPS2").frozen = False

        f_plus = pint.fitter.WLSFitter(t, m_plus)
        f_plus.fit_toas(maxiter=maxiter_while)

        # residuals
        r = pint.residuals.Residuals(t, f.model)
        r_plus = pint.residuals.Residuals(t, f_plus.model)
        if r_plus.chi2 <= r.chi2:
            f = deepcopy(f_plus)

        # save final model as .fin file
        print("Final Model:\n", f.model.as_parfile())

        # save as .fin
        fin_name = Path(f.model.PSR.value + ".fin")
        with open(alg_saves_mask_Path / fin_name, "w") as finfile:
            finfile.write(f.model.as_parfile())

        # plot final residuals if plot_final True
        xt = t.get_mjds()
        plt.clf()
        plt.close()
        fig, ax = plt.subplots()
        twinx = ax.twinx()
        ax.errorbar(
            xt.value,
            pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,
            t.get_errors().to(u.us).value,
            fmt=".b",
            label="post-fit (time)",
        )
        twinx.errorbar(
            xt.value,
            pint.residuals.Residuals(t, f.model).phase_resids,
            t.get_errors().to(u.us).value * float(f.model.F0.value) / 1e6,
            fmt=".b",
            label="post-fit (phase)",
        )
        ax.set_title(f"{m.PSR.value} Final Post-Fit Timing Residuals")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Residual (us)")
        twinx.set_ylabel("Residual (phase)", labelpad=15)
        span = (0.5 / float(f.model.F0.value)) * (10**6)
        plt.grid()

        time_end_main = time.monotonic()
        print(
            f"Final Runtime (not including plots): {time_end_main - start_time} seconds, or {(time_end_main - start_time) / 60.0} minutes"
        )
        if args.plot_final:
            plt.show()

        fig.savefig(
            alg_saves_mask_Path / Path(f"{pulsar_name}_final.png"), bbox_inches="tight"
        )
        plt.close()

        # if success, stop trying and end programl
        if pint.residuals.Residuals(t, f.model).chi2_reduced < float(args.chisq_cutoff):
            print(
                "SUCCESS! A solution was found with reduced chi2 of",
                pint.residuals.Residuals(t, f.model).chi2_reduced,
                "after",
                iteration,
                "iterations",
            )
            if args.parfile_compare:
                while True:
                    try:
                        identical_solution = solution_compare(
                            args.parfile_compare,
                            alg_saves_mask_Path / Path(f"{f.model.PSR.value}.fin"),
                            timfile,
                        )
                        # if succesful, break the loop
                        break

                    # if an error occurs, attempt again with the correct solution path
                    except FileNotFoundError as e:
                        args.parfile_compare = input(
                            "Solution file not found. Input the full path here or enter 'q' to quit: "
                        )
                        if args.parfile_compare == "q":
                            identical_solution = "Unknown"
                            break
                        # else, try the loop again

                if identical_solution != "Unknown":
                    print(
                        f"\n\nThe .fin solution and comparison solution ARE {['NOT', ''][identical_solution]} identical.\n\n"
                    )
                else:
                    print(
                        f"\nSolution compare failed because the solution file could not be found."
                    )

            print(f"The input parameters for this fit were:\n {args}")
            print(
                f"\nThe final fit parameters are: {[key for key in f.get_fitparams().keys()]}"
            )
            starting_TOAs = t[mask]
            print(f"starting points (clusters):\n {starting_TOAs.get_clusters()}")
            print(f"starting points (MJDs): {starting_TOAs.get_mjds()}")
            print(f"TOAs Removed (MJD): {bad_mjds}")
            break
    return "Completed"


if __name__ == "__main__":
    import sys

    pint.logging.setup(level="WARNING")
    global args, parser

    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        description="PINT tool for agorithmically timing binary pulsars."
    )
    args, parser = APT_argument_parse(parser, argv=None)

    if args.multiprocessing:
        from multiprocessing import Pool

        # from pathos.multiprocessing import ProcessingPool as Pool
        # from multiprocessing.pool import ThreadPool as Pool

        print("\n\nMultiprocessing in use!\n")
        log.warning(
            "Using muliprocessing while running APTB on several pulsars will not notably increase\n"
            + "the efficiency due to multiprocessing using multiple cores anyway."
        )
        with Pool(args.max_starts) as p:
            # settting args and parser to None is needed to prevent the multiprocessing
            # package from trying to pickle them, giving an error.
            # doing it this way also prevents the need for globals
            print(
                p.starmap(
                    main,
                    [
                        (None, None, mask_selector)
                        for mask_selector in range(args.max_starts)
                    ],
                )
            )
    else:
        main(args, parser)
    end_time = time.monotonic()
    print(
        f"Final Runtime (including plots): {end_time - start_time} seconds, or {(end_time - start_time) / 60.0} minutes"
    )
