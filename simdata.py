#!/usr/bin/env python
import pint.toa
import pint.models
import pint.utils
import os
import numpy.random as r
import numpy as np
import astropy.units as u
from copy import deepcopy
import socket
from astropy.time import TimeDelta
import sys
import pint.logging
from loguru import logger as log
from pathlib import Path


log.remove()
log.add(
    sys.stderr,
    level="WARNING",
    colorize=True,
    format=pint.logging.format,
    filter=pint.logging.LogFilter(),
)

import pint.fitter
import pint.simulation
import pint.residuals


def write_solfile(args, sol_name):
    """
    write a parfile with the "solution"  parameter values of the simulated pulsar

    :param args: command line input arguments
    :param sol_name: name of the output .sol file
    """

    solfile = open(Path(f"./fake_data/{sol_name}"), "w")

    h = r.randint(0, 24)
    m = r.randint(0, 60)
    s = r.uniform(0, 60)

    # randomly assign values in appropriate ranges to RAJ
    if args.RAJ_value == None:
        raj = (str(h) + ":" + str(m) + ":" + str(s), args.RAJ_error)
    else:
        raj = (args.RAJ_value, args.RAJ_error)

    # not just -90 to 90, has to be spherically symmetric
    d = int((np.arcsin(2.0 * r.random() - 1.0)) * 180.0 / np.pi)
    arcm = r.randint(0, 60)
    arcs = r.uniform(0, 60)

    # randomly assign values in appropriate ranges to DECJ
    if args.DECJ_value == None:
        decj = (str(d) + ":" + str(arcm) + ":" + str(arcs), args.DECJ_error)
    else:
        decj = (args.DECJ_value, args.DECJ_error)

    # randomly assign values in apporiate range to F0 (100-800 Hz is millisecond pulsar range. Slow pulsars are 2-20 Hz range)
    if args.F0_value == None:
        f0 = (r.uniform(125, 250), args.F0_error)
    else:
        f0 = (args.F0_value, args.F0_error)

    # assign F1 based on general ranges in P-Pdot diagram
    if f0[0] < 1000 and f0[0] > 100:
        f1 = (10 ** (r.randint(-16, -14)), args.F1_error)
    elif f0[0] < 100 and f0[0] > 10:
        f1 = (10 ** (r.randint(-16, -15)), args.F1_error)
    elif f0[0] < 10 and f0[0] > 0.1:
        f1 = (10 ** (r.randint(-16, -11)), args.F1_error)
    else:
        f1 = (10 ** (-16), args.F1_error)

    if type(args.F1_value) == float:
        f1 = (args.F1_value, args.F1_error)

    f1 = (-f1[0], f1[0])

    # assign DM param to be zero
    dm = (0.0, 0.0)

    # assign positional parameters
    pepoch = args.PEPOCH
    tzrmjd = args.TZRMJD
    tzrfrq = args.TZRFRQ
    tzrsite = args.TZRSITE

    # save the value of F0 for later use
    f0_save = deepcopy(f0[0])

    # write the lines to the solution parfile in TEMPO2 format
    solfile.write("PSR\t" + sol_name[:-4] + "\n")
    solfile.write("RAJ\t" + str(raj[0]) + "\t\t1\t" + str(raj[1]) + "\n")
    solfile.write("DECJ\t" + str(decj[0]) + "\t\t1\t" + str(decj[1]) + "\n")
    solfile.write("F0\t" + str(f0[0]) + "\t\t1\t" + str(f0[1]) + "\n")
    solfile.write("F1\t" + str(f1[0]) + "\t\t1\t" + str(f1[1]) + "\n")
    solfile.write("DM\t" + str(dm[0]) + "\t\t1\t" + str(dm[1]) + "\n")
    solfile.write("PEPOCH\t" + str(pepoch) + "\n")
    solfile.write("TZRMJD\t" + str(tzrmjd) + "\n")
    solfile.write("TZRFRQ\t" + str(tzrfrq) + "\n")
    solfile.write("TZRSITE\t" + tzrsite)

    solfile.close()

    return f0_save, h, m, s, d, arcm, arcs, f0, f1, dm


def write_timfile(args, f0_save, tim_name, sol_name, pulsar_number_column=True):
    """
    write a timfile with simulated TOA data

    :param args: command line input arguments
    :param f0_save: the solution value of F), saved from write_solfile
    :param tim_name: name of the output .tim file
    :param sol_name: name of the .sol file, used to make the simulated TOAs
    """

    # use zima to write the parfile into a timfile
    density = r.uniform(args.density_range[0], args.density_range[1])
    duration = int(r.uniform(args.span[0], args.span[1]))
    ntoas = int(duration / density)

    # 1 observation is a set of anywhere from 1 to 8 consecutive toas
    # 2 obs on 1 day, then obs 3 of 5 days, then 2 of next 10 days, then 1 a week later, then monthly
    # n_obs per timespan  2, 2-4, 2-4, 1-3, monthly until end
    # length between each obsevration for each timespan 0.1 - 0.9 d, 0.8 - 2.2 d, 4 - 7 d, 6 - 14 d, 20-40 d
    # 1-8 toas
    d1 = [int(r.uniform(0.1, 0.9) / density) for i in range(2)]
    d2 = [int(r.uniform(0.8, 2.2) / density) for i in range(r.randint(2, 4))]
    d3 = [int(r.uniform(4, 7) / density) for i in range(r.randint(2, 4))]
    d4 = [int(r.uniform(6, 14) / density) for i in range(r.randint(1, 3))]
    distances = d1 + d2 + d3 + d4

    # make a mask which only allows TOAs to exist on those spans specified by the distances above
    mask = np.zeros(ntoas, dtype=bool)

    i = 0
    count = 0

    for distance in distances:
        count += 1
        if count <= 2:
            # for first two observations, allow 3 to 8 TOAs per observation
            obs_length = r.randint(3, 8)
            ntoa2 = obs_length
        else:
            obs_length = r.randint(1, 8)

        mask[i : i + obs_length] = ~mask[i : i + obs_length]
        i = i + obs_length + distance

    # once distance list is used up, continue adding observations ~monthly until end of TOAs is reached
    while i < ntoas:
        obs_length = r.randint(1, 8)
        distance = int(r.uniform(20, 40) / density)
        mask[i : i + obs_length] = ~mask[i : i + obs_length]
        i = i + obs_length + distance

    # maximum possible residual based on F0
    max_resid = (0.5 / f0_save) * 10**6

    # randomly chosen error for TOAs
    percent = r.uniform(0.0003, 0.03)

    # error = percent*max resid, scale error relevant to possible residual difference
    error = int(max_resid * percent)

    # startmjd = 56000, always

    # run zima with the parameters given, this may take a long time if the number of TOAs is high (i.e. over 20000)
#     print(
#         f"running the function zima with the following parameters ./fake_data/{sol_name} ./fake_data/{tim_name} \
# --ntoa {ntoas} --duration {duration} --error {error} --addnoise True, and the rest are their default values"
#     )

    # os.system(
    #     f"zima ./fake_data/{sol_name} ./fake_data/{tim_name} --ntoa {ntoas} --duration {duration} --error {error}"
    # )
    zima(
        f"./fake_data/{sol_name}",
        f"./fake_data/{tim_name}",
        ntoa=ntoas,
        duration=duration,
        error=error,
        addnoise=True,
    )

    # turn the TOAs into a TOAs object and use the mask to remove all TOAs not in the correct ranges
    t = pint.toa.get_TOAs(Path(f"./fake_data/{tim_name}"))
    t.table = t.table[mask].group_by("obs")

    # print(t.table) # "clusters" has not been added yet, I commented out these lines, I do not think any functionality is omitted
    # # reset the TOA table group column
    # print("clusters" in t.table.columns)
    # print(t.table["clusters"][:10])

    # del t.table["clusters"]

    print("clusters" in t.table.columns)

    t.table["clusters"] = t.get_clusters()

    print(t.table["clusters"][:10])

    # save timfile
    t.write_TOA_file(Path(f"./fake_data/{tim_name}"), format="TEMPO2")

    if not pulsar_number_column:
        with open(Path(f"./fake_data/{tim_name}")) as file:
            contents = file.read().split("\n")

        for i, line in enumerate(contents):
            contents[i] = contents[i][0:50]

        with open(Path(f"./fake_data/{tim_name}"), "w") as file:
            for line in contents:
                file.write(f"{line}\n")

    return ntoa2, density


def write_parfile(args, par_name, h, m, s, d, arcm, arcs, f0, f1, dm, ntoa2, density):
    """
    write a parfile with the skewed parameter values of the simulated pulsar

    :param args: command line input arguments
    :param par_name: name of the output .par file
    """

    # write parfile as a skewed version of the solution file, the same way real data is a corrupted or blurred version of the "true" nature of the distant pulsar
    parfile = open(Path(f"./fake_data/{par_name}"), "w")

    # randomly choose a FWHM for the telescope beam in arcseconds
    FWHM = r.uniform(120, 2400)

    # randomly choose an angular separation given the FWHM above (stndard deviation = FWHM/2.35)
    separation = (FWHM / 2.35) * r.standard_normal()

    dblur = r.uniform(0.0, separation)

    rsign = [-1, 1][r.randint(2)]

    # read in argument blurring factor, or calculate from dblur
    if args.rblur != None:
        rblur = args.rblur
    else:
        rblur = (
            60.0
            * 60.0
            * (180.0 / np.pi)
            * rsign
            * np.arccos(
                np.cos((separation / 60.0 / 60.0) * np.pi / 180.0)
                / np.sin((90 - dblur / 60.0 / 60.0) * np.pi / 180)
            )
        ) / 15.0

    raj_s = (h * 60 * 60) + (m * 60) + s
    raj_s += rblur
    h = int(raj_s / (60 * 60))
    m = int((raj_s - h * 60 * 60) / 60)
    s = (raj_s - h * 60 * 60) - m * 60

    #raj = (str(h) + ":" + str(m) + ":" + str(s), 0.01)
    raj = (f"{h}:{m}:{s}", 0.01)

    if args.dblur != None:
        dblur = args.dblur

    if d > 0:
        decj_arcs = d * 60 * 60 + arcm * 60 + arcs
    else:
        decj_arcs = d * 60 * 60 - arcm * 60 - arcs

    decj_arcs += dblur
    d = int(decj_arcs / (60 * 60))
    arcm = int((decj_arcs - d * 60 * 60) / 60)
    arcs = (decj_arcs - d * 60 * 60) - arcm * 60
    if arcm < 0:
        arcm = -arcm
    if arcs < 0:
        arcs = -arcs

    #decj = (str(d) + ":" + str(arcm) + ":" + str(arcs), 0.01)
    decj = (f"{d}:{arcm}:{arcs}", 0.01)

    # the length of the observation
    Tobs = (ntoa2 * density) * 24 * 60 * 60

    if args.f0blur != None:
        f0blur = args.f0blur
    else:
        # f0blur = (~0.1)/length_obs(in s)
        f0blur = r.uniform(args.f0blur_range[0], args.f0blur_range[1]) / (
            (ntoa2 * density) * 24 * 60 * 60
        )

    # blur F0 by amount f0blur
    f0 = (f0[0] + f0blur, 0.000001)

    # blur F1 if given a blurring factor, otherwise set F1 to zero, as is the case in most starting parfiles
    if args.f1blur != None:
        f1 = (f1[0] + args.f1blur, 0.0)
    else:
        f1 = (0.0, 0.0)

    # set positional parameters
    pepoch = args.PEPOCH
    tzrmjd = args.TZRMJD
    tzrfrq = args.TZRFRQ
    tzrsite = args.TZRSITE

    # write the parfile
    parfile.write(f"PSR\t{par_name[:-4]}\n")
    parfile.write(f"RAJ\t{raj[0]}\t0\t{raj[1]}\n")
    parfile.write(f"DECJ\t{decj[0]}\t0\t{decj[1]}\n")
    parfile.write(f"F0\t{f0[0]}\t1\t{f0[1]}\n")
    parfile.write(f"F1\t{f1[0]}\t\t\t0\t{f1[1]}\n")
    parfile.write(f"DM\t{dm[0]}\t0\t{dm[1]}\n")
    parfile.write(f"PEPOCH\t{pepoch}\n")
    parfile.write(f"TZRMJD\t{tzrmjd}\n")
    parfile.write(f"TZRFRQ\t{tzrfrq}\n")
    parfile.write(f"TZRSITE\t {tzrsite}")

    parfile.close()
    # end write parfile


def zima(
    parfile,
    timfile,
    inputtim: str = None,
    startMJD: float = 56000.0,
    ntoa: int = 100,
    duration: float = 400.0,
    obs: str = "GBT",
    freq: float = 1400.0,
    error: float = 1.0,
    addnoise: bool = False,
    fuzzdays: float = 0.0,
    plot: bool = False,
    format: str = "TEMPO2",
    loglevel: str = pint.logging.script_level,
):
    log.remove()
    log.add(
        sys.stderr,
        level=loglevel,
        colorize=True,
        format=pint.logging.format,
        filter=pint.logging.LogFilter(),
    )

    log.info(f"Reading model from {parfile}")
    m = pint.models.get_model(parfile)

    out_format = format
    error = error * u.microsecond

    if inputtim is None:
        log.info("Generating uniformly spaced TOAs")
        ts = pint.simulation.make_fake_toas_uniform(
            startMJD=startMJD,
            endMJD=startMJD + duration,
            ntoas=ntoa,
            model=m,
            obs=obs,
            error=error,
            freq=np.atleast_1d(freq) * u.MHz,
            fuzz=fuzzdays * u.d,
            add_noise=addnoise,
        )
    else:
        log.info(f"Reading initial TOAs from {inputtim}")
        ts = pint.simulation.make_fake_toas_fromtim(
            inputtim,
            model=m,
            obs=obs,
            error=error,
            freq=np.atleast_1d(freq) * u.MHz,
            fuzz=fuzzdays * u.d,
            add_noise=addnoise,
        )

    # Write TOAs to a file
    ts.write_TOA_file(timfile, name="fake", format=out_format)

    if plot:
        # This should be a very boring plot with all residuals flat at 0.0!
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support

        quantity_support()

        r = pint.residuals.Residuals(ts, m)
        plt.errorbar(
            ts.get_mjds(),
            r.calc_time_resids(calctype="taylor").to(u.us),
            yerr=ts.get_errors().to(u.us),
            fmt=".",
        )
        plt.xlabel("MJD")
        plt.ylabel("Residual (us)")
        plt.grid(True)
        plt.show()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument(
        "--iter",
        help="number of pulsar systems to produce",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--name",
        help="name for the pulsar, output files will be of format <name>.par, etc.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--F0_value",
        help="value of F0 (Hz)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--F0_error",
        help="error of F0 (Hz)",
        type=float,
        default=0.0000000001,
    )
    parser.add_argument(
        "--f0blur",
        help="how much to skew the known F0 value by (Hz)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--f0blur_range",
        help="range of uniform random phases to skew F0 by (phase)",
        type=str,
        default="0.05, 0.15",
    )
    parser.add_argument(
        "--RAJ_value",
        help="value of RAJ (degrees)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--RAJ_error",
        help="error of RAJ (degrees)",
        type=float,
        default=0.0000000001,
    )
    parser.add_argument(
        "--rblur",
        help="how much to skew the known value of RAJ by (degrees)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--DECJ_value",
        help="value of DECJ (degrees)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--DECJ_error",
        help="error of DECJ (degrees)",
        type=float,
        default=0.0000000001,
    )
    parser.add_argument(
        "--dblur",
        help="how much to skew the known value of DECJ by (degrees)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--F1_value",
        help="value of F1 (1/s^2)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--F1_error",
        help="error of F1 (1/s^2)",
        type=float,
        default=0.0000000001,
    )
    parser.add_argument(
        "--f1blur",
        help="how much to skew the known value of F1 by ()",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--PEPOCH",
        help="period epoch for pulsar (MJD)",
        type=float,
        default=56000,
    )
    parser.add_argument(
        "--TZRFRQ",
        help="Frequency (Hz) of observation",
        type=float,
        default=1400,
    )
    parser.add_argument(
        "--TZRMJD",
        help="Observation start time (MJD)",
        type=float,
        default=56000,
    )
    parser.add_argument(
        "--TZRSITE",
        help="observation site code",
        type=str,
        default="GBT",
    )
    parser.add_argument(
        "--density_range",
        help="range of toa densities to choose from (days)",
        type=str,
        default="0.004, 0.02",
    )
    parser.add_argument(
        "--span",
        help="range of time spans to choose from (days)",
        type=str,
        default="200,700",
    )
    # parse comma-seperated pairs
    args = parser.parse_args(argv)
    args.span = [float(i) for i in args.span.split(",")]
    args.f0blur_range = [float(i) for i in args.f0blur_range.split(",")]
    args.density_range = [float(i) for i in args.density_range.split(",")]

    # write 3 files
    # fake toas - unevenly distributed, randomized errors
    # fake perfect parfile - solution to the fake system
    # fake starting parfile - starting parfile smeared/dispersed by a random amount
    # save the files in fake_data folder

    # check that there is a directory to save the fake data in
    # want fake data in data1 folder
    fake_data = Path("./fake_data")
    original_path = os.getcwd()
    if socket.gethostname() == "nimrod":
        while not os.path.exists(Path("data1/people")):
            os.chdir("..")
        os.chdir(Path("data1/people/jdtaylor"))
        if not fake_data.is_dir() and os.getcwd().split("/")[-1] == "jdtaylor":
            fake_data.mkdir()

    # The path above is not in the fitzroy station
    # Fake data on the fitzroy host should be deleted as soon as reasonable
    elif socket.gethostname() == "fitzroy":
        os.chdir(Path("/users/jdtaylor/Jackson/"))
        if not fake_data.is_dir():
            fake_data.mkdir()
    else:
        if not fake_data.is_dir():
            fake_data.mkdir()

    # determine highest number system from files in fake_data
    try:
        temp_list = []
        for filename in os.listdir(Path("./fake_data/")):
            if (
                "fake" in filename
                and (".tim" in filename[-4:] or ".par" in filename[-4:])
                and "#" not in filename
            ):
                temp_list.append(int(filename[:-4][5:]))

        maxnum = max(temp_list)
    except ValueError:
        maxnum = 0
        print("no files in the directory")
        print(os.getcwd())
        print(os.listdir("fake_data"))

    iter = args.iter

    for num in range(maxnum + 1, maxnum + 1 + iter):

        if args.name == None:
            sol_name = f"fake_{num}.sol"
            par_name = f"fake_{num}.par"
            tim_name = f"fake_{num}.tim"
        else:
            sol_name = f"{args.name}.sol"
            par_name = f"{args.name}.par"
            tim_name = f"{args.name}.tim"

        # write solfile
        f0_save, h, m, s, d, arcm, arcs, f0, f1, dm = write_solfile(args, sol_name)

        # wrie timfile
        ntoa2, density = write_timfile(
            args, f0_save, tim_name, sol_name, pulsar_number_column=False
        )

        # write parfile
        write_parfile(
            args, par_name, h, m, s, d, arcm, arcs, f0, f1, dm, ntoa2, density
        )
    os.chdir(original_path)


if __name__ == "__main__":
    import time

    time1 = time.monotonic()
    main()
    time2 = time.monotonic()
    print(f"Took {time2 - time1} seconds ({(time2-time1) / 60} minutes)")
