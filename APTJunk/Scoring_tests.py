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

data_path = "/Users/jackson/Desktop/Pint_Personal/APT/binary_test_data/binary6_1"
suffix = data_path.split("_")[-1]
if socket.gethostname()[0] == "J":
    os.chdir(data_path)
timfile = f"fake_{suffix}.tim"
parfile = f"fake_{suffix}.par"

m, toas = mb.get_model_and_toas(parfile, timfile)


def scoring_original(dts):
    return (1.0 / dts).sum(axis=1)


def scoring_0_3(dts):
    return (1.0 / dts**0.3).sum(axis=1)


def scoring_3(dts):
    return (1.0 / dts**3).sum(axis=1)


def gaussian(dts):
    return (np.exp(-((dts) ** 2))).sum(axis=1)


top_mask_lists = []
top_cluster_lists = []

for scroring_function in [scoring_original, scoring_0_3, scoring_3, gaussian]:
    t = deepcopy(toas)
    if "clusters" not in t.table.columns:
        t.table["clusters"] = t.get_clusters()
    mjd_values = t.get_mjds().value
    dts = np.fabs(mjd_values - mjd_values[:, np.newaxis]) + np.eye(len(mjd_values))

    score_list = scroring_function(dts)

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
    top_mask_lists.append(mask_list[:5])
    top_cluster_lists.append([scroring_function.__name__, starting_cluster_list[:5]])


print(top_cluster_lists)
