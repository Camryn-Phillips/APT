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
import APT_binary


"""
The intention of this file is to store functions to make adding additional binary models to APT_binary
seamless and standardized. For example, APTB will, instead of checking for the type of binary model 
and its features (like not fitting for EPS1 and EPS2 immediately), APTB will call a function from this
file that will do the equivalent process. This also serves to prevent clutter in APTB.
"""


def do_Ftests_binary(m, t, f, f_params, span, Ftests, args):
    """
    Helper function for APT_binary.
    """

    if args.binary_model.lower() == "ell1":
        if "EPS1" not in f_params and span > args.EPS1_lim * u.d:
            Ftest_F = APT_binary.Ftest_param(m, f, "EPS1")
            Ftests[Ftest_F] = "EPS1"
        if "EPS2" not in f_params and span > args.EPS2_lim * u.d:
            Ftest_F = APT_binary.Ftest_param(m, f, "EPS2")
            Ftests[Ftest_F] = "EPS2"

    elif args.binary_model.lower() == "other models":
        pass

    return m, t, f, f_params, span, Ftests, args
