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
import APTB
import treelib


"""
The intention of this file is to store functions to make adding additional binary models to APTB
seamless and standardized. For example, APTB will, instead of checking for the type of binary model 
and its features (like not fitting for EPS1 and EPS2 immediately), APTB will call a function from this
file that will do the equivalent process. This also serves to prevent clutter in APTB.
"""


def set_binary_pars_lim(m, args):
    if args.binary_model.lower() == "ell1" and args.EPS_lim is None:
        if args.EPS_lim == "inf":
            args.EPS_lim = np.inf
        else:
            args.EPS_lim = m.PB.value * 5
            args.EPS_lim = m.PB.value * 5

    elif args.binary_model.lower() == "bt":
        if args.ECC_lim:
            if args.ECC_lim == "inf":
                args.ECC_lim = np.inf
        else:
            args.ECC_lim = 0

        if args.OM_lim:
            if args.OM_lim == "inf":
                args.OM_lim = np.inf
        else:
            args.OM_lim = 0

    return args


def do_Ftests_binary(m, t, f, f_params, span, Ftests, args):
    """
    Helper function for APTB.
    """

    if args.binary_model.lower() == "ell1":
        # want to add eps1 and eps2 at the same time
        if "EPS1" not in f_params and span > args.EPS_lim * u.d:
            Ftest_F, m_plus_p = APTB.Ftest_param(m, f, "EPS1&2", args)
            Ftests[Ftest_F] = "EPS1&2"
        # if "EPS2" not in f_params and span > args.EPS2_lim * u.d:
        #     Ftest_F = APTB.Ftest_param(m, f, "EPS2", args)
        #     Ftests[Ftest_F] = "EPS2"

    elif args.binary_model.lower() == "bt":
        for param in ["ECC", "OM"]:
            if (
                param not in f_params and span > getattr(args, f"{param}_lim") * u.d
            ):  # args.F0_lim * u.d:
                Ftest_R, m_plus_p = APTB.Ftest_param(m, f, param, args)
                Ftests[Ftest_R] = param

    return m, t, f, f_params, span, Ftests, args


def skeleton_tree_creator(blueprint, iteration_dict=None):
    """
    This creates what the tree looks like, without any of the data attributes.

    Parameters
    ----------
    blueprint : blueprint in the form [(parent, child) for node in tree]

    Returns
    -------
    tree : treelib.Tree
    """
    tree = treelib.Tree()
    tree.create_node("Root", "Root")
    U_counter = 0
    if iteration_dict:
        for parent, child in blueprint:
            # while tree.contains(child):
            #     child += child[-1]
            if parent != "Root":
                i_index = parent.index("i")
                d_index = parent.index("d")
                parent = f"i{iteration_dict.get(parent, f'U{(U_counter:=U_counter+1)}')}_{parent[d_index:i_index-1]}"
            i_index = child.index("i")
            d_index = child.index("d")
            child = f"i{iteration_dict.get(child, f'U{(U_counter:=U_counter+1)}')}_{child[d_index:i_index-1]}"
            tree.create_node(child, child, parent=parent)
    else:
        for parent, child in blueprint:
            # while tree.contains(child):
            #     child += child[-1]
            tree.create_node(child, child, parent=parent)
    return tree
