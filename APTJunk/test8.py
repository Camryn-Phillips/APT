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
#import APT_argparse
import socket
import sys

from APT_binary import JUMP_adder_begginning_cluster, starting_points

os.chdir(Path("/users/jdtaylor/Jackson/APT/binary_test_data/binary4_100"))

output_parfile = sys.argv[1]

with open(output_parfile) as parfile:
    contents = parfile.read().split("\n")
with open(output_parfile, "w") as parfile:
    for line in contents:
        if "JUMP" not in line:
            parfile.write(line + "\n")

timfile = "fake_100.jump.tim"
model, t = pint.models.get_model_and_toas(output_parfile, timfile)
flag_name = "jump_tim"
masks, clusters = starting_points(t)
mask = masks[0]
t.table["clusters"] = t.get_clusters()

with open(output_parfile, "w") as parfile:
    parfile.write(model.as_parfile())
    for i in set(t.table[~mask]["clusters"]):
        parfile.write(f"JUMP\t\t-{flag_name} {i}\t0 1 0\n")