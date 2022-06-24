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

from APT_binary import JUMP_adder_begginning_cluster, starting_points

#os.chdir("/data1/people/jdtaylor/fake_data")
os.chdir(Path("/users/jdtaylor/Jackson/APT/binary_test_data/binary4_100"))

parfile = "fake_100.par"
timfile = "fake_100.tim"

m, t = pint.models.get_model_and_toas(parfile, timfile)

mask_list, cluster_list = starting_points(toas=t)
mask = mask_list[0]

m, t = JUMP_adder_begginning_cluster(mask, t, m, "fake_100.jump.par", "fake_100.jump.tim")