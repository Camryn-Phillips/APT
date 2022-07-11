import pint.models as pm
import pint.logging
import numpy as np
import sys
import os
from pathlib import Path
from pint.models.model_builder import parse_parfile
from copy import deepcopy

pint.logging.setup(level='WARNING')
os.chdir("/data1/people/jdtaylor/binary_fake_data6")
alg_saves = Path("alg_saves")
successes = np.zeros(100, dtype = bool)
for i in range(1,101):
    m_cor, t_cor = pm.get_model_and_toas(f"fake_{i}sol.par", f"fake_{i}.tim")
    t_cor_copy = deepcopy(t_cor)
    t_cor_copy.compute_pulse_numbers(m_cor)
    pn_cor = t_cor_copy.table["pulse_number"]

    top_path = Path(f"fake_{i}")
    os.chdir(alg_saves / top_path)
    print(f"cwd = {os.getcwd()}")
    print(f"listdir = {os.listdir()}")
    for dir in os.listdir():
        dir = Path(dir)
        os.chdir(dir)
        if ".fin" not in os.listdir() or "old" in os.listdir():
            continue
        m_fin = pm.get_model(f"fake_{i}.fin")
        t_fin = deepcopy(t_cor)
        t_fin.compute_pulse_numbers(m_fin)
        pn_fin = t_fin.table["pulse_number"]

        identical = np.array_equal(pn_fin, pn_cor)

        if identical:
            successes[i] = 1
            break

        
        os.chdir("..")
    os.chdir("..")

print(successes)
    