import pint.models
import numpy as np
import sys
import os
from pathlib import Path

original_path = Path.cwd()
def solution_compare(parfile1: Path, parfile2: Path, timfile: Path) -> bool:
    """
    Compares two solutions to see if identical solution, where identical means they identify
    the same pulse number for each TOA.
    """
    m1, t1 = pint.models.get_model_and_toas(parfile1, timfile)
    t1.compute_pulse_numbers(m1)
    pn1 = t1.table["pulse_number"]

    m2, t2 = pint.models.get_model_and_toas(parfile2, timfile)
    t2.compute_pulse_numbers(m2)
    pn2 = t2.table["pulse_number"]

    return np.array_equal(pn1, pn2)

print(solution_compare("/data1/people/jdtaylor/fake_1.fin", 
"/data1/people/jdtaylor/fake_data/fake_1.sol", 
"/data1/people/jdtaylor/fake_data/fake_1.tim"))