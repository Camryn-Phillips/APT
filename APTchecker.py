import pint.models as pm
import numpy as np
import sys
import os
from pathlib import Path

original_path = Path.cwd()
def model_checker(number: int = None, path: Path = Path("/data1/people/jdtaylor")):
    """
    Checks if two parameter files give indetical timing solutions
    # FIXME This can be easily implented in the general case if desired.
    """
    os.chdir(path)

    numbers = [number]

    if not number:
        number = int(sys.argv[1])
        if number == -1:
            numbers = np.arange(1, 19)

    identical_bool_array = np.zeros(len(numbers))

    for number in numbers:
        m_cor, t_cor = pm.get_model_and_toas(f"./fake_data/fake_{number}.sol", f"./fake_data/fake_{number}.tim")

        t_cor.compute_pulse_numbers(m_cor)
        pn_cor = t_cor.table["pulse_number"]

        m_fin, t_fin = pm.get_model_and_toas(f"fake_{number}.fin", f"./fake_data/fake_{number}.tim")

        t_fin.compute_pulse_numbers(m_fin)
        pn_fin = t_fin.table["pulse_number"]

        value = np.array_equal(pn_cor, pn_fin)
        
        if not value:
            print("\n" * 6)
        print(f"number {number} is {value}")
        if not value:
            print("\n" * 6)

        identical_bool_array[number-1] = value

    print(f"{identical_bool_array}, length = {len(identical_bool_array)}")
    return identical_bool_array

def main():
    model_checker()

if __name__ == "__main__":
    main()