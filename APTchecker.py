import pint.models as pm
import numpy as np
import sys
import os
from pathlib import Path

"""
This script is intended to determine how many soltuions APT succesfully solved.
"""

original_path = Path.cwd()


def model_checker(number: int = None, path: Path = Path("/data1/people/jdtaylor")):
    """
    Checks if two parameter files give indetical timing solutions
    # FIXME This can be easily implented in the general case if desired.
    """
    os.chdir(path)
    # print(sys.argv[1])
    whole_list = False

    if number is None:
        number = int(sys.argv[1])
        numbers = [number]
        if number == -1:
            whole_list = True
            numbers = np.arange(1, 102)
    else:
        numbers = [number]
    identical_bool_array = [0 for i in range(len(numbers))]

    for number in numbers:
        try:
            m_cor, t_cor = pm.get_model_and_toas(
                f"./fake_data/fake_{number}.sol", f"./fake_data/fake_{number}.tim"
            )

            t_cor.compute_pulse_numbers(m_cor)
            pn_cor = t_cor.table["pulse_number"]

            m_fin, t_fin = pm.get_model_and_toas(
                f"fake_{number}.fin", f"./fake_data/fake_{number}.tim"
            )

            t_fin.compute_pulse_numbers(m_fin)
            pn_fin = t_fin.table["pulse_number"]

            value = np.array_equal(pn_cor, pn_fin)

            if not value:
                print("\n" * 6)
            print(f"number {number} is {value}")
            if not value:
                print("\n" * 6)

            if whole_list:
                identical_bool_array[number - 1] = [value, number]
            else:
                identical_bool_array[0] = value
        except Exception as error:
            print("\n" * 6)
            print(f"{number} had an error. It is being skipped and marked False.")
            print(error)
            print("\n" * 6)
            if whole_list:
                identical_bool_array[number - 1] = [False, number]
            else:
                identical_bool_array[0] = False

            continue

    print(f"{identical_bool_array}, length = {len(identical_bool_array)}")
    return identical_bool_array


def main():
    model_checker()


if __name__ == "__main__":
    main()
