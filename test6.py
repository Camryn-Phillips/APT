from simdata import zima
import pint
import numpy as np
import numpy.random as r
from pathlib import Path

# zima("fake_29sol.par", "fake_29.tim")
density = 0.008
duration = 400
ntoas = int(duration / density)

zima(
    "fake_35sol.par",
    "fake_35.tim",
    ntoa=ntoas,
    duration=duration,
    error=1,
    addnoise=False,
)


def write_timfile(f0_save, tim_name, sol_name, pulsar_number_column=True):
    """
    write a timfile with simulated TOA data

    :param args: command line input arguments
    :param f0_save: the solution value of F), saved from write_solfile
    :param tim_name: name of the output .tim file
    :param sol_name: name of the .sol file, used to make the simulated TOAs
    """

    # use zima to write the parfile into a timfile

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

    # turn the TOAs into a TOAs object and use the mask to remove all TOAs not in the correct ranges
    t = pint.toa.get_TOAs(Path(f"{tim_name}"))
    t.table = t.table[mask].group_by("obs")

    t.table["clusters"] = t.get_clusters()

    print(t.table["clusters"][:10])

    # save timfile
    t.write_TOA_file(Path(f"{tim_name}"), format="TEMPO2")

    if not pulsar_number_column:  # do not include the pulse number in the tim file
        with open(Path(f"{tim_name}")) as file:
            contents = file.read().split("\n")

        for i, line in enumerate(contents):
            contents[i] = contents[i][0:50]

        with open(Path(f"{tim_name}"), "w") as file:
            for line in contents:
                file.write(f"{line}\n")

    return ntoa2, density


write_timfile(4.242640765910615, "fake_29.tim", "g")
