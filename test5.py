from pathlib import Path
import os, sys

for num in range(46, 52):

    tim_name = f"/data1/people/jdtaylor/fake_data/fake_{num}.tim"

    with open(Path(f"{tim_name}")) as file:
        contents = file.read().split("\n")

    for i, line in enumerate(contents):
        try:
            contents[i] = contents[i][: line.index("-pn")]
        except ValueError:
            pass

    with open(Path(f"{tim_name}"), "w") as file:
        for line in contents:
            file.write(f"{line}\n")
