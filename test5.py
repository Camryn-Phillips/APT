from pathlib import Path
import os, sys

tim_name = sys.argv[1]

with open(Path(f"{tim_name}")) as file:
    contents = file.read().split("\n")

for i, line in enumerate(contents):
    contents[i] = contents[i][0:50]

with open(Path(f"{tim_name}"), "w") as file:
    for line in contents:
        file.write(f"{line}\n")
