import sys
import argparse
parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
parser.add_argument(
    "--iter",
    help="number of pulsar systems to produce",
    type=str,
    default=1,
)

args = parser.parse_args()

print(args.iter)