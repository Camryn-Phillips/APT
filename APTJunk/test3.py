import os
import argparse
import sys

sol_name = 1
tim_name =23
ntoas = 4
duration = 3
error =4

# print(f"zima ./fake_data/{sol_name} ./fake_data/{tim_name} --ntoa {ntoas} --duration {duration} --error {error}")

# from pint.testJT import test
# test()

# with open("/data1/people/jdtaylor/test3.txt", "w") as file:
#     file.write("hello")

# print("done")

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

parser = argparse.ArgumentParser(description="Fibonacci")

parser.add_argument("n", help="enter the nth fib number", type = int)
parser.add_argument("--t", action=argparse.BooleanOptionalAction, help="enter the nth fib number", type = bool, default = True)

args = parser.parse_args()

def fib(n):
    n = int(n)
    if n == 0 or n == 1:
        return n
    else:
        return fib(n-2) + fib(n-1)

print(f"{args.n}: {fib(args.n)}")

print(f"{args.t}")