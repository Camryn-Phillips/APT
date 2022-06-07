import os

sol_name = 1
tim_name =23
ntoas = 4
duration = 3
error =4

print(f"zima ./fake_data/{sol_name} ./fake_data/{tim_name} --ntoa {ntoas} --duration {duration} --error {error}")

from pint.testJT import test
test()

with open("/data1/people/jdtaylor/test3.txt", "w") as file:
    file.write("hello")

print("done")