from pint.models.model_builder import parse_parfile
import pint
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.chdir("/data1/people/jdtaylor/fake_data")

sol_list = []
F1_list = []
F0_list = []
for file in os.listdir():
    if file[-4:] == ".sol":
        sol_list.append(file)

for file in sol_list:
    F1_info = parse_parfile(file)["F1"][0]
    F0_info = parse_parfile(file)["F0"][0]

    stop_index = F1_info.index(" 1 ")
    F1 = float(F1_info[:stop_index])
    F1_list.append(F1)

    stop_index = F0_info.index(" 1 ")
    F0 = float(F0_info[:stop_index])
    F0_list.append(F0)

F0 = np.array(F0_list)
F1 = np.array(F1_list)

P0 = 1 / F0
P1 = - 1 / (F0**2) * F1

fig, ax = plt.subplots(figsize = (15, 10))

p1, = ax.plot(P0, P1, 'bo')

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel("P (s)")
ax.set_ylabel("dP/dt")
#ax.set_ylabel("P\u0307")
#ax.yaxis.set_label_coords(-0, .5)

#print("P\u0307")

plt.show()

# fig, ax = plt.subplots(figsize = (15, 10))

# p1, = ax.plot(F0, F1, 'bo')

# plt.show()


