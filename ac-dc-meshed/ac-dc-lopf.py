

# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import


import os
import pypsa

import pandas as pd

import numpy as np

from itertools import chain

network = pypsa.Network()

folder_name = "ac-dc-data"
network.import_from_csv_folder(folder_name)

now = network.snapshots[4]
network.now = now
parameters = {'discount_rate':0.00,'lifetime':1}
network.lopf(network.snapshots,milp=True,parameters = parameters)

for sn in network.sub_networks.obj:
    print(sn,network.sub_networks.at[sn.name,"carrier"],len(sn.buses()),len(sn.branches()))

print("\nControllable branches:")

print(network.links)

now = network.snapshots[5]


print("\nCheck power balance at each branch:")

branches = network.branches()

for bus in network.buses.index:
    print("\n"*3+bus)
    generators = sum(network.generators_t.p.loc[now,network.generators.bus==bus])
    loads = sum(network.loads_t.p.loc[now,network.loads.bus==bus])
    print("Generators:",generators)
    print("Loads:",loads)
    print("Total:",generators-loads)

    p0 = 0.
    p1 = 0.

    for c in network.iterate_components(pypsa.components.branch_components):

        bs = (c.df.bus0 == bus)

        if bs.any():
            print(c,"\n",c.pnl.p0.loc[now,bs])
            p0 += c.pnl.p0.loc[now,bs].sum()

        bs = (c.df.bus1 == bus)

        if bs.any():
            print(c,"\n",c.pnl.p1.loc[now,bs])
            p1 += c.pnl.p1.loc[now,bs].sum()

    print("Branches",p0+p1)

    np.testing.assert_allclose(generators-loads+1.,p0+p1+1.)

    print("")

print(sum(network.generators_t.p.loc[now]))

print(sum(network.loads_t.p.loc[now]))

results_folder_name = os.path.join(folder_name,"results-lopf")

if True:
    network.export_to_csv_folder(results_folder_name)
