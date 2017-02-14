## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David
## Schlachtberger (FIAS)
## Copyright 2017 João Gorenstein Dedecca

## The following code is a modification of the optimal power flow script of the PyPSA package (Python for Power System Analysis), pypsa.org

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


# make the code as Python 3 compatible as possible
from __future__ import division, absolute_import
from six import iteritems, string_types


__author__ = """
Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS)
Joao Gorenstein Dedecca
"""

__copyright__ = """
Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS), GNU GPL 3
Copyright 2017 João Gorenstein Dedecca, GNU GPL 3
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve
from pyomo.environ import (ConcreteModel, Var, Objective,
                           NonNegativeReals, Constraint, Reals,
                           Suffix, Expression, Boolean, Param)
from pyomo.opt import SolverFactory
from itertools import chain
from . import components

import logging
logger = logging.getLogger(__name__)

import datetime
import networkx as nx


from distutils.version import StrictVersion, LooseVersion
try:
    _pd_version = StrictVersion(pd.__version__)
except ValueError:
    _pd_version = LooseVersion(pd.__version__)

from .pf import (calculate_dependent_values, find_slack_bus,
                 find_bus_controls, calculate_B_H, calculate_PTDF, find_tree,
                 find_cycles)
from .opt import (l_constraint, l_objective, LExpression, LConstraint,
                  patch_optsolver_free_model_before_solving,
                  patch_optsolver_record_memusage_before_solving,
                  empty_network)
from .descriptors import get_switchable_as_dense, allocate_series_dataframes



def network_opf(network,snapshots=None):
    """Optimal power flow for snapshots."""

    raise NotImplementedError("Non-linear optimal power flow not supported yet")



def define_generator_variables_constraints(network,snapshots):

    extendable_gens_i = network.generators.index[network.generators.p_nom_extendable]
    fixed_gens_i = network.generators.index[~ network.generators.p_nom_extendable]

    p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu')
    p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu')

    ## Define generator dispatch variables ##

    gen_p_bounds = {(gen,sn) : (None,None)
                    for gen in extendable_gens_i
                    for sn in snapshots}

    if len(fixed_gens_i):
        var_lower = p_min_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])
        var_upper = p_max_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])

        gen_p_bounds.update({(gen,sn) : (var_lower[gen][sn],var_upper[gen][sn])
                             for gen in fixed_gens_i
                             for sn in snapshots})

    def gen_p_bounds_f(model,gen_name,snapshot):
        return gen_p_bounds[gen_name,snapshot]

    network.model.generator_p = Var(list(network.generators.index), snapshots,
                                    domain=Reals, bounds=gen_p_bounds_f)

    ## Define generator capacity variables if generator is extendable ##

    def gen_p_nom_bounds(model, gen_name):
        return (network.generators.at[gen_name,"p_nom_min"],
                network.generators.at[gen_name,"p_nom_max"])

    network.model.generator_p_nom = Var(list(extendable_gens_i),
                                        domain=NonNegativeReals, bounds=gen_p_nom_bounds)


    ## Define generator dispatch constraints for extendable generators ##

    gen_p_lower = {(gen,sn) :
                   [[(1,network.model.generator_p[gen,sn]),
                     (-p_min_pu.at[sn, gen],
                      network.model.generator_p_nom[gen])],">=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_lower", gen_p_lower,
                 list(extendable_gens_i), snapshots)

    gen_p_upper = {(gen,sn) :
                   [[(1,network.model.generator_p[gen,sn]),
                     (-p_max_pu.at[sn, gen],
                      network.model.generator_p_nom[gen])],"<=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_upper", gen_p_upper,
                 list(extendable_gens_i), snapshots)



def define_storage_variables_constraints(network,snapshots):

    sus = network.storage_units
    ext_sus_i = sus.index[sus.p_nom_extendable]
    fix_sus_i = sus.index[~ sus.p_nom_extendable]

    model = network.model

    ## Define storage dispatch variables ##

    p_max_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_max_pu')
    p_min_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_min_pu')

    bounds = {(su,sn) : (0,None) for su in ext_sus_i for sn in snapshots}
    bounds.update({(su,sn) :
                   (0,sus.at[su,"p_nom"]*p_max_pu.at[sn, su])
                   for su in fix_sus_i for sn in snapshots})

    def su_p_dispatch_bounds(model,su_name,snapshot):
        return bounds[su_name,snapshot]

    network.model.storage_p_dispatch = Var(list(network.storage_units.index), snapshots,
                                           domain=NonNegativeReals, bounds=su_p_dispatch_bounds)



    bounds = {(su,sn) : (0,None) for su in ext_sus_i for sn in snapshots}
    bounds.update({(su,sn) :
                   (0,-sus.at[su,"p_nom"]*p_min_pu.at[sn, su])
                   for su in fix_sus_i
                   for sn in snapshots})

    def su_p_store_bounds(model,su_name,snapshot):
        return bounds[su_name,snapshot]

    network.model.storage_p_store = Var(list(network.storage_units.index), snapshots,
                                        domain=NonNegativeReals, bounds=su_p_store_bounds)

    ## Define spillage variables only for hours with inflow>0. ##
    inflow = get_switchable_as_dense(network, 'StorageUnit', 'inflow', snapshots)
    spill_sus_i = sus.index[inflow.max()>0] #skip storage units without any inflow
    inflow_gt0_b = inflow>0
    spill_bounds = {(su,sn) : (0,inflow.at[sn,su])
                    for su in spill_sus_i
                    for sn in snapshots
                    if inflow_gt0_b.at[sn,su]}
    spill_index = spill_bounds.keys()

    def su_p_spill_bounds(model,su_name,snapshot):
        return spill_bounds[su_name,snapshot]

    network.model.storage_p_spill = Var(list(spill_index),
                                        domain=NonNegativeReals, bounds=su_p_spill_bounds)



    ## Define generator capacity variables if generator is extendable ##

    def su_p_nom_bounds(model, su_name):
        return (sus.at[su_name,"p_nom_min"],
                sus.at[su_name,"p_nom_max"])

    network.model.storage_p_nom = Var(list(ext_sus_i), domain=NonNegativeReals,
                                      bounds=su_p_nom_bounds)



    ## Define generator dispatch constraints for extendable generators ##

    def su_p_upper(model,su_name,snapshot):
        return (model.storage_p_dispatch[su_name,snapshot] <=
                model.storage_p_nom[su_name]*p_max_pu.at[snapshot, su_name])

    network.model.storage_p_upper = Constraint(list(ext_sus_i),snapshots,rule=su_p_upper)


    def su_p_lower(model,su_name,snapshot):
        return (model.storage_p_store[su_name,snapshot] <=
                -model.storage_p_nom[su_name]*p_min_pu.at[snapshot, su_name])

    network.model.storage_p_lower = Constraint(list(ext_sus_i),snapshots,rule=su_p_lower)



    ## Now define state of charge constraints ##

    network.model.state_of_charge = Var(list(network.storage_units.index), snapshots,
                                        domain=NonNegativeReals, bounds=(0,None))

    upper = {(su,sn) : [[(1,model.state_of_charge[su,sn]),
                         (-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
             for su in ext_sus_i for sn in snapshots}
    upper.update({(su,sn) : [[(1,model.state_of_charge[su,sn])],"<=",
                             sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
                  for su in fix_sus_i for sn in snapshots})

    l_constraint(model, "state_of_charge_upper", upper,
                 list(network.storage_units.index), snapshots)


    #this builds the constraint previous_soc + p_store - p_dispatch + inflow - spill == soc
    #it is complicated by the fact that sometimes previous_soc and soc are floats, not variables
    soc = {}

    #store the combinations with a fixed soc
    fixed_soc = {}

    state_of_charge_set = get_switchable_as_dense(network, 'StorageUnit', 'state_of_charge_set', snapshots)

    for su in sus.index:
        for i,sn in enumerate(snapshots):

            soc[su,sn] =  [[],"==",0.]

            elapsed_hours = network.snapshot_weightings[sn]

            if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
                previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
                soc[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours
                                  * previous_state_of_charge)
            else:
                previous_state_of_charge = model.state_of_charge[su,snapshots[i-1]]
                soc[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                      previous_state_of_charge))


            state_of_charge = state_of_charge_set.at[sn,su]
            if pd.isnull(state_of_charge):
                state_of_charge = model.state_of_charge[su,sn]
                soc[su,sn][0].append((-1,state_of_charge))
            else:
                soc[su,sn][2] += state_of_charge
                #make sure the variable is also set to the fixed state of charge
                fixed_soc[su,sn] = [[(1,model.state_of_charge[su,sn])],"==",state_of_charge]

            soc[su,sn][0].append((sus.at[su,"efficiency_store"]
                                  * elapsed_hours,model.storage_p_store[su,sn]))
            soc[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                                  model.storage_p_dispatch[su,sn]))
            soc[su,sn][2] -= inflow.at[sn,su] * elapsed_hours

    for su,sn in spill_index:
        storage_p_spill = model.storage_p_spill[su,sn]
        soc[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))

    l_constraint(model,"state_of_charge_constraint",
                 soc,list(network.storage_units.index), snapshots)

    l_constraint(model, "state_of_charge_constraint_fixed",
                 fixed_soc, list(fixed_soc.keys()))



def define_store_variables_constraints(network,snapshots):

    stores = network.stores
    ext_stores = stores.index[stores.e_nom_extendable]
    fix_stores = stores.index[~ stores.e_nom_extendable]

    e_max_pu = get_switchable_as_dense(network, 'Store', 'e_max_pu')
    e_min_pu = get_switchable_as_dense(network, 'Store', 'e_min_pu')



    model = network.model

    ## Define store dispatch variables ##

    network.model.store_p = Var(list(stores.index), snapshots, domain=Reals)


    ## Define store energy variables ##

    bounds = {(store,sn) : (None,None) for store in ext_stores for sn in snapshots}

    bounds.update({(store,sn) :
                   (stores.at[store,"e_nom"]*e_min_pu.at[sn,store],stores.at[store,"e_nom"]*e_max_pu.at[sn,store])
                   for store in fix_stores for sn in snapshots})

    def store_e_bounds(model,store,snapshot):
        return bounds[store,snapshot]


    network.model.store_e = Var(list(stores.index), snapshots, domain=Reals,
                                bounds=store_e_bounds)


    ## Define energy capacity variables if store is extendable ##

    def store_e_nom_bounds(model, store):
        return (stores.at[store,"e_nom_min"],
                stores.at[store,"e_nom_max"])

    network.model.store_e_nom = Var(list(ext_stores), domain=Reals,
                                    bounds=store_e_nom_bounds)


    ## Define energy capacity constraints for extendable generators ##

    def store_e_upper(model,store,snapshot):
        return (model.store_e[store,snapshot] <=
                model.store_e_nom[store]*e_max_pu.at[snapshot,store])

    network.model.store_e_upper = Constraint(list(ext_stores), snapshots, rule=store_e_upper)

    def store_e_lower(model,store,snapshot):
        return (model.store_e[store,snapshot] >=
                model.store_e_nom[store]*e_min_pu.at[snapshot,store])

    network.model.store_e_lower = Constraint(list(ext_stores), snapshots, rule=store_e_lower)

    ## Builds the constraint previous_e - p == e ##

    e = {}

    for store in stores.index:
        for i,sn in enumerate(snapshots):

            e[store,sn] =  LConstraint(sense="==")

            e[store,sn].lhs.variables.append((-1,model.store_e[store,sn]))

            elapsed_hours = network.snapshot_weightings[sn]

            if i == 0 and not stores.at[store,"e_cyclic"]:
                previous_e = stores.at[store,"e_initial"]
                e[store,sn].lhs.constant += ((1-stores.at[store,"standing_loss"])**elapsed_hours
                                         * previous_e)
            else:
                previous_e = model.store_e[store,snapshots[i-1]]
                e[store,sn].lhs.variables.append(((1-stores.at[store,"standing_loss"])**elapsed_hours,
                                              previous_e))

            e[store,sn].lhs.variables.append((-elapsed_hours, model.store_p[store,sn]))

    l_constraint(model,"store_constraint", e, list(stores.index), snapshots)



def define_branch_extension_variables(network,snapshots):

    passive_branches = network.passive_branches()

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    bounds = {b : (extendable_passive_branches.at[b,"s_nom_min"],
                   extendable_passive_branches.at[b,"s_nom_max"])
              for b in extendable_passive_branches.index}

    def branch_s_nom_bounds(model, branch_type, branch_name):
        return bounds[branch_type,branch_name]

    network.model.passive_branch_s_nom = Var(list(extendable_passive_branches.index),
                                             domain=NonNegativeReals, bounds=branch_s_nom_bounds)

    extendable_links = network.links[network.links.p_nom_extendable]

    bounds = {b : (extendable_links.at[b,"p_nom_min"],
                   extendable_links.at[b,"p_nom_max"])
              for b in extendable_links.index}

    def branch_p_nom_bounds(model, branch_name):
        return bounds[branch_name]

    network.model.link_p_nom = Var(list(extendable_links.index),
                                   domain=NonNegativeReals, bounds=branch_p_nom_bounds)

def define_MILP_branch_extension_variables(network,snapshots):
    # TODO Add lines and link s_nom_min constraint
    passive_branches = network.passive_branches()

    extendable_passive_branches_i = passive_branches.index[passive_branches.s_nom_extendable]

    bounds = {b: (0, 1) for b in extendable_passive_branches_i}

    def pb_inv_ratio_bounds(model, branch_type, branch_name):
        return bounds[branch_type, branch_name]

    network.model.pb_inv_ratio = Var(list(extendable_passive_branches_i),domain=NonNegativeReals, bounds=pb_inv_ratio_bounds)
    network.model.pb_bin_inv = Var(list(extendable_passive_branches_i), domain=Boolean)

    extendable_ptp_links_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]

    bounds = {b: (0, 1) for b in extendable_ptp_links_i}

    def cb_inv_ratio_bounds(model, branch_name):
        return bounds[branch_name]

    network.model.cb_inv_ratio = Var(list(extendable_ptp_links_i),domain=NonNegativeReals, bounds=cb_inv_ratio_bounds)
    network.model.cb_bin_inv = Var(list(extendable_ptp_links_i), domain=Boolean)

    extendable_converters_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "converter")]

    bounds = {b : (network.links.at[b,"p_nom_min"],
                   network.links.at[b,"p_nom_max"])
              for b in extendable_converters_i}

    def conv_p_nom_bounds(model, branch_name):
        return bounds[branch_name]

    network.model.conv_p_nom = Var(list(extendable_converters_i),
                                   domain=NonNegativeReals, bounds=conv_p_nom_bounds)

def define_link_flows(network,snapshots):

    extendable_links_i = network.links.index[network.links.p_nom_extendable]

    fixed_links_i = network.links.index[~ network.links.p_nom_extendable]

    p_max_pu = get_switchable_as_dense(network, 'Link', 'p_max_pu')
    p_min_pu = get_switchable_as_dense(network, 'Link', 'p_min_pu')

    fixed_lower = p_min_pu.loc[:,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom'])
    fixed_upper = p_max_pu.loc[:,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom'])

    bounds = {(cb,sn) : (fixed_lower.at[sn, cb],fixed_upper.at[sn, cb])
              for cb in fixed_links_i for sn in snapshots}
    bounds.update({(cb,sn) : (None,None)
                   for cb in extendable_links_i for sn in snapshots})

    def cb_p_bounds(model,cb_name,snapshot):
        return bounds[cb_name,snapshot]

    network.model.link_p = Var(list(network.links.index),
                               snapshots, domain=Reals, bounds=cb_p_bounds)

    def cb_p_upper(model,cb_name,snapshot):
        return (model.link_p[cb_name,snapshot] <=
                model.link_p_nom[cb_name]
                * p_max_pu.at[snapshot, cb_name])

    network.model.link_p_upper = Constraint(list(extendable_links_i),snapshots,rule=cb_p_upper)


    def cb_p_lower(model,cb_name,snapshot):
        return (model.link_p[cb_name,snapshot] >=
                model.link_p_nom[cb_name]
                * p_min_pu.at[snapshot, cb_name])

    network.model.link_p_lower = Constraint(list(extendable_links_i),snapshots,rule=cb_p_lower)


def define_MILP_link_flows(network,snapshots):

    extendable_links_i = network.links.index[network.links.p_nom_extendable]

    fixed_links_i = network.links.index[~ network.links.p_nom_extendable]

    p_max_pu = get_switchable_as_dense(network, 'Link', 'p_max_pu')
    p_min_pu = get_switchable_as_dense(network, 'Link', 'p_min_pu')

    fixed_lower = p_min_pu.loc[:,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom'])
    fixed_upper = p_max_pu.loc[:,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom'])

    extendable_ptp_links_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]
    extendable_converters_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "converter")]

    bounds = {(cb,sn) : (fixed_lower.at[sn, cb],fixed_upper.at[sn, cb])
              for cb in fixed_links_i for sn in snapshots}
    bounds.update({(cb,sn) : (None,None)
                   for cb in extendable_links_i for sn in snapshots})

    def cb_p_bounds(model,cb_name,snapshot):
        return bounds[cb_name,snapshot]

    network.model.link_p = Var(list(network.links.index),
                               snapshots, domain=Reals, bounds=cb_p_bounds)

    def ext_ptp_p_upper(model,cb_name,snapshot):
        return (model.link_p[cb_name,snapshot] - network.links.at[cb_name,"p_nom_max"]
                * p_max_pu.at[snapshot, cb_name] * network.model.cb_inv_ratio[cb_name] <= 0)

    network.model.ext_link_p_upper = Constraint(list(extendable_ptp_links_i),snapshots,rule=ext_ptp_p_upper)

    def ext_ptp_p_lower(model,cb_name,snapshot):
        return (model.link_p[cb_name,snapshot] - network.links.at[cb_name,"p_nom_max"]
                * p_min_pu.at[snapshot, cb_name] * network.model.cb_inv_ratio[cb_name] >= 0)

    network.model.ext_link_p_lower = Constraint(list(extendable_ptp_links_i),snapshots,rule=ext_ptp_p_lower)

    def conv_p_upper(model,cb_name,snapshot):
        return (model.link_p[cb_name,snapshot] <=
                model.conv_p_nom[cb_name]
                * p_max_pu.at[snapshot, cb_name])

    network.model.conv_p_upper = Constraint(list(extendable_converters_i),snapshots,rule=conv_p_upper)

    def conv_p_lower(model,cb_name,snapshot):
        return (model.link_p[cb_name,snapshot] >=
                model.conv_p_nom[cb_name]
                * p_min_pu.at[snapshot, cb_name])

    network.model.conv_p_lower = Constraint(list(extendable_converters_i),snapshots,rule=conv_p_lower)



def define_passive_branch_flows(network,snapshots,formulation="angles",ptdf_tolerance=0.,milp=False):

    if formulation == "angles":
        if not milp:
            define_passive_branch_flows_with_angles(network,snapshots)
        else:
            define_MILP_passive_branch_flows_with_angles(network,snapshots)
    elif formulation == "ptdf":
        define_passive_branch_flows_with_PTDF(network,snapshots,ptdf_tolerance)
    elif formulation == "cycles":
        define_passive_branch_flows_with_cycles(network,snapshots)
    elif formulation == "kirchhoff":
        define_passive_branch_flows_with_kirchhoff(network,snapshots)



def define_passive_branch_flows_with_angles(network,snapshots):

    network.model.voltage_angles = Var(list(network.buses.index), snapshots)

    slack = {(sub,sn) :
             [[(1,network.model.voltage_angles[network.sub_networks.slack_bus[sub],sn])], "==", 0.]
             for sub in network.sub_networks.index for sn in snapshots}

    l_constraint(network.model,"slack_angle",slack,list(network.sub_networks.index),snapshots)


    passive_branches = network.passive_branches()

    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    flows = {}
    for branch in passive_branches.index:
        bus0 = passive_branches.at[branch,"bus0"]
        bus1 = passive_branches.at[branch,"bus1"]
        bt = branch[0]
        bn = branch[1]
        sub = passive_branches.at[branch,"sub_network"]
        attribute = "r_pu" if network.sub_networks.at[sub,"carrier"] == "DC" else "x_pu"
        y = 1/(passive_branches.at[branch,attribute]*(passive_branches.at[branch,"tap_ratio"] if bt == "Transformer" else 1.))
        for sn in snapshots:
            lhs = LExpression([(y,network.model.voltage_angles[bus0,sn]),
                               (-y,network.model.voltage_angles[bus1,sn]),
                               (-1,network.model.passive_branch_p[bt,bn,sn])],
                              -y*(passive_branches.at[branch,"phase_shift"]*np.pi/180. if bt == "Transformer" else 0.))
            flows[bt,bn,sn] = LConstraint(lhs,"==",LExpression())

    l_constraint(network.model, "passive_branch_p_def", flows,
                 list(passive_branches.index), snapshots)


def define_MILP_passive_branch_flows_with_angles(network,snapshots):

    network.model.voltage_angles = Var(list(network.buses.index), snapshots)

    slack = {(sub,sn) :
             [[(1,network.model.voltage_angles[network.sub_networks.slack_bus[sub],sn])], "==", 0.]
             for sub in network.sub_networks.index for sn in snapshots}

    l_constraint(network.model,"slack_angle",slack,list(network.sub_networks.index),snapshots)


    passive_branches = network.passive_branches()
    extendable_branches_i = passive_branches.index[passive_branches.s_nom_extendable]
    fixed_branches_i = passive_branches.index[~ passive_branches.s_nom_extendable]
    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)
    flows = {}
    for branch in fixed_branches_i:
        bus0 = passive_branches.bus0[branch]
        bus1 = passive_branches.bus1[branch]
        bt = branch[0]
        bn = branch[1]
        sub = passive_branches.sub_network[branch]
        attribute = "r_pu" if network.sub_networks.carrier[sub] == "DC" else "x_pu"
        y = 1/passive_branches[attribute][bt,bn]
        for sn in snapshots:
            lhs = LExpression([(y,network.model.voltage_angles[bus0,sn]),
                               (-y,network.model.voltage_angles[bus1,sn]),
                               (-1,network.model.passive_branch_p[bt,bn,sn])])
            flows[bt,bn,sn] = LConstraint(lhs,"==",LExpression())

    l_constraint(network.model, "passive_fixed_branch_p_def", flows,
             list(fixed_branches_i), snapshots)

    lower_flows = {}
    upper_flows = {}

    for b,branch in passive_branches.iterrows():
        sub = branch.sub_network
        network.lines.loc[b[1],'attribute'] = "r_pu" if network.sub_networks.at[sub,"carrier"] == "DC" else "x_pu"

    existing_network = network.copy(with_time=False)
    for l,line in existing_network.lines.iterrows():
        if line['s_nom'] == 0:
            existing_network.remove('Line',l)

    graph = existing_network.graph(components.passive_branch_components)
    graph.remove_nodes_from(nx.isolates(graph))
    theta_max = existing_network.lines.loc[:, ['s_nom_max']].multiply([branch[branch['attribute']] for b, branch in existing_network.lines.iterrows()], axis=0)
    theta_max_dict = {edge: theta_max.loc[edge[2][1],'s_nom_max']
              for edge in graph.edges(keys=True) if edge[2][1] in existing_network.lines.index}
    nx.set_edge_attributes(graph, 'theta_max', theta_max_dict)
    min_theta = nx.shortest_path_length(graph,weight = 'theta_max')

    for branch in extendable_branches_i:
        bus0 = passive_branches.bus0[branch]
        bus1 = passive_branches.bus1[branch]
        bt = branch[0]
        bn = branch[1]
        attribute = network.lines.loc[bn,'attribute']
        y = 1/(passive_branches.at[branch,attribute]*(passive_branches.at[branch,"tap_ratio"] if bt == "Transformer" else 1.))
        big_M = 3E5
        if bus0 in min_theta.keys():
            if bus1 in min_theta[bus0].keys():
                big_M = min(min_theta[bus0][bus1] * y * 1.05,big_M)

        for sn in snapshots:
            lhs = LExpression([(1,network.model.passive_branch_p[bt,bn,sn]),(-y,network.model.voltage_angles[bus0,sn]),
                               (y,network.model.voltage_angles[bus1,sn]),
                               (big_M,network.model.pb_bin_inv[bt,bn])])
            upper_flows[bt,bn,sn] = LConstraint(lhs,"<=",LExpression(constant=big_M))

            lhs = LExpression([(1,network.model.passive_branch_p[bt,bn,sn]),(-y,network.model.voltage_angles[bus0,sn]),
                               (y,network.model.voltage_angles[bus1,sn]),
                               (-big_M,network.model.pb_bin_inv[bt,bn])])
            lower_flows[bt,bn,sn] = LConstraint(lhs,">=",LExpression(constant=-big_M))

    l_constraint(network.model, "MILP_lower_flow", lower_flows,
                 list(extendable_branches_i), snapshots)

    l_constraint(network.model, "MILP_upper_flow", upper_flows,
                 list(extendable_branches_i), snapshots)

def define_passive_branch_flows_with_PTDF(network,snapshots,ptdf_tolerance=0.):

    passive_branches = network.passive_branches()

    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    flows = {}

    for sub_network in network.sub_networks.obj:
        find_bus_controls(sub_network)

        branches_i = sub_network.branches_i()
        if len(branches_i) > 0:
            calculate_PTDF(sub_network)

            #kill small PTDF values
            sub_network.PTDF[abs(sub_network.PTDF) < ptdf_tolerance] = 0

        for i,branch in enumerate(branches_i):
            bt = branch[0]
            bn = branch[1]

            for sn in snapshots:
                lhs = sum(sub_network.PTDF[i,j]*network._p_balance[bus,sn]
                          for j,bus in enumerate(sub_network.buses_o)
                          if sub_network.PTDF[i,j] != 0)
                rhs = LExpression([(1,network.model.passive_branch_p[bt,bn,sn])])
                flows[bt,bn,sn] = LConstraint(lhs,"==",rhs)


    l_constraint(network.model, "passive_branch_p_def", flows,
                 list(passive_branches.index), snapshots)


def define_passive_branch_flows_with_cycles(network,snapshots):

    for sub_network in network.sub_networks.obj:
        find_tree(sub_network)
        find_cycles(sub_network)

        #following is necessary to calculate angles post-facto
        find_bus_controls(sub_network)
        if len(sub_network.branches_i()) > 0:
            calculate_B_H(sub_network)


    cycle_index = [(sub_network.name,i)
                   for sub_network in network.sub_networks.obj
                   for i in range(sub_network.C.shape[1])]

    network.model.cycles = Var(cycle_index, snapshots, domain=Reals, bounds=(None,None))

    passive_branches = network.passive_branches()


    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    flows = {}

    for sn in network.sub_networks.obj:
        branches = sn.branches()
        buses = sn.buses()
        for i,branch in enumerate(branches.index):
            bt = branch[0]
            bn = branch[1]

            cycle_is = sn.C[i,:].nonzero()[1]
            tree_is = sn.T[i,:].nonzero()[1]

            for snapshot in snapshots:
                expr = LExpression([(sn.C[i,j], network.model.cycles[sn.name,j,snapshot])
                                    for j in cycle_is])
                lhs = expr + sum(sn.T[i,j]*network._p_balance[buses.index[j],snapshot]
                                 for j in tree_is)

                rhs = LExpression([(1,network.model.passive_branch_p[bt,bn,snapshot])])

                flows[bt,bn,snapshot] = LConstraint(lhs,"==",rhs)

    l_constraint(network.model, "passive_branch_p_def", flows,
                 list(passive_branches.index), snapshots)

    cycle_constraints = {}


    for sn in network.sub_networks.obj:

        branches = sn.branches()
        attribute = "r_pu" if network.sub_networks.at[sn.name,"carrier"] == "DC" else "x_pu"

        for j in range(sn.C.shape[1]):

            cycle_is = sn.C[:,j].nonzero()[0]

            for snapshot in snapshots:
                lhs = LExpression([(branches.at[branches.index[i],attribute]*
                                   (branches.at[branches.index[i],"tap_ratio"] if branches.index[i][0] == "Transformer" else 1.)*sn.C[i,j],
                                    network.model.passive_branch_p[branches.index[i][0],branches.index[i][1],snapshot])
                                   for i in cycle_is])
                cycle_constraints[sn.name,j,snapshot] = LConstraint(lhs,"==",LExpression())

    l_constraint(network.model, "cycle_constraints", cycle_constraints,
                 cycle_index, snapshots)




def define_passive_branch_flows_with_kirchhoff(network,snapshots):

    for sub_network in network.sub_networks.obj:
        find_tree(sub_network)
        find_cycles(sub_network)

        #following is necessary to calculate angles post-facto
        find_bus_controls(sub_network)
        if len(sub_network.branches_i()) > 0:
            calculate_B_H(sub_network)

    cycle_index = [(sub_network.name,i)
                   for sub_network in network.sub_networks.obj
                   for i in range(sub_network.C.shape[1])]

    passive_branches = network.passive_branches()

    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)


    cycle_constraints = {}

    for sn in network.sub_networks.obj:

        branches = sn.branches()
        attribute = "r_pu" if network.sub_networks.at[sn.name,"carrier"] == "DC" else "x_pu"

        for j in range(sn.C.shape[1]):

            cycle_is = sn.C[:,j].nonzero()[0]

            for snapshot in snapshots:
                lhs = LExpression([(branches.at[branches.index[i],attribute]*
                                    (branches.at[branches.index[i],"tap_ratio"] if branches.index[i][0] == "Transformer" else 1.)*sn.C[i,j],
                                    network.model.passive_branch_p[branches.index[i][0], branches.index[i][1], snapshot])
                                   for i in cycle_is])
                cycle_constraints[sn.name,j,snapshot] = LConstraint(lhs,"==",LExpression())

    l_constraint(network.model, "cycle_constraints", cycle_constraints,
                 cycle_index, snapshots)

def define_passive_branch_constraints(network,snapshots):

    passive_branches = network.passive_branches()
    extendable_branches = passive_branches[passive_branches.s_nom_extendable]
    fixed_branches = passive_branches[~ passive_branches.s_nom_extendable]

    flow_upper = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    "<=", fixed_branches.s_nom[b]]
                  for b in fixed_branches.index
                  for sn in snapshots}

    flow_upper.update({(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn]),
                                          (-1,network.model.passive_branch_s_nom[b[0],b[1]])],"<=",0]
                       for b in extendable_branches.index
                       for sn in snapshots})

    l_constraint(network.model, "flow_upper", flow_upper,
                 list(passive_branches.index), snapshots)

    flow_lower = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    ">=", -fixed_branches.s_nom[b]]
                  for b in fixed_branches.index
                  for sn in snapshots}

    flow_lower.update({(b[0],b[1],sn): [[(1,network.model.passive_branch_p[b[0],b[1],sn]),
                                         (1,network.model.passive_branch_s_nom[b[0],b[1]])],">=",0]
                       for b in extendable_branches.index
                       for sn in snapshots})

    l_constraint(network.model, "flow_lower", flow_lower,
                 list(passive_branches.index), snapshots)


def define_MILP_passive_branch_constraints(network,snapshots):

    passive_branches = network.passive_branches()
    extendable_branches = passive_branches[passive_branches.s_nom_extendable]
    fixed_branches = passive_branches[~ passive_branches.s_nom_extendable]

    def pb_inv_ratio_rule(model, branch_type, branch_name):
        return (model.pb_inv_ratio[branch_type, branch_name] - model.pb_bin_inv[branch_type, branch_name] <= 0)

    network.model.pb_inv_ratio_constraint = Constraint(list(extendable_branches.index),rule=pb_inv_ratio_rule)

    flow_upper = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    "<=", fixed_branches.s_nom[b]]
                  for b in fixed_branches.index
                  for sn in snapshots}

    l_constraint(network.model, "flow_upper", flow_upper,
                 list(fixed_branches.index), snapshots)

    flow_lower = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    ">=", -fixed_branches.s_nom[b]]
                  for b in fixed_branches.index
                  for sn in snapshots}

    l_constraint(network.model, "flow_lower", flow_lower,
                 list(fixed_branches.index), snapshots)


    integer_flow_upper = ({(b[0], b[1], sn): [[(1, network.model.passive_branch_p[b[0], b[1], sn]),
                                           (-extendable_branches.at[(b[0], b[1]),"s_nom_max"], network.model.pb_inv_ratio[b[0],b[1]])], "<=", 0]
                       for b in extendable_branches.index
                       for sn in snapshots})

    l_constraint(network.model, "integer_flow_upper", integer_flow_upper,
                 list(extendable_branches.index), snapshots)

    integer_flow_lower = ({(b[0], b[1], sn): [[(1, network.model.passive_branch_p[b[0], b[1], sn]),
                                           (extendable_branches.at[(b[0], b[1]), "s_nom_max"], network.model.pb_inv_ratio[b[0],b[1]])], ">=", 0]
                       for b in extendable_branches.index
                       for sn in snapshots})

    l_constraint(network.model, "integer_flow_lower", integer_flow_lower,
                 list(extendable_branches.index), snapshots)

def define_MILP_controllable_branch_constraints(network,snapshots):

    extendable_ptp_links_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]

    def cb_inv_ratio_rule(model, branch_name):
        return (model.cb_inv_ratio[branch_name] - model.cb_bin_inv[branch_name] <= 0)

    network.model.cb_inv_ratio_constraint = Constraint(list(extendable_ptp_links_i),rule=cb_inv_ratio_rule)

def define_MILP_minimum_ratio(network):

    extendable_ptp_links = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]
    links_with_minimum_i = extendable_ptp_links.index[extendable_ptp_links.p_nom_min > 0]

    big_M = -1

    def cb_minimum_ratio_rule(model, branch_name):
        return (model.cb_inv_ratio[branch_name] - big_M * (1 - model.cb_bin_inv[branch_name]) >= network.links.at[branch_name,"p_nom_min"]/network.links.at[branch_name,"p_nom_max"])

    network.model.cb_minimum_ratio = Constraint(list(links_with_minimum_i),rule=cb_minimum_ratio_rule)

    passive_branches = network.passive_branches()
    extendable_branches = passive_branches[passive_branches.s_nom_extendable]
    branches_with_minimum = extendable_branches[extendable_branches.s_nom_min > 0]

    def pb_minimum_ratio_rule(model, branch_type, branch_name):
        return (model.pb_inv_ratio[branch_type, branch_name] - big_M * (1 - model.pb_bin_inv[branch_type, branch_name]) >= branches_with_minimum.loc[(branch_type,branch_name),"s_nom_min"]/branches_with_minimum.loc[(branch_type,branch_name),"s_nom_max"])

    network.model.pb_minimum_ratio = Constraint(list(branches_with_minimum.index),rule=pb_minimum_ratio_rule)

def define_parallelism_constraints(network,snapshots):
    passive_branches = network.passive_branches()
    extendable_branches = passive_branches[passive_branches.s_nom_extendable]
    extendable_ptp_links = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]

    def pb_parallelism_rule(model,base):
        parallel_lines = extendable_branches[extendable_branches["base_branch"] == base]
        return (sum([model.pb_bin_inv[line] for line in parallel_lines.index]) <= 1)

    network.model.pb_parallelism = Constraint(list(extendable_branches["base_branch"].dropna().unique()),rule=pb_parallelism_rule)

    def cb_parallelism_rule(model,base):
        parallel_links = extendable_ptp_links[extendable_ptp_links["base_branch"] == base]
        return (sum([model.cb_bin_inv[link] for link in parallel_links.index]) <= 1)

    network.model.cb_parallelism = Constraint(list(extendable_ptp_links["base_branch"].dropna().unique()),rule=cb_parallelism_rule)

def define_nodal_balances(network,snapshots):
    """Construct the nodal balance for all elements except the passive
    branches.

    Store the nodal balance expression in network._p_balance.
    """

    #dictionary for constraints
    network._p_balance = {(bus,sn) : LExpression()
                          for bus in network.buses.index
                          for sn in snapshots}

    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency')

    for cb in network.links.index:
        bus0 = network.links.at[cb,"bus0"]
        bus1 = network.links.at[cb,"bus1"]

        for sn in snapshots:
            network._p_balance[bus0,sn].variables.append((-1,network.model.link_p[cb,sn]))
            network._p_balance[bus1,sn].variables.append((efficiency.at[sn,cb],network.model.link_p[cb,sn]))


    for gen in network.generators.index:
        bus = network.generators.bus[gen]
        sign = network.generators.sign[gen]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.generator_p[gen,sn]))

    load_p_set = get_switchable_as_dense(network, 'Load', 'p_set')
    for load in network.loads.index:
        bus = network.loads.bus[load]
        sign = network.loads.sign[load]
        for sn in snapshots:
            network._p_balance[bus,sn].constant += sign*load_p_set.at[sn,load]

    for su in network.storage_units.index:
        bus = network.storage_units.bus[su]
        sign = network.storage_units.sign[su]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.storage_p_dispatch[su,sn]))
            network._p_balance[bus,sn].variables.append((-sign,network.model.storage_p_store[su,sn]))

    for store in network.stores.index:
        bus = network.stores.bus[store]
        sign = network.stores.sign[store]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.store_p[store,sn]))


def define_nodal_balance_constraints(network,snapshots):

    passive_branches = network.passive_branches()


    for branch in passive_branches.index:
        bus0 = passive_branches.bus0[branch]
        bus1 = passive_branches.bus1[branch]
        bt = branch[0]
        bn = branch[1]
        for sn in snapshots:
            network._p_balance[bus0,sn].variables.append((-1,network.model.passive_branch_p[bt,bn,sn]))
            network._p_balance[bus1,sn].variables.append((1,network.model.passive_branch_p[bt,bn,sn]))

    power_balance = {k: LConstraint(v,"==",LExpression()) for k,v in iteritems(network._p_balance)}

    l_constraint(network.model, "power_balance", power_balance,
                 list(network.buses.index), snapshots)


def define_sub_network_balance_constraints(network,snapshots):

    sn_balance = {}

    for sub_network in network.sub_networks.obj:
        for sn in snapshots:
            sn_balance[sub_network.name,sn] = LConstraint(LExpression(),"==",LExpression())
            for bus in sub_network.buses().index:
                sn_balance[sub_network.name,sn].lhs.variables.extend(network._p_balance[bus,sn].variables)
                sn_balance[sub_network.name,sn].lhs.constant += network._p_balance[bus,sn].constant

    l_constraint(network.model,"sub_network_balance_constraint", sn_balance,
                 list(network.sub_networks.index), snapshots)


def define_co2_constraint(network,snapshots):

    def co2_constraint(model):

        #use the prime mover carrier
        co2_gens = sum(network.carriers.at[network.generators.at[gen,"carrier"],"co2_emissions"]
                       * (1/network.generators.at[gen,"efficiency"])
                       * network.snapshot_weightings[sn]
                       * model.generator_p[gen,sn]
                       for gen in network.generators.index
                       for sn in snapshots)

        #store inherits the carrier from the bus
        co2_stores = sum(network.carriers.at[network.buses.at[network.stores.at[store,"bus"],"carrier"],"co2_emissions"]
                         * network.snapshot_weightings[sn]
                         * model.store_p[store,sn]
                         for store in network.stores.index
                         for sn in snapshots)

        return  co2_gens + co2_stores <= network.co2_limit

    network.model.co2_constraint = Constraint(rule=co2_constraint)





def define_linear_objective(network,snapshots):

    model = network.model

    extendable_generators = network.generators[network.generators.p_nom_extendable]

    ext_sus = network.storage_units[network.storage_units.p_nom_extendable]

    ext_stores = network.stores[network.stores.e_nom_extendable]

    passive_branches = network.passive_branches()

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    extendable_links = network.links[network.links.p_nom_extendable]

    objective = LExpression()

    objective.variables.extend([(network.generators.at[gen,"marginal_cost"]
                                 * network.snapshot_weightings[sn],
                                 model.generator_p[gen,sn])
                                for gen in network.generators.index
                                for sn in snapshots])

    objective.variables.extend([(network.storage_units.at[su,"marginal_cost"]
                                 * network.snapshot_weightings[sn],
                                 model.storage_p_dispatch[su,sn])
                                for su in network.storage_units.index
                                for sn in snapshots])

    objective.variables.extend([(network.stores.at[store,"marginal_cost"]
                                 * network.snapshot_weightings[sn],
                                 model.store_p[store,sn])
                                for store in network.stores.index
                                for sn in snapshots])

    objective.variables.extend([(network.links.at[link,"marginal_cost"]
                                 * network.snapshot_weightings[sn],
                                 model.link_p[link,sn])
                                for link in network.links.index
                                for sn in snapshots])


    #NB: for capital costs we subtract the costs of existing infrastructure p_nom/s_nom

    objective.variables.extend([(extendable_generators.at[gen,"capital_cost"], model.generator_p_nom[gen])
                                for gen in extendable_generators.index])
    objective.constant -= (extendable_generators.capital_cost * extendable_generators.p_nom).sum()

    objective.variables.extend([(ext_sus.at[su,"capital_cost"], model.storage_p_nom[su])
                                for su in ext_sus.index])
    objective.constant -= (ext_sus.capital_cost*ext_sus.p_nom).sum()

    objective.variables.extend([(ext_stores.at[store,"capital_cost"], model.store_e_nom[store])
                                for store in ext_stores.index])
    objective.constant -= (ext_stores.capital_cost*ext_stores.e_nom).sum()

    objective.variables.extend([(extendable_passive_branches.at[b,"capital_cost"], model.passive_branch_s_nom[b])
                                for b in extendable_passive_branches.index])
    objective.constant -= (extendable_passive_branches.capital_cost * extendable_passive_branches.s_nom).sum()

    objective.variables.extend([(extendable_links.at[b,"capital_cost"], model.link_p_nom[b])
                                for b in extendable_links.index])
    objective.constant -= (extendable_links.capital_cost * extendable_links.p_nom).sum()

    l_objective(model,objective)


def define_MILP_objective(network, snapshots, annuity_factor):

    model = network.model

    extendable_generators = network.generators[network.generators.p_nom_extendable]

    def gen_capital_cost_rule(model,gen):
        return extendable_generators.at[gen,"capital_cost"]

    model.gen_capital_cost = Param(list(extendable_generators.index), initialize=gen_capital_cost_rule,mutable=True)

    #slack_generators = network.generators[network.generators.carrier == "Slack"]
    #def slack_marginal_cost_rule(model,gen):
    #    return network.generators.at[gen,"marginal_cost"]
    #model.slack_marginal_cost = Param(list(slack_generators.index), initialize=slack_marginal_cost_rule,mutable=True)

    ext_sus = network.storage_units[network.storage_units.p_nom_extendable]

    ext_stores = network.stores[network.stores.e_nom_extendable]

    passive_branches = network.passive_branches()

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    extendable_ptp_links = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]
    extendable_converters = network.links[network.links.p_nom_extendable & (network.links.branch_type == "converter")]

    objective = LExpression()
    #gen_i = network.generators[network.generators.carrier != "Slack"].index

    for sn in snapshots:
        weight = network.snapshot_weightings[sn]
        for gen in network.generators.index:
            coefficient = network.generators.at[gen, "marginal_cost"] * weight
            objective.variables.extend([(coefficient, model.generator_p[gen, sn])])

        for su in network.storage_units.index:
            coefficient = network.storage_units.at[su, "marginal_cost"] * weight
            objective.variables.extend([(coefficient, model.storage_p_dispatch[su,sn])])

        for store in network.stores.index:
            coefficient = network.stores.at[store, "marginal_cost"] * weight
            objective.variables.extend([(coefficient, model.store_p[store,sn])])

        for link in network.links.index:
            coefficient = network.links.at[link, "marginal_cost"] * weight
            objective.variables.extend([(coefficient, model.link_p[link,sn])])


    #NB: for capital costs we subtract the costs of existing infrastructure p_nom/s_nom

    objective.variables.extend([(ext_sus.at[su,"capital_cost"] * annuity_factor, model.storage_p_nom[su])
                                for su in ext_sus.index])
    objective.constant -= (ext_sus.capital_cost * annuity_factor * ext_sus.p_nom).sum()

    objective.variables.extend([(ext_stores.at[store,"capital_cost"] * annuity_factor, model.store_e_nom[store])
                                for store in ext_stores.index])
    objective.constant -= (ext_stores.capital_cost * annuity_factor * ext_stores.e_nom).sum()

    objective.variables.extend([(extendable_passive_branches.at[b, "capital_cost"] * annuity_factor * extendable_passive_branches.at[b, "s_nom_max"], model.pb_inv_ratio[b]) for b in extendable_passive_branches.index])
    objective.constant -= (extendable_passive_branches.capital_cost * annuity_factor * extendable_passive_branches.s_nom).sum()

    objective.variables.extend([(extendable_ptp_links.at[b, "capital_cost"] * annuity_factor * extendable_ptp_links.at[b, "p_nom_max"], model.cb_inv_ratio[b]) for b in extendable_ptp_links.index])
    objective.constant -= (extendable_ptp_links.capital_cost * annuity_factor * extendable_ptp_links.p_nom).sum()

    objective.variables.extend([(extendable_converters.at[b,"capital_cost"] * annuity_factor, model.conv_p_nom[b])
                                for b in extendable_converters.index])
    objective.constant -= (extendable_converters.capital_cost * annuity_factor * extendable_converters.p_nom).sum()

    objective.constant -= (extendable_generators.capital_cost * annuity_factor * extendable_generators.p_nom).sum()

    l_objective(model,objective)

    #TODO include storage capital costs if storage is created
    model.comp_obj = Expression(expr=sum(model.gen_capital_cost[gen] * annuity_factor * model.generator_p_nom[gen]
                                         for gen in extendable_generators.index))

    model.new_obj = model.objective.expr + model.comp_obj.expr# + model.comp_slack_obj

    model.del_component("objective")

    model.objective = Objective(expr = model.new_obj)

def extract_optimisation_results(network, snapshots, formulation="angles", milp=False):

    from .components import \
        passive_branch_components, branch_components, controllable_one_port_components

    if isinstance(snapshots, pd.DatetimeIndex) and _pd_version < '0.18.0':
        # Work around pandas bug #12050 (https://github.com/pydata/pandas/issues/12050)
        snapshots = pd.Index(snapshots.values)

    allocate_series_dataframes(network, {'Generator': ['p'],
                                         'Load': ['p'],
                                         'StorageUnit': ['p', 'state_of_charge', 'spill'],
                                         'Store': ['p', 'e'],
                                         'Bus': ['p', 'v_ang', 'v_mag_pu', 'marginal_price'],
                                         'Line': ['p0', 'p1'],
                                         'Transformer': ['p0', 'p1'],
                                         'Link': ['p0', 'p1']})

    #get value of objective function
    network.objective = network.results["Problem"][0]["Lower bound"]

    model = network.model

    def as_series(indexedvar):
        return pd.Series(indexedvar.get_values())

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    if len(network.generators):
        set_from_series(network.generators_t.p, as_series(model.generator_p))

    if len(network.storage_units):
        set_from_series(network.storage_units_t.p,
                        as_series(model.storage_p_dispatch)
                        - as_series(model.storage_p_store))

        set_from_series(network.storage_units_t.state_of_charge,
                        as_series(model.state_of_charge))

        if (network.storage_units_t.inflow.max() > 0).any():
            set_from_series(network.storage_units_t.spill,
                            as_series(model.storage_p_spill))
        network.storage_units_t.spill.fillna(0, inplace=True) #p_spill doesn't exist if inflow=0

    if len(network.stores):
        set_from_series(network.stores_t.p, as_series(model.store_p))
        set_from_series(network.stores_t.e, as_series(model.store_e))

    if len(network.loads):
        load_p_set = get_switchable_as_dense(network, 'Load', 'p_set')
        network.loads_t["p"].loc[snapshots] = load_p_set.loc[snapshots]

    if len(network.buses):
        network.buses_t.p.loc[snapshots] = \
            pd.concat({c.name:
                       c.pnl.p.loc[snapshots].multiply(c.df.sign, axis=1)
                       .groupby(c.df.bus, axis=1).sum()
                       for c in network.iterate_components(controllable_one_port_components)}) \
              .sum(level=1) \
              .reindex_axis(network.buses_t.p.columns, axis=1, fill_value=0.)


    # passive branches
    passive_branches = as_series(model.passive_branch_p)
    for c in network.iterate_components(passive_branch_components):
        set_from_series(c.pnl.p0, passive_branches.loc[c.name])
        c.pnl.p1.loc[snapshots] = - c.pnl.p0.loc[snapshots]


    # active branches
    if len(network.links):
        set_from_series(network.links_t.p0, as_series(model.link_p))

        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency')

        network.links_t.p1.loc[snapshots] = - network.links_t.p0.loc[snapshots]*efficiency.loc[snapshots,:]

        network.buses_t.p.loc[snapshots] -= (network.links_t.p0.loc[snapshots]
                                             .groupby(network.links.bus0, axis=1).sum()
                                             .reindex(columns=network.buses_t.p.columns, fill_value=0.))

        network.buses_t.p.loc[snapshots] -= (network.links_t.p1.loc[snapshots]
                                             .groupby(network.links.bus1, axis=1).sum()
                                             .reindex(columns=network.buses_t.p.columns, fill_value=0.))


    if len(network.buses):
        if formulation in {'angles', 'kirchhoff'}:
            set_from_series(network.buses_t.marginal_price,
                            pd.Series(list(model.power_balance.values()),
                                      index=pd.MultiIndex.from_tuples(list(model.power_balance.keys())))
                            .map(pd.Series(list(model.dual.values()), index=pd.Index(list(model.dual.keys())))))

        if formulation == "angles":
            set_from_series(network.buses_t.v_ang,
                            as_series(model.voltage_angles))
        elif formulation in ["ptdf","cycles","kirchhoff"]:
            for sn in network.sub_networks.obj:
                network.buses_t.v_ang.loc[snapshots,sn.slack_bus] = 0.
                if len(sn.pvpqs) > 0:
                    network.buses_t.v_ang.loc[snapshots,sn.pvpqs] = spsolve(sn.B[1:, 1:], network.buses_t.p.loc[snapshots,sn.pvpqs].T).T

        network.buses_t.v_mag_pu.loc[snapshots,network.buses.carrier=="AC"] = 1.
        network.buses_t.v_mag_pu.loc[snapshots,network.buses.carrier=="DC"] = 1 + network.buses_t.v_ang.loc[snapshots,network.buses.carrier=="DC"]


    #now that we've used the angles to calculate the flow, set the DC ones to zero
    network.buses_t.v_ang.loc[snapshots,network.buses.carrier=="DC"] = 0.

    network.generators.p_nom_opt = network.generators.p_nom

    network.generators.loc[network.generators.p_nom_extendable, 'p_nom_opt'] = \
        as_series(network.model.generator_p_nom)

    network.storage_units.p_nom_opt = network.storage_units.p_nom

    network.storage_units.loc[network.storage_units.p_nom_extendable, 'p_nom_opt'] = \
        as_series(network.model.storage_p_nom)

    network.stores.e_nom_opt = network.stores.e_nom

    network.stores.loc[network.stores.e_nom_extendable, 'e_nom_opt'] = \
        as_series(network.model.store_e_nom)

    if not milp:
        s_nom_extendable_passive_branches = as_series(model.passive_branch_s_nom)
        for c in network.iterate_components(passive_branch_components):
            c.df['s_nom_opt'] = c.df.s_nom
            if c.df.s_nom_extendable.any():
                c.df.loc[c.df.s_nom_extendable, 's_nom_opt'] = s_nom_extendable_passive_branches.loc[c.name]

        network.links.p_nom_opt = network.links.p_nom

        network.links.loc[network.links.p_nom_extendable, "p_nom_opt"] = \
            as_series(network.model.link_p_nom)
    else:
        passive_branches = network.passive_branches()
        extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]
        pb_inv_ratio_series = as_series(model.pb_inv_ratio)

        if len(pb_inv_ratio_series):
            pb_inv_ratio_series.index.levels[1].name = extendable_passive_branches.index.levels[1].name
            s_nom_MILP_extendable_passive_branches = pb_inv_ratio_series * extendable_passive_branches.loc[:,"s_nom_max"]

        for c in network.iterate_components(passive_branch_components):
            c.df['s_nom_opt'] = c.df.s_nom
            if c.df.s_nom_extendable.any():
                c.df.loc[c.df.s_nom_extendable, 's_nom_opt'] = s_nom_MILP_extendable_passive_branches.loc[c.name]

        extendable_ptp_links = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]
        cb_inv_ratio_series = as_series(model.cb_inv_ratio)

        network.links.p_nom_opt = network.links.p_nom
        if len(cb_inv_ratio_series):
            #cb_inv_ratio_series.index.levels[1].name = controllable_ptp_links.index.levels[1].name
            p_nom_MILP_extendable_ptp_links = cb_inv_ratio_series * extendable_ptp_links.loc[:, "p_nom_max"]
            network.links.loc[p_nom_MILP_extendable_ptp_links.index,"p_nom_opt"] = p_nom_MILP_extendable_ptp_links
            #for t in network.iterate_components(controllable_branch_types):
                #if t.df.p_nom_extendable.any():
                    #t.df.loc[t.df.p_nom_extendable, 'p_nom_opt'] = p_nom_MILP_controllable_ptp_links.loc[t.name]

        network.links.loc[network.links.p_nom_extendable & (network.links.branch_type == "converter"), "p_nom_opt"] = \
            as_series(network.model.conv_p_nom)

    if network.co2_limit is not None:
        try:
            network.co2_price = - network.model.dual[network.model.co2_constraint]
        except (AttributeError, KeyError) as e:
            logger.warning("Could not read out co2_price, although a co2_limit was set")


def network_lopf(network, snapshots=None, solver_name="glpk",
                 skip_pre=False, extra_functionality=None, solver_options={},
                 keep_files=False, formulation="angles", ptdf_tolerance=0.,
                 free_memory={}, milp=False, parameters=None):
    """
    Linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.now
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    formulation : string
        Formulation of the linear power flow equations to use; must be
        one of ["angles","cycles","kirchhoff","ptdf"]
    ptdf_tolerance : float
        Value below which PTDF entries are ignored
    free_memory : set, default {}
        Any subset of {'pypsa', 'pyomo_hack'}. Beware that the
        pyomo_hack is slow and only tested on small systems.  Stash
        time series data and/or pyomo model away while the solver runs.

    Returns
    -------
    None
    """

    if not skip_pre:
        network.determine_network_topology()
        calculate_dependent_values(network)
        for sub_network in network.sub_networks.obj:
            find_slack_bus(sub_network)
        logger.info("Performed preliminary steps")


    if snapshots is None:
        snapshots = [network.now]


    logger.info("Building pyomo model using `%s` formulation", formulation)
    network.model = ConcreteModel("Mixed Integer Optimal Investment & Operation")

    define_generator_variables_constraints(network,snapshots)

    define_storage_variables_constraints(network,snapshots)

    define_store_variables_constraints(network,snapshots)

    if not milp:
        define_branch_extension_variables(network,snapshots)

        define_link_flows(network, snapshots)
    else:
        define_MILP_branch_extension_variables(network,snapshots)

        define_MILP_link_flows(network, snapshots)

    define_nodal_balances(network,snapshots)

    define_passive_branch_flows(network,snapshots,formulation,ptdf_tolerance,milp)

    if not milp:
        define_passive_branch_constraints(network,snapshots)
    else:
        define_MILP_passive_branch_constraints(network,snapshots)

        define_MILP_controllable_branch_constraints(network, snapshots)

    if formulation in ["angles", "kirchhoff"]:
        define_nodal_balance_constraints(network,snapshots)
    elif formulation in ["ptdf", "cycles"]:
        define_sub_network_balance_constraints(network,snapshots)

    if network.co2_limit is not None:
        define_co2_constraint(network,snapshots)

    #force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    if extra_functionality is not None:
        extra_functionality(network,snapshots)

    if not milp:
        define_linear_objective(network,snapshots)
    else:
        annuity_factor = np.pmt(parameters["discount_rate"], parameters["lifetime"], -1)
        define_MILP_objective(network,snapshots,annuity_factor)

    if milp:
        define_parallelism_constraints(network,snapshots)
        define_MILP_minimum_ratio(network)

    #tidy up auxilliary expressions
    del network._p_balance

    logger.info("Solving model using %s", solver_name)
    opt = SolverFactory(solver_name)

    patch_optsolver_record_memusage_before_solving(opt, network)

    def convergence_status():
        status = network.results["Solver"][0]["Status"].key
        termination_condition = network.results["Solver"][0]["Termination condition"].key

        if status == "ok" and termination_condition == "optimal":
            logger.info("Optimization successful")
            extract_optimisation_results(network, snapshots, formulation, milp=milp)
        elif status == "warning" and termination_condition == "other":
            logger.warn("WARNING! Optimization might be sub-optimal. Writing output anyway")
            extract_optimisation_results(network, snapshots, formulation, milp=milp)
        else:
            logger.error("Optimisation failed with status %s and terminal condition %s"
                         % (status, termination_condition))

        return status, termination_condition

    def fix_model(status=False):

        transfer_increase = 1E-5
        fixing_threshold = 1E-8

        for epb in extendable_passive_branches.index:
            network.model.pb_bin_inv[epb].fixed = status
            network.model.pb_inv_ratio[epb].fixed = status
            if status:
                if network.model.pb_bin_inv[epb].value == 1:
                    network.model.pb_inv_ratio[epb].value = min(network.model.pb_inv_ratio[epb].value * (1 + transfer_increase), 1.0)

        for ecb in extendable_ptp_links_i:
            network.model.cb_bin_inv[ecb].fixed = status
            network.model.cb_inv_ratio[ecb].fixed = status
            if status:
                if network.model.cb_bin_inv[ecb].value == 1:
                    network.model.cb_inv_ratio[ecb].value = min(network.model.cb_inv_ratio[ecb].value * (1 + transfer_increase), 1.0)

        for gen in extendable_gens.index:
            network.model.generator_p_nom[gen].fixed = status
            network.model.gen_capital_cost[gen] = network.generators.loc[gen, "capital_cost"] * (not status)

        if status:
            for gen in network.generators.index:
                for sn in snapshots:
                    if abs(network.model.generator_p[(gen, sn)].value) <= fixing_threshold:
                        network.model.generator_p[(gen, sn)].fixed = status
                        network.model.generator_p[(gen, sn)].value = 0
        else:
            for gen in network.generators.index:
                for sn in snapshots:
                    network.model.generator_p[(gen, sn)].fixed = status

        network.model.preprocess()

    def solver_run():
        if 'pypsa' in free_memory:
            with empty_network(network):
                network.results = opt.solve(network.model, suffixes=["dual"],
                                            keepfiles=keep_files, options=solver_options)
        else:
            network.results = opt.solve(network.model, suffixes=["dual"],
                                        keepfiles=keep_files, options=solver_options)
        print("Solver finished at {:%H:%M}.".format(datetime.datetime.now()))

    def model_run():

        starttime = datetime.datetime.now()
        print("Free model run at {:%H:%M}...".format(starttime))

        fix_model(False)

        solver_run()

        convergence_status()

        if logger.level > 0:
            network.results.write()

        if milp:

            print("Fixed model run at {:%H:%M}...".format(datetime.datetime.now()))

            fix_model(True)

            solver_run()

            convergence_status()

            upper_dual = pd.Series(list(network.model.MILP_upper_flow.values()),index=pd.MultiIndex.from_tuples(list(network.model.MILP_upper_flow.keys()))).map(pd.Series(list(network.model.dual.values()), index=pd.Index(list(network.model.dual.keys())))).fillna(0)
            upper_dual.name = 'upper_dual'
            lower_dual = pd.Series(list(network.model.MILP_lower_flow.values()), index=pd.MultiIndex.from_tuples(list(network.model.MILP_lower_flow.keys()))).map(pd.Series(list(network.model.dual.values()), index=pd.Index(list(network.model.dual.keys())))).fillna(0)
            lower_dual.name = 'lower_dual'
            inv_bin = pd.Series({l: network.model.pb_bin_inv[l].value == 0 for l in network.model.pb_bin_inv})
            inv_bin.index = inv_bin.index.droplevel(0)
            branch_dual = pd.concat([upper_dual,lower_dual],axis=1).stack().multiply(inv_bin,level=1)
            if branch_dual.abs().values.max() > 1E-1:
                print("Disjunctive KVL constraint dual of {:.2f} at {}".format(branch_dual.abs().values.max(), branch_dual.abs().idxmax()))
                print(branch_dual[branch_dual.abs()>1E-1])
                sys.exit()

        print("Run finished at {:%H:%M}.".format(datetime.datetime.now()))
        print("Duration: {}.".format(datetime.datetime.now() - starttime))

    if isinstance(free_memory, string_types):
        free_memory = {free_memory}

    if 'pyomo_hack' in free_memory:
        patch_optsolver_free_network_before_solving(opt, network.model)

    if milp:

        passive_branches = network.passive_branches()
        extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]
        extendable_ptp_links_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]
        extendable_gens = network.generators[network.generators.p_nom_extendable]

        model_run()

    else:
        solver_run()

        convergence_status()

        if logger.level > 0:
            network.results.write()


