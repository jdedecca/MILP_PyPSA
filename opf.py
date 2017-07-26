

## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David
## Schlachtberger (FIAS)
## Copyright 2017 João Gorenstein Dedecca

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

"""Optimal Power Flow functions.
"""


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
                           Suffix, Expression, Boolean, Param, NonNegativeReals, value, Binary)
from pyomo.opt import SolverFactory
from itertools import chain
import logging
logger = logging.getLogger(__name__)


from distutils.version import StrictVersion, LooseVersion
try:
    _pd_version = StrictVersion(pd.__version__)
except ValueError:
    _pd_version = LooseVersion(pd.__version__)

from .pf import (calculate_dependent_values, find_slack_bus,
                 find_bus_controls, calculate_B_H, calculate_PTDF, find_tree,
                 find_cycles, _as_snapshots)
from .opt import (l_constraint, l_objective, LExpression, LConstraint,
                  patch_optsolver_free_model_before_solving,
                  patch_optsolver_record_memusage_before_solving,
                  empty_network)
from .descriptors import get_switchable_as_dense, allocate_series_dataframes

#Custom imports
import time, datetime, operator
from copy import deepcopy
import os,sys,psutil, gc
import networkx as nx
import pyomo

def network_opf(network,snapshots=None):
    """Optimal power flow for snapshots."""

    raise NotImplementedError("Non-linear optimal power flow not supported yet")



def define_generator_variables_constraints(network,snapshots):

    extendable_gens_i = network.generators.index[network.generators.p_nom_extendable]
    fixed_gens_i = network.generators.index[~network.generators.p_nom_extendable & ~network.generators.committable]
    fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]

    if (network.generators.p_nom_extendable & network.generators.committable).any():
        logger.warning("The following generators have both investment optimisation and unit commitment:\n{}\nCurrently PyPSA cannot do both these functions, so PyPSA is choosing investment optimisation for these generators.".format(network.generators.index[network.generators.p_nom_extendable & network.generators.committable]))

    p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu')
    p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu')

    ## Define generator dispatch variables ##

    gen_p_bounds = {(gen,sn) : (None,None)
                    for gen in extendable_gens_i | fixed_committable_gens_i
                    for sn in snapshots}

    if len(fixed_gens_i):
        var_lower = (p_min_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])).fillna(0.)
        var_upper = p_max_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])

        gen_p_bounds.update({(gen,sn) : (var_lower[gen][sn],var_upper[gen][sn])
                             for gen in fixed_gens_i
                             for sn in snapshots})

        for bound in gen_p_bounds:
            if gen_p_bounds[bound][1] == np.inf:
                gen_p_bounds[bound] = (gen_p_bounds[bound][0],None)

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



    ## Define committable generator statuses ##

    network.model.generator_status = Var(list(fixed_committable_gens_i), snapshots,
                                         within=Binary)

    var_lower = p_min_pu.loc[:,fixed_committable_gens_i].multiply(network.generators.loc[fixed_committable_gens_i, 'p_nom'])
    var_upper = p_max_pu.loc[:,fixed_committable_gens_i].multiply(network.generators.loc[fixed_committable_gens_i, 'p_nom'])


    committable_gen_p_lower = {(gen,sn) : LConstraint(LExpression([(var_lower[gen][sn],network.model.generator_status[gen,sn]),(-1.,network.model.generator_p[gen,sn])]),"<=") for gen in fixed_committable_gens_i for sn in snapshots}

    l_constraint(network.model, "committable_gen_p_lower", committable_gen_p_lower,
                 list(fixed_committable_gens_i), snapshots)


    committable_gen_p_upper = {(gen,sn) : LConstraint(LExpression([(var_upper[gen][sn],network.model.generator_status[gen,sn]),(-1.,network.model.generator_p[gen,sn])]),">=") for gen in fixed_committable_gens_i for sn in snapshots}

    l_constraint(network.model, "committable_gen_p_upper", committable_gen_p_upper,
                 list(fixed_committable_gens_i), snapshots)


    ## Deal with minimum up time ##

    up_time_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"min_up_time"] > 0]

    for gen_i, gen in enumerate(up_time_gens):

        min_up_time = network.generators.loc[gen,"min_up_time"]
        initial_status = network.generators.loc[gen,"initial_status"]

        blocks = max(1,len(snapshots)-min_up_time+1)

        gen_up_time = {}

        for i in range(blocks):
            lhs = LExpression([(1,network.model.generator_status[gen,snapshots[j]]) for j in range(i,i+min_up_time)])

            if i == 0:
                rhs = LExpression([(min_up_time,network.model.generator_status[gen,snapshots[i]])],-min_up_time*initial_status)
            else:
                rhs = LExpression([(min_up_time,network.model.generator_status[gen,snapshots[i]]),(-min_up_time,network.model.generator_status[gen,snapshots[i-1]])])

            gen_up_time[i] = LConstraint(lhs,">=",rhs)

        l_constraint(network.model, "gen_up_time_{}".format(gen_i), gen_up_time,
                     range(blocks))



    ## Deal with minimum down time ##

    down_time_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"min_down_time"] > 0]

    for gen_i, gen in enumerate(down_time_gens):

        min_down_time = network.generators.loc[gen,"min_down_time"]
        initial_status = network.generators.loc[gen,"initial_status"]

        blocks = max(1,len(snapshots)-min_down_time+1)

        gen_down_time = {}

        for i in range(blocks):
            #sum of 1-status
            lhs = LExpression([(-1,network.model.generator_status[gen,snapshots[j]]) for j in range(i,i+min_down_time)],min_down_time)

            if i == 0:
                rhs = LExpression([(-min_down_time,network.model.generator_status[gen,snapshots[i]])],min_down_time*initial_status)
            else:
                rhs = LExpression([(-min_down_time,network.model.generator_status[gen,snapshots[i]]),(min_down_time,network.model.generator_status[gen,snapshots[i-1]])])

            gen_down_time[i] = LConstraint(lhs,">=",rhs)

        l_constraint(network.model, "gen_down_time_{}".format(gen_i), gen_down_time,
                     range(blocks))

    ## Deal with start up costs ##

    suc_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"start_up_cost"] > 0]

    network.model.generator_start_up_cost = Var(list(suc_gens),snapshots,
                                                domain=NonNegativeReals)

    sucs = {}

    for gen in suc_gens:
        suc = network.generators.loc[gen,"start_up_cost"]
        initial_status = network.generators.loc[gen,"initial_status"]

        for i,sn in enumerate(snapshots):

            if i == 0:
                rhs = LExpression([(suc, network.model.generator_status[gen,sn])],-suc*initial_status)
            else:
                rhs = LExpression([(suc, network.model.generator_status[gen,sn]),(-suc,network.model.generator_status[gen,snapshots[i-1]])])

            lhs = LExpression([(1,network.model.generator_start_up_cost[gen,sn])])

            sucs[gen,sn] = LConstraint(lhs,">=",rhs)

    l_constraint(network.model, "generator_start_up", sucs, list(suc_gens), snapshots)



    ## Deal with shut down costs ##

    sdc_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"shut_down_cost"] > 0]

    network.model.generator_shut_down_cost = Var(list(sdc_gens),snapshots,
                                                domain=NonNegativeReals)

    sdcs = {}

    for gen in sdc_gens:
        sdc = network.generators.loc[gen,"shut_down_cost"]
        initial_status = network.generators.loc[gen,"initial_status"]

        for i,sn in enumerate(snapshots):

            if i == 0:
                rhs = LExpression([(-sdc, network.model.generator_status[gen,sn])],sdc*initial_status)
            else:
                rhs = LExpression([(-sdc, network.model.generator_status[gen,sn]),(sdc,network.model.generator_status[gen,snapshots[i-1]])])

            lhs = LExpression([(1,network.model.generator_shut_down_cost[gen,sn])])

            sdcs[gen,sn] = LConstraint(lhs,">=",rhs)

    l_constraint(network.model, "generator_shut_down", sdcs, list(sdc_gens), snapshots)



    ## Deal with ramp limits without unit commitment ##

    sns = snapshots[1:]

    ru_gens = network.generators.index[~network.generators.ramp_limit_up.isnull()]

    ru = {}

    for gen in ru_gens:
        for i,sn in enumerate(sns):
            if network.generators.at[gen, "p_nom_extendable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]), (-1, network.model.generator_p[gen,snapshots[i]]), (-network.generators.at[gen, "ramp_limit_up"], network.model.generator_p_nom[gen])])
            elif not network.generators.at[gen, "committable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]), (-1, network.model.generator_p[gen,snapshots[i]])], -network.generators.at[gen, "ramp_limit_up"]*network.generators.at[gen, "p_nom"])
            else:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]), (-1, network.model.generator_p[gen,snapshots[i]]), ((network.generators.at[gen, "ramp_limit_start_up"] - network.generators.at[gen, "ramp_limit_up"])*network.generators.at[gen, "p_nom"], network.model.generator_status[gen,snapshots[i]]), (-network.generators.at[gen, "ramp_limit_start_up"]*network.generators.at[gen, "p_nom"], network.model.generator_status[gen,sn])])

            ru[gen,sn] = LConstraint(lhs,"<=")

    l_constraint(network.model, "ramp_up", ru, list(ru_gens), sns)



    rd_gens = network.generators.index[~network.generators.ramp_limit_down.isnull()]

    rd = {}


    for gen in rd_gens:
        for i,sn in enumerate(sns):
            if network.generators.at[gen, "p_nom_extendable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]), (-1, network.model.generator_p[gen,snapshots[i]]), (network.generators.at[gen, "ramp_limit_down"], network.model.generator_p_nom[gen])])
            elif not network.generators.at[gen, "committable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]), (-1, network.model.generator_p[gen,snapshots[i]])], network.generators.loc[gen, "ramp_limit_down"]*network.generators.at[gen, "p_nom"])
            else:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]), (-1, network.model.generator_p[gen,snapshots[i]]), ((network.generators.at[gen, "ramp_limit_down"] - network.generators.at[gen, "ramp_limit_shut_down"])*network.generators.at[gen, "p_nom"], network.model.generator_status[gen,sn]), (network.generators.at[gen, "ramp_limit_shut_down"]*network.generators.at[gen, "p_nom"], network.model.generator_status[gen,snapshots[i]])])

            rd[gen,sn] = LConstraint(lhs,">=")

    l_constraint(network.model, "ramp_down", rd, list(rd_gens), sns)





def define_storage_variables_constraints(network,snapshots,OGEM_options):

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

    if OGEM_options["reference"] != "expansion":
        upper = {(su,sn) : [[(1,model.state_of_charge[su,sn]),
                             (-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
                 for su in ext_sus_i for sn in snapshots}
        upper.update({(su,sn) : [[(1,model.state_of_charge[su,sn])],"<=",
                                 sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
                      for su in fix_sus_i for sn in snapshots})

        l_constraint(model, "state_of_charge_upper", upper,
                     list(network.storage_units.index), snapshots)

        # Storage units dispatch and store in expansion cases is fixed, so SOC constraints are not necessary.

        #this builds the constraint previous_soc + p_store - p_dispatch + inflow - spill == soc
        #it is complicated by the fact that sometimes previous_soc and soc are floats, not variables
        soc = {}

        #store the combinations with a fixed soc
        fixed_soc = {}

        state_of_charge_set = get_switchable_as_dense(network, 'StorageUnit', 'state_of_charge_set', snapshots)

        # TODO Resolve application of snapshot_weightings to store and dispatch

        for su in sus.index:
            for g,sn_group in snapshots.groupby(network.scenarios).items():
                for i,sn in enumerate(sn_group):
                    soc[su,sn] =  [[],"==",0.]
                    #TODO Resolve application of snapshot_weightings to store and dispatch
                    elapsed_hours = network.snapshot_weightings[sn] * len(network.snapshots) / network.snapshot_weightings.sum()

                    if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
                        previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
                        soc[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours
                                          * previous_state_of_charge)
                    else:
                        previous_state_of_charge = model.state_of_charge[su,sn_group[i-1]]
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

def define_energy_constraints(network, snapshots):
    """ Used only for base, linear case.
        Limits the total generation of inflow generators.
    """

    inflow_generators = network.generators[network.generators.inflow > 0]

    model = network.model
    #TODO Possibly keep constrain in MIP as new cut
    _energy_constraint = {}

    for gen in inflow_generators.index:
        _energy_constraint[gen] = [[], "<=", inflow_generators.loc[gen, "inflow"]]
        for sn in snapshots:
            _energy_constraint[gen][0].append((1.0, model.generator_p[gen, sn]))

    l_constraint(network.model, "energy_constraint", _energy_constraint, list(inflow_generators.index))

def define_branch_extension_variables(network,snapshots):
    """ Creates the investment ratio and binary investment variables for branches.
        e.g. the transmission capacity of a passive branch is pb_inv_ratio * s_nom_max, with pb_inv_ratio <= pb_bin_inv and pb_bin_inv a binary variable.
        Converters investment uses continuous variables.
    """
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
    """ Flows for extendable integer branches is limited by the investment ratio x maximum nominal capacity.
        AC/DC converters are modelled as conventional continuous-investment links.
    """

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


def define_passive_branch_flows(network,snapshots,formulation="angles",ptdf_tolerance=0.):

    if formulation == "angles":
        define_passive_branch_flows_with_angles(network, snapshots)

def define_passive_branch_flows_with_angles(network, snapshots):
    """ Non-extendable passive branches flow is unaltered.
        Extendable passive branches uses disjunctive flow constraints with a Big M parameter.
        flow + bin_inv * big_M <= delta theta / x + big_M
        flow - bin_inv * big_M >= delta theta / x - big_M
        The big_M is optimized as the minimum between a standard and the minimum existing path of voltage angle/value deltas between the branch nodes +5%, as in S. Binato, M.V.F. Pereira, and S. Granville A New Benders Decomposition Approach to Solve Power Transmission Network Design Problems IEEE Transactions on Power Systems 16(2): 235-240, May 2001 10.1109/59.918292
    """
    from .components import passive_branch_components

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

    if len(extendable_branches_i) > 0:

        # Calculation of minimum path of voltage angle/value delta for connected nodes in base network.
        base_network = network.copy(with_time=False)
        for pb, branch in base_network.lines.iterrows():
            if branch['s_nom'] == 0:
                base_network.remove('Line', pb)

        graph = base_network.graph(passive_branch_components) # Controllable branches do not count to calculate the minimum path.
        graph.remove_nodes_from(nx.isolates(graph))
        delta_theta = base_network.lines.loc[:,'s_nom_max'].multiply([branch["r_pu" if branch['branch_type'] == "DC" else "x_pu"] for b, branch in base_network.lines.iterrows()], axis=0)
        theta = {edge: delta_theta.loc[edge[2][1]]
                  for edge in graph.edges(keys=True) if edge[2][1] in base_network.lines.index}
        nx.set_edge_attributes(graph, 'theta', theta)
        min_theta = nx.shortest_path_length(graph,weight = 'theta')

        # Construction of disjunctive flow constraints.
        for branch in extendable_branches_i:
            bus0 = passive_branches.bus0[branch]
            bus1 = passive_branches.bus1[branch]
            bt = branch[0]
            bn = branch[1]

            sub = passive_branches.sub_network[branch]
            attribute = "r_pu" if network.sub_networks.at[sub,"carrier"] == "DC" else "x_pu"
            y = 1/(passive_branches.at[branch,attribute]*(passive_branches.at[branch,"tap_ratio"] if bt == "Transformer" else 1.))

            big_M = 1e2 # Standard big_M value. Increase if the KVL duals check accuses a binding flow constraint for a not-invested branch.
            # Big M optimization.
            if bus0 in min_theta.keys():
                if bus1 in min_theta[bus0].keys():
                    big_M = min(min_theta[bus0][bus1] * y * 1.05,big_M)

            # TODO graph shortest path to determine big_M by bus pair

            for sn in snapshots:
                lhs = LExpression([(1,network.model.passive_branch_p[bt,bn,sn]),(-y,network.model.voltage_angles[bus0,sn]),
                                   (y,network.model.voltage_angles[bus1,sn]),
                                   (big_M,network.model.pb_bin_inv[bt,bn])])
                upper_flows[bt,bn,sn] = LConstraint(lhs,"<=",LExpression(constant=big_M))

                lhs = LExpression([(1,network.model.passive_branch_p[bt,bn,sn]),(-y,network.model.voltage_angles[bus0,sn]),
                                   (y,network.model.voltage_angles[bus1,sn]),
                                   (-big_M,network.model.pb_bin_inv[bt,bn])])
                lower_flows[bt,bn,sn] = LConstraint(lhs,">=",LExpression(constant=-big_M))

        l_constraint(network.model, "passive_extendable_branch_p_lower", lower_flows,
                     list(extendable_branches_i), snapshots)

        l_constraint(network.model, "passive_extendable_branch_p_upper", upper_flows,
                     list(extendable_branches_i), snapshots)

def define_passive_branch_constraints(network, snapshots):

    passive_branches = network.passive_branches()
    extendable_branches_i = passive_branches[passive_branches.s_nom_extendable].index
    fixed_branches_i = passive_branches[~ passive_branches.s_nom_extendable].index

    def pb_inv_ratio_rule(model, branch_type, branch_name):
        return (model.pb_inv_ratio[branch_type, branch_name] - model.pb_bin_inv[branch_type, branch_name] <= 0)

    # Force the investment ratio = 0 when the binary investment = 0
    network.model.pb_inv_ratio_constraint = Constraint(list(extendable_branches_i),rule=pb_inv_ratio_rule)

    flow_upper = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    "<=", passive_branches.s_nom[b]]
                  for b in fixed_branches_i
                  for sn in snapshots}

    l_constraint(network.model, "flow_upper", flow_upper,
                 list(fixed_branches_i), snapshots)

    flow_lower = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    ">=", -passive_branches.s_nom[b]]
                  for b in fixed_branches_i
                  for sn in snapshots}

    l_constraint(network.model, "flow_lower", flow_lower,
                 list(fixed_branches_i), snapshots)

    # For extendable branches the flow is not limited by s_nom_max anymore, but s_nom_max x investment ratio.
    integer_flow_upper = ({(b[0], b[1], sn): [[(1, network.model.passive_branch_p[b[0], b[1], sn]),
                                           (-passive_branches.at[(b[0], b[1]),"s_nom_max"], network.model.pb_inv_ratio[b[0],b[1]])], "<=", 0]
                       for b in extendable_branches_i
                       for sn in snapshots})

    l_constraint(network.model, "integer_flow_upper", integer_flow_upper,
                 list(extendable_branches_i), snapshots)

    integer_flow_lower = ({(b[0], b[1], sn): [[(1, network.model.passive_branch_p[b[0], b[1], sn]),
                                           (passive_branches.at[(b[0], b[1]), "s_nom_max"], network.model.pb_inv_ratio[b[0],b[1]])], ">=", 0]
                       for b in extendable_branches_i
                       for sn in snapshots})

    l_constraint(network.model, "integer_flow_lower", integer_flow_lower,
                 list(extendable_branches_i), snapshots)

def define_controllable_branch_constraints(network, snapshots):
    """ Force the investment ratio = 0 when the binary investment = 0 """

    extendable_ptp_links_i = network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]

    def cb_inv_ratio_rule(model, branch_name):
        return (model.cb_inv_ratio[branch_name] - model.cb_bin_inv[branch_name] <= 0)

    network.model.cb_inv_ratio_constraint = Constraint(list(extendable_ptp_links_i),rule=cb_inv_ratio_rule)

def define_minimum_ratio(network):
    """ Branches may be split into capacity ranges to better represent resistance and reactances for each range.
        This defines a disjunctive lower bound for the investment ratio to respect the s_nom_min/p_nom_min applicable only to branches invested in.
    """
    extendable_ptp_links = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]
    links_with_minimum_i = extendable_ptp_links[extendable_ptp_links.p_nom_min > 0].index

    big_M = -1

    def cb_minimum_ratio_rule(model, branch_name):
        return (model.cb_inv_ratio[branch_name] - big_M * (1 - model.cb_bin_inv[branch_name]) >= network.links.at[branch_name,"p_nom_min"]/network.links.at[branch_name,"p_nom_max"])

    network.model.cb_minimum_ratio = Constraint(list(links_with_minimum_i),rule=cb_minimum_ratio_rule)

    passive_branches = network.passive_branches()
    branches_with_minimum_i = passive_branches[(passive_branches.s_nom_min > 0)&(passive_branches.s_nom_extendable)].index

    def pb_minimum_ratio_rule(model, branch_type, branch_name):
        return (model.pb_inv_ratio[branch_type, branch_name] - big_M * (1 - model.pb_bin_inv[branch_type, branch_name]) >= passive_branches.loc[(branch_type,branch_name),"s_nom_min"]/passive_branches.loc[(branch_type,branch_name),"s_nom_max"])

    network.model.pb_minimum_ratio = Constraint(list(branches_with_minimum_i),rule=pb_minimum_ratio_rule)

def define_parallelism_constraints(network,snapshots):
    """ For node pairs of parallel branches of the same technology but different capacity ranges allow only one branch. """
    #TODO Include as special ordered set to speed up solution
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
        bus = network.generators.at[gen,"bus"]
        sign = network.generators.at[gen,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.generator_p[gen,sn]))

    load_p_set = get_switchable_as_dense(network, 'Load', 'p_set')
    for load in network.loads.index:
        bus = network.loads.at[load,"bus"]
        sign = network.loads.at[load,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].constant += sign*load_p_set.at[sn,load]

    for su in network.storage_units.index:
        bus = network.storage_units.at[su,"bus"]
        sign = network.storage_units.at[su,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.storage_p_dispatch[su,sn]))
            network._p_balance[bus,sn].variables.append((-sign,network.model.storage_p_store[su,sn]))

    for store in network.stores.index:
        bus = network.stores.at[store,"bus"]
        sign = network.stores.at[store,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.store_p[store,sn]))


def define_nodal_balance_constraints(network,snapshots):

    passive_branches = network.passive_branches()


    for branch in passive_branches.index:
        bus0 = passive_branches.at[branch,"bus0"]
        bus1 = passive_branches.at[branch,"bus1"]
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

def define_participation_variables(network, snapshots, base_welfare):
    """ country_participation variables are 1 if the country builds any cooperative branch, and 0 otherwise.
        Used for the Pareto welfare constraint and for cooperation analysis in general.
    """
    cooperative_ptp_links = network.links[network.links.p_nom_extendable & network.links.cooperative & (network.links.branch_type == "ptp")]
    passive_branches = network.passive_branches()
    cooperative_passive_branches = passive_branches[passive_branches.s_nom_extendable & passive_branches.cooperative]
    max_branches = len(network.model.cb_bin_inv) + len(network.model.pb_bin_inv)
    countries = base_welfare[base_welfare!=0].index

    _country_participation_upper = {(country) : LExpression()
                          for country in countries}

    for cb,branch in cooperative_ptp_links.iterrows():
        bus0 = network.buses.loc[branch["bus0"]]
        bus1 = network.buses.loc[branch["bus1"]]
        if (bus0["country"] in countries):
            _country_participation_upper[bus0["country"]].variables.append((-1,network.model.cb_bin_inv[cb]))
        if (bus1["country"] in countries):
            if bus1["country"] != bus0["country"]:
                _country_participation_upper[bus1["country"]].variables.append((-1,network.model.cb_bin_inv[cb]))

    for pb,branch in cooperative_passive_branches.iterrows():
        bus0 = network.buses.loc[branch["bus0"]]
        bus1 = network.buses.loc[branch["bus1"]]
        if (bus0["country"] in countries):
            _country_participation_upper[bus0["country"]].variables.append((-1,network.model.pb_bin_inv[pb]))
        if (bus1["country"] in countries):
            if bus1["country"] != bus0["country"]:
                _country_participation_upper[bus1["country"]].variables.append((-1,network.model.pb_bin_inv[pb]))

    # Although the _country_participation_upper and _country_participation_lower are exactly the same, deep copying the former after build is not possible, so they are built separately.

    _country_participation_lower = {(country) : LExpression()
                          for country in countries}

    for cb,branch in cooperative_ptp_links.iterrows():
        bus0 = network.buses.loc[branch["bus0"]]
        bus1 = network.buses.loc[branch["bus1"]]
        if (bus0["country"] in countries):
            _country_participation_lower[bus0["country"]].variables.append((-1,network.model.cb_bin_inv[cb]))
        if (bus1["country"] in countries):
            _country_participation_lower[bus1["country"]].variables.append((-1,network.model.cb_bin_inv[cb]))

    for pb,branch in cooperative_passive_branches.iterrows():
        bus0 = network.buses.loc[branch["bus0"]]
        bus1 = network.buses.loc[branch["bus1"]]
        if (bus0["country"] in countries):
            _country_participation_lower[bus0["country"]].variables.append((-1,network.model.pb_bin_inv[pb]))
        if (bus1["country"] in countries):
            _country_participation_lower[bus1["country"]].variables.append((-1,network.model.pb_bin_inv[pb]))

    countries = list([k for k,v in _country_participation_upper.items() if v.variables != LExpression().variables])

    network.model.country_participation = Var(countries,domain = Boolean)

    _country_participation_upper = {k: v for k, v in iteritems(_country_participation_upper) if k in countries}
    _country_participation_lower = {k: v for k, v in iteritems(_country_participation_lower) if k in countries}

    for k,v in _country_participation_upper.items():
        v.variables.append((1,network.model.country_participation[k]))

    for k,v in _country_participation_lower.items():
        v.variables.append((max_branches, network.model.country_participation[k]))

    country_participation_upper = {k: LConstraint(v, "<=", LExpression()) for k, v in iteritems(_country_participation_upper)}
    country_participation_lower = {k: LConstraint(v, ">=", LExpression()) for k, v in iteritems(_country_participation_lower)}

    l_constraint(network.model, "country_participation_upper", country_participation_upper,countries)
    l_constraint(network.model, "country_participation_lower", country_participation_lower,countries)

def define_objective(network, snapshots, low_totex_factor, high_totex_factor):

    model = network.model

    extendable_generators = network.generators[network.generators.p_nom_extendable]

    ext_sus = network.storage_units[network.storage_units.p_nom_extendable]

    ext_stores = network.stores[network.stores.e_nom_extendable]

    passive_branches = network.passive_branches()

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    extendable_ptp_links = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]

    extendable_converters = network.links[network.links.p_nom_extendable & (network.links.branch_type == "converter")]

    # The capitals cost of extendable generators is a Pyomo parameter.
    # Allows to set them to zero when fixing and re-running the MIP to obtain short-run instead of long-run marginal prices.
    # def gen_capital_cost_rule(model,gen):
    #     return extendable_generators.at[gen,"capital_cost"]

    objective = LExpression()

    investment_objective = LExpression()

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

        # Add a transmission cost to the objective proportional to absolute flows to prioritize local generation when marginal costs are equal.
        for link in network.links.index:
            coefficient = network.links.at[link, "marginal_cost"] * weight
            objective.variables.extend([(coefficient, model.abs_link_p[link,sn])])

        for bt,bn in passive_branches.index:
            coefficient = passive_branches.at[(bt,bn), "marginal_cost"] * weight
            objective.variables.extend([(coefficient, model.abs_passive_branch_p[bt,bn,sn])])


    #NB: for capital costs we subtract the costs of existing infrastructure p_nom/s_nom
    #TODO include generation totex factor for other techs if needed

    investment_objective.variables.extend([(ext_sus.at[su,"capital_cost"], model.storage_p_nom[su])
                                for su in ext_sus.index])
    investment_objective.constant -= (ext_sus.capital_cost * ext_sus.p_nom).sum()

    investment_objective.variables.extend([(ext_stores.at[store,"capital_cost"], model.store_e_nom[store])
                                for store in ext_stores.index])
    investment_objective.constant -= (ext_stores.capital_cost * ext_stores.e_nom).sum()

    investment_objective.variables.extend([(extendable_passive_branches.at[b, "capital_cost"] * low_totex_factor * extendable_passive_branches.at[b, "s_nom_max"], model.pb_inv_ratio[b]) for b in extendable_passive_branches.index])
    investment_objective.constant -= (extendable_passive_branches.capital_cost * low_totex_factor * extendable_passive_branches.s_nom).sum()

    investment_objective.variables.extend([(extendable_ptp_links.at[b, "capital_cost"] * low_totex_factor * extendable_ptp_links.at[b, "p_nom_max"], model.cb_inv_ratio[b]) for b in extendable_ptp_links.index])
    investment_objective.constant -= (extendable_ptp_links.capital_cost * low_totex_factor * extendable_ptp_links.p_nom).sum()

    investment_objective.variables.extend([(extendable_converters.at[b,"capital_cost"] * low_totex_factor, model.conv_p_nom[b])
                                for b in extendable_converters.index])
    investment_objective.constant -= (extendable_converters.capital_cost * low_totex_factor * extendable_converters.p_nom).sum()

    investment_objective.variables.extend([(extendable_generators.at[gen,'capital_cost'] * high_totex_factor,model.generator_p_nom[gen])
                                            for gen in extendable_generators.index])
    investment_objective.constant -= (extendable_generators.capital_cost * high_totex_factor * extendable_generators.p_nom).sum()

    objective = objective + investment_objective

    l_objective(model,objective)

    def l_expression(model, expression):

        # initialise with a dummy
        model.investment_objective = Expression(expr=0.)

        model.investment_objective._expr = pyomo.core.base.expr_coopr3._SumExpression()
        model.investment_objective._expr._args = [item[1] for item in expression.variables]
        model.investment_objective._expr._coef = [item[0] for item in expression.variables]
        model.investment_objective._expr._const = expression.constant

    l_expression(model,investment_objective)
    
    #TODO include storage capital costs if there is endogenous storage investment

def define_absolute_flows(network, snapshots):
    """ Create variables of absolute flows of branches.
        These are used to add a penalty to the objective function to prioritize local generation when marginal costs are equal.
    """

    network.model.abs_link_p = Var(list(network.links.index),
                               snapshots, domain=NonNegativeReals)

    def abs_link_positive(model, link, sn):
        return model.abs_link_p[link, sn] - model.link_p[link, sn] >= 0

    def abs_link_negative(model, link, sn):
        return model.abs_link_p[link, sn] + model.link_p[link, sn] >= 0

    network.model.abs_link_positive = Constraint(list(network.model.abs_link_p._index),rule=abs_link_positive)

    network.model.abs_link_negative = Constraint(list(network.model.abs_link_p._index),rule=abs_link_negative)

    passive_branches = network.passive_branches()

    network.model.abs_passive_branch_p = Var(list(passive_branches.index), snapshots, domain=NonNegativeReals)

    def abs_passive_positive(model, bt, bn, sn):
        return model.abs_passive_branch_p[bt, bn, sn] - model.passive_branch_p[bt, bn, sn] >= 0

    def abs_passive_negative(model, bt, bn, sn):
        return model.abs_passive_branch_p[bt, bn, sn] + model.passive_branch_p[bt, bn, sn] >= 0

    network.model.abs_passive_positive = Constraint(list(network.model.abs_passive_branch_p._index),rule=abs_passive_positive)

    network.model.abs_passive_negative = Constraint(list(network.model.abs_passive_branch_p._index),rule=abs_passive_negative)

def define_pareto_welfare(network, snapshots, base_welfare, low_totex_factor, high_totex_factor, pareto_tolerance):
    """ Pareto welfare constraints for each country.
        A country welfare must increase compared to the base case.

        National welfare change =
        - consumer payments change
        + producer surplus change (including storage cost)
        + congestion rent change (for lines, ptp links and converters)
        - generation investment cost
        - transmission investment cost
        for all nodes of each country.

        Values for cross-border components (lines and ptp links) are split equally between both nodes.
        The disjunctive constraint applies only when country_participation = 1.
     """

    model = network.model

    passive_branches = network.passive_branches()
    ptp_links = network.links[network.links.branch_type == "ptp"]
    converters = network.links[network.links.branch_type == "converter"]
    weights = network.snapshot_weightings
    hours = weights.sum()

    # Prices and margins are set as parameters to update them in iterations without rebuilding the model.
    model.marginal_prices = Param(list(model.power_balance.keys()), default = 0, mutable=True)
    model.gen_margins = Param(list(network.generators.index), snapshots, default = 0, mutable=True)
    model.su_margins = Param(list(network.storage_units.index), snapshots, default = 0, mutable=True)
    model.cb_margins = Param(list(network.links.index), snapshots, default = 0, mutable=True)
    model.pb_margins = Param(list(network.lines.index), snapshots, default = 0, mutable=True)

    countries = list(model.country_participation._index)

    # Creates the Pareto welfare big M parameter. The actual value is set in the main Pareto welfare loop.
    model.welfare_big_M = Param(countries,mutable=True,default=0)

    """
    Implements a variable for each welfare components with a constraint variable = expression
    + consumer payments
    + producer surplus
    + storage unit surplus
    + passive branch congestion rent
    + point-to-point link congestion rent
    + converter congestion rent
    - investment costs
    + big M * (1 - country participation)
    - base welfare
    >= 0
     """

    def CP_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_loads = network.loads[network.loads["bus"].isin(country_buses)].index
        country_gens = network.generators[network.generators["bus"].isin(country_buses)]
        country_sus = network.storage_units[network.storage_units["bus"].isin(country_buses)]
        country_lines0 = passive_branches[passive_branches["bus0"].isin(country_buses)]
        country_lines1 = passive_branches[passive_branches["bus1"].isin(country_buses)]
        country_links0 = ptp_links[ptp_links["bus0"].isin(country_buses)]
        country_links1 = ptp_links[ptp_links["bus1"].isin(country_buses)]
        country_convs = converters[converters["bus0"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return sum(network.loads_t.p.loc[sn, load] * model.marginal_prices[(load,sn)] for sn in snapshots for load in country_loads)

    def CP_constraint(model, country):
        return  model.consumer_payments_e[country] - model.consumer_payments[country] == 0

    model.consumer_payments_e = Expression(countries, rule=CP_rule)

    model.consumer_payments = Var(countries, domain = Reals, initialize = CP_rule)

    model.consumer_payments_c = Constraint(countries, rule=CP_constraint)



    def PS_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_gens = network.generators[network.generators["bus"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return sum(model.generator_p[(g, sn)] * model.gen_margins[(g,sn)] for g,gen in country_gens.iterrows() for sn in snapshots)

    def PS_constraint(model, country):
        return  model.producer_surplus_e[country] - model.producer_surplus[country] == 0

    model.producer_surplus = Var(countries, domain = Reals, initialize = PS_rule)

    model.producer_surplus_e = Expression(countries, rule=PS_rule)

    model.producer_surplus_c = Constraint(countries, rule=PS_constraint)
    
    
    
    def SS_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_sus = network.storage_units[network.storage_units["bus"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return sum(model.storage_p_dispatch[(s, sn)] * model.su_margins[(s,sn)] for s,su in country_sus.iterrows() for sn in snapshots) + sum(model.storage_p_store[(s, sn)] * model.marginal_prices[(su['bus'], sn)] for s,su in country_sus.iterrows() for sn in snapshots)

    def SS_constraint(model, country):
        return  model.storage_surplus_e[country] - model.storage_surplus[country] == 0

    model.storage_surplus_e = Expression(countries, rule=SS_rule)

    model.storage_surplus = Var(countries, domain = Reals, initialize = SS_rule)

    model.storage_surplus_c = Constraint(countries, rule=SS_constraint)
    
    
    
    def PBR_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_lines0 = passive_branches[passive_branches["bus0"].isin(country_buses)]
        country_lines1 = passive_branches[passive_branches["bus1"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return sum(network.model.passive_branch_p[bt,bn,sn] * model.pb_margins[(bn,sn)] / 2 for (bt,bn), branch in country_lines0.iterrows() for sn in snapshots) + sum(network.model.passive_branch_p[bt,bn,sn] * model.pb_margins[(bn,sn)] / 2 for (bt,bn), branch in country_lines1.iterrows() for sn in snapshots)

    def PBR_constraint(model, country):
        return  model.pb_rent_e[country] - model.pb_rent[country] == 0

    model.pb_rent_e = Expression(countries, rule=PBR_rule)

    model.pb_rent = Var(countries, domain = Reals, initialize = PBR_rule)

    model.pb_rent_c = Constraint(countries, rule=PBR_constraint)


    def PTPR_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_links0 = ptp_links[ptp_links["bus0"].isin(country_buses)]
        country_links1 = ptp_links[ptp_links["bus1"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return sum(network.model.link_p[l,sn] * model.cb_margins[(l,sn)] / 2 for l, link in country_links0.iterrows() for sn in snapshots) + sum(network.model.link_p[l,sn] * model.cb_margins[(l,sn)] / 2 for l, link in country_links1.iterrows() for sn in snapshots)

    def PTPR_constraint(model, country):
        return  model.ptp_rent_e[country] - model.ptp_rent[country] == 0

    model.ptp_rent_e = Expression(countries, rule=PTPR_rule)

    model.ptp_rent = Var(countries, domain = Reals, initialize = PTPR_rule)

    model.ptp_rent_c = Constraint(countries, rule=PTPR_constraint)



    def CR_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_convs = converters[converters["bus0"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return sum(network.model.link_p[c,sn] * model.cb_margins[(c,sn)] for c, conv in country_convs.iterrows() for sn in snapshots)

    def CR_constraint(model, country):
        return  model.conv_rent_e[country] - model.conv_rent[country] == 0

    model.conv_rent_e = Expression(countries, rule=CR_rule)

    model.conv_rent = Var(countries, domain = Reals, initialize = CR_rule)

    model.conv_rent_c = Constraint(countries, rule=CR_constraint)


    def IC_rule(model, country):
        # Define all components belonging to the country.
        country_buses = network.buses[network.buses["country"] == country].index
        country_gens = network.generators[network.generators["bus"].isin(country_buses)]
        country_sus = network.storage_units[network.storage_units["bus"].isin(country_buses)]
        country_lines0 = passive_branches[passive_branches["bus0"].isin(country_buses)]
        country_lines1 = passive_branches[passive_branches["bus1"].isin(country_buses)]
        country_links0 = ptp_links[ptp_links["bus0"].isin(country_buses)]
        country_links1 = ptp_links[ptp_links["bus1"].isin(country_buses)]
        country_convs = converters[converters["bus0"].isin(country_buses)]

        # Build expressions for each welfare change component.
        return  + sum(gen["capital_cost"] * high_totex_factor * (network.model.generator_p_nom[g] - gen['p_nom']) for g, gen in country_gens[country_gens.p_nom_extendable].iterrows())\
                + sum(branch["capital_cost"] * branch["s_nom_max"] * network.model.pb_inv_ratio[b] * low_totex_factor / 2 for b, branch in country_lines0[country_lines0.s_nom_extendable].iterrows())\
                + sum(branch["capital_cost"] * branch["s_nom_max"] * network.model.pb_inv_ratio[b] * low_totex_factor / 2 for b, branch in country_lines1[country_lines1.s_nom_extendable].iterrows())\
                + sum(link["capital_cost"] * link["p_nom_max"] * network.model.cb_inv_ratio[l] * low_totex_factor / 2 for l, link in country_links0[country_links0.p_nom_extendable].iterrows())\
                + sum(link["capital_cost"] * link["p_nom_max"] * network.model.cb_inv_ratio[l] * low_totex_factor / 2 for l, link in country_links1[country_links1.p_nom_extendable].iterrows())\
                + sum(conv["capital_cost"] * low_totex_factor * (network.model.conv_p_nom[c] - conv['p_nom']) for c, conv in country_convs[country_convs.p_nom_extendable].iterrows())

    def IC_constraint(model, country):
        return  model.investment_cost_e[country] - model.investment_cost[country] == 0

    model.investment_cost_e = Expression(countries, rule=IC_rule)

    model.investment_cost = Var(countries, domain = Reals, initialize = IC_rule)

    model.investment_cost_c = Constraint(countries, rule=IC_constraint)
    
    
    def welfare_constant_rule(model, country):

        # Build expressions for each welfare change component.

        if base_welfare.loc[country] < 0:
            pareto_sign = - 1
        else:
            pareto_sign = + 1
        welfare_constant = base_welfare.loc[country] * hours * (1. - pareto_sign * pareto_tolerance) - model.welfare_big_M[country]
        return  welfare_constant

    def welfare_constant_constraint(model, country):
        return  model.welfare_constant_e[country] - model.welfare_constant[country] == 0

    model.welfare_constant_e = Expression(countries, rule=welfare_constant_rule)

    model.welfare_constant = Var(countries, domain = Reals, initialize = welfare_constant_rule)

    model.welfare_constant_c = Constraint(countries, rule=welfare_constant_constraint)

    def final_welfare(model,country):
        return - model.consumer_payments[country] + model.producer_surplus[country] + model.storage_surplus[country] + model.pb_rent[country] + model.ptp_rent[country] + model.conv_rent[country] - model.investment_cost[country] - model.welfare_big_M[country] * model.country_participation[country] - model.welfare_constant[country] >= 0

    model.pareto_welfare = Constraint(countries, rule=final_welfare)

def define_cooperation_constraint(network,cooperation_limit):
    """ Applicable when there is a cooperation limit.
        For each period, the number of branches invested in for each node <= cooperation_limit
    """
    cooperative_ptp_links = network.links[network.links.p_nom_extendable & network.links.cooperative & (network.links.branch_type == "ptp")]
    passive_branches = network.passive_branches()
    cooperative_passive_branches = passive_branches[passive_branches.s_nom_extendable & passive_branches.cooperative]
    offshore_buses_i = network.buses.loc[network.buses.terminal_type.isin(["owf","hub"]),'base_bus'].unique()

    _cooperation_constraint = {(bus) : LExpression()
                          for bus in offshore_buses_i}

    for cb,branch in cooperative_ptp_links.iterrows():
        base_bus0 = network.buses.loc[branch["bus0"],'base_bus']
        base_bus1 = network.buses.loc[branch["bus1"],'base_bus']
        if base_bus0 in offshore_buses_i:
            _cooperation_constraint[base_bus0].variables.append((1,network.model.cb_bin_inv[cb]))
        if base_bus1 in offshore_buses_i:
            _cooperation_constraint[base_bus1].variables.append((1,network.model.cb_bin_inv[cb]))

    for pb,branch in cooperative_passive_branches.iterrows():
        base_bus0 = network.buses.loc[branch["bus0"],'base_bus']
        base_bus1 = network.buses.loc[branch["bus1"],'base_bus']
        if base_bus0 in offshore_buses_i:
            _cooperation_constraint[base_bus0].variables.append((1,network.model.pb_bin_inv[pb]))
        if base_bus1 in offshore_buses_i:
            _cooperation_constraint[base_bus1].variables.append((1,network.model.pb_bin_inv[pb]))

    cooperation_constraint = {k: LConstraint(v, "<=", LExpression(constant=cooperation_limit)) for k, v in iteritems(_cooperation_constraint)}

    l_constraint(network.model, "cooperation_constraint", cooperation_constraint,list(offshore_buses_i))

def fix_constrained_units(network, snapshots):
    """ In expansion cases, fix dispatch and storage for generators and storage units with a p_set. """

    for s, su in network.storage_units.iterrows():
        for sn in snapshots:
            dispatch = network.storage_units_t.p_set.loc[sn, s]
            network.model.storage_p_dispatch[s, sn].fixed = True
            network.model.storage_p_store[s, sn].fixed = True
            network.model.storage_p_dispatch[s, sn].value = 0.0
            network.model.storage_p_store[s, sn].value = 0.0
            if dispatch > 0:
                network.model.storage_p_dispatch[s, sn].value = abs(dispatch)
            elif dispatch < 0:
                network.model.storage_p_store[s, sn].value = abs(dispatch)

    for g, gen in network.generators.loc[network.generators_t.p_set.columns].iterrows():
        for sn in snapshots:
            network.model.generator_p[g, sn].value = network.generators_t.p_set.loc[sn, g]
            network.model.generator_p[g, sn].fixed = True

def extract_optimisation_results(network, snapshots,reference,formulation="angles"):

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
    network.objective = value(network.model.objective)
    print("Objective:",network.objective)

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

    # Integer extendable branches actual s_nom/p_nom is defined by nominal max capacity x investment ratio
    passive_branches = network.passive_branches()
    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]
    pb_inv_ratio_series = as_series(model.pb_inv_ratio)

    if len(pb_inv_ratio_series):
        pb_inv_ratio_series.index.levels[1].name = extendable_passive_branches.index.levels[1].name
        s_nom_extendable_passive_branches = pb_inv_ratio_series * extendable_passive_branches.loc[:,"s_nom_max"]

    for c in network.iterate_components(passive_branch_components):
        c.df['s_nom_opt'] = c.df.s_nom
        if c.df.s_nom_extendable.any():
            c.df.loc[c.df.s_nom_extendable, 's_nom_opt'] = s_nom_extendable_passive_branches.loc[c.name]

    network.links.p_nom_opt = network.links.p_nom

    extendable_ptp_links_i = network.links[network.links.p_nom_extendable & (network.links.branch_type == "ptp")].index
    cb_inv_ratio_series = as_series(model.cb_inv_ratio)

    if len(cb_inv_ratio_series):
        network.links.loc[extendable_ptp_links_i,"p_nom_opt"] = cb_inv_ratio_series * network.links.loc[extendable_ptp_links_i, "p_nom_max"]

    # Converters p_nom is the actual p_nom value.
    extendable_converters_i = network.links[network.links.p_nom_extendable & (network.links.branch_type == "converter")].index
    network.links.loc[extendable_converters_i, "p_nom_opt"] = as_series(network.model.conv_p_nom)

    if network.co2_limit is not None:
        try:
            network.co2_price = - network.model.dual[network.model.co2_constraint]
        except (AttributeError, KeyError) as e:
            logger.warning("Could not read out co2_price, although a co2_limit was set")

    #extract unit commitment statuses
    if network.generators.committable.any():
        allocate_series_dataframes(network, {'Generator': ['status']})

        fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]

        if len(fixed_committable_gens_i) > 0:
            network.generators_t.status.loc[snapshots,fixed_committable_gens_i] = \
                as_series(model.generator_status).unstack(0)


def memory_usage(step):

    gc.collect()
    process = psutil.Process(os.getpid())
    print('{:s} memory usage: {:.2f}'.format(step,process.memory_info().rss / float(2 ** 20)))

def network_lopf(network, snapshots=None, solver_name="glpk", solver_io=None,
                 skip_pre=False, extra_functionality=None, solver_options={},
                 keep_files=False, formulation="angles", ptdf_tolerance=0.,
                 free_memory={}, OGEM_options=None, base_welfare=None,parameters=None):
    """
    Mixed-integer optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    solver_io : string, default None
        Solver Input-Output option, e.g. "python" to use "gurobipy" for
        solver_name="gurobi"
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
    OGEM_options : dictionary
        OGEM options indicating reference ('base' or 'expansion), and pareto_welfare and cooperation_limit parameters.
    base_welfare : dataframe
        Total benefit of base case aggregated per country.
    parameters : dataframe
        OGEM simulation parameters dataframe.
    Returns
    -------
    None
    """

    memory_usage('Start')
    print("MILP OPF build at {:%H:%M}...".format(datetime.datetime.now()))

    # Variables declaration
    warmstart = False

    # The snapshot scale may improve the numerical properties of the problem. Default = 1.
    if OGEM_options["reference"] != "expansion":
        snapshot_scale = parameters['MILP_snapshot_scale']
    else:
        snapshot_scale = parameters['snapshot_scale']

    # The low and high totex factor transform the CAPEX and OPEX (maintenance) costs to hourly values
    low_totex_factor = (np.pmt(parameters["discount_rate"], parameters["high_lifetime"], -1.) + parameters['low_OPEX_%']) / 8760 * snapshot_scale # Factor to calculate investment hourly payments for transmission.
    high_totex_factor = (np.pmt(parameters["discount_rate"], parameters["low_lifetime"], -1.) + parameters['high_OPEX_%']) / 8760 * snapshot_scale # Factor to calculate investment hourly payments for generation.

    if not skip_pre:
        network.determine_network_topology()
        calculate_dependent_values(network)
        for sub_network in network.sub_networks.obj:
            find_slack_bus(sub_network)
        logger.info("Performed preliminary steps")


    snapshots = _as_snapshots(network, snapshots)

    logger.info("Building pyomo model using `%s` formulation", formulation)
    network.model = ConcreteModel("Mixed Integer Optimal Investment & Operation")

    define_generator_variables_constraints(network,snapshots)

    define_storage_variables_constraints(network,snapshots,OGEM_options)

    # Full year base cases are energy constrained, while storage units and energy-constrained generators are fixed in expansion cases
    if OGEM_options["reference"] != "expansion":
        define_energy_constraints(network, snapshots)

    define_store_variables_constraints(network,snapshots)

    define_branch_extension_variables(network,snapshots)

    define_link_flows(network,snapshots)

    define_nodal_balances(network,snapshots)

    define_passive_branch_flows(network,snapshots,formulation,ptdf_tolerance)

    define_passive_branch_constraints(network,snapshots)

    define_controllable_branch_constraints(network, snapshots) # New constraint to limit force investment ratio to 0 when line is not invested in

    if formulation in ["angles"]:
        define_nodal_balance_constraints(network,snapshots)
    else:
        print('Other formulation than angles not supported')
        sys.exit()

    if network.co2_limit is not None:
        define_co2_constraint(network,snapshots)

    #force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    if extra_functionality is not None:
        extra_functionality(network,snapshots)

    define_absolute_flows(network, snapshots) # New constraint to obtain absolute values of flow variables

    define_objective(network, snapshots, low_totex_factor, high_totex_factor)

    if OGEM_options["reference"] == "expansion":
        define_parallelism_constraints(network,snapshots) # Constraint limiting investment of parallel branches to one per technology

        if OGEM_options["cooperation_limit"] is not False:
            define_cooperation_constraint(network,OGEM_options["cooperation_limit"]) # Add cooperation constraint if there is a cooperation limit

    #tidy up auxilliary expressions
    del network._p_balance

    logger.info("Solving model using %s", solver_name)
    opt = SolverFactory(solver_name, solver_io=solver_io)

    patch_optsolver_record_memusage_before_solving(opt, network)

    # Definition of various sub-functions
    def convergence_status():
        """ Analyze solver output for feasibility and optimility, and run one more time in case of error """

        status = network.results["Solver"][0]["Status"].key
        termination_condition = network.results["Solver"][0]["Termination condition"].key

        if status == "ok" and termination_condition == "optimal":
            logger.info("Optimization successful")
            extract_optimisation_results(network, snapshots,OGEM_options["reference"], formulation)
        else:
            logger.error("Optimisation failed with status %s and terminal condition %s. Resolving with symbolic labels."
                         % (status, termination_condition))

            # If there is an error re-run, keeping files and using symbolic variable names for debugging.
            solver_run(keepfiles=True, symbolic=True)

            extract_optimisation_results(network, snapshots,OGEM_options["reference"], formulation)

    def fix_variables(status=False):
        """ Fix integer variables to obtain marginal prices or unfix problems for run iterations in MILPs """

        # TODO include storage in variable fixing if necessary
        extendable_passive_branches_i = network.passive_branches()
        extendable_passive_branches_i = extendable_passive_branches_i[extendable_passive_branches_i.s_nom_extendable].index

        for epb in extendable_passive_branches_i:
            network.model.pb_bin_inv[epb].fixed = status
            network.model.pb_inv_ratio[epb].fixed = status
            # Slightly increase transfer capacities to eliminate congestion and harmonize marginal prices.
            if False:
                if status:
                    if network.model.pb_bin_inv[epb].value == 1:
                        network.model.pb_inv_ratio[epb].value = min(network.model.pb_inv_ratio[epb].value * (1 + transfer_increase), 1.0)

        for ecb in network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]:
            network.model.cb_bin_inv[ecb].fixed = status
            network.model.cb_inv_ratio[ecb].fixed = status
            # Slightly increase transfer capacities to eliminate congestion and harmonize marginal prices.
            if False:
                if status:
                    if network.model.cb_bin_inv[ecb].value == 1:
                        network.model.cb_inv_ratio[ecb].value = min(network.model.cb_inv_ratio[ecb].value * (1 + transfer_increase), 1.0)

        for ecb in network.links.index[network.links.p_nom_extendable & (network.links.branch_type == "ptp")]:
            network.model.cb_bin_inv[ecb].fixed = status
            network.model.cb_inv_ratio[ecb].fixed = status
            # Slightly increase transfer capacities to eliminate congestion and harmonize marginal prices.
            if False:
                if status:
                    if network.model.cb_bin_inv[ecb].value == 1:
                        network.model.cb_inv_ratio[ecb].value = min(network.model.cb_inv_ratio[ecb].value * (1 + transfer_increase), 1.0)

        # Fix all extendable generators to fix solution.
        for gen in network.generators[network.generators.p_nom_extendable].index:
            # WARNING Disabling fixing of generators affects marginal costs
            network.model.generator_p_nom[gen].fixed = status
            #network.model.gen_capital_cost[gen] = network.generators.loc[gen, "capital_cost"] * (not status)

        # country_participation is integer and needs to be fixed.
        if hasattr(network.model, "country_participation"):
            countries = list(network.model.country_participation_index)
            for country in countries:
                network.model.country_participation[country].fixed = status

        # Fix/unfix generators who were not fixed before and which have only residual generation, in order not to influence marginal price.
        numerical_precision = 1e-5  # Differentiates residual from significant values.
        if status:
            for gen in network.generators.index.difference(network.generators_t.p_set.columns):
                for sn in snapshots:
                    if abs(network.model.generator_p[(gen, sn)].value) <= numerical_precision:
                        network.model.generator_p[(gen, sn)].fixed = status
        else:
            for gen in network.generators.index.difference(network.generators_t.p_set.columns):
                for sn in snapshots:
                    network.model.generator_p[(gen, sn)].fixed = status

        network.model.preprocess()

        # Print objective function value without investment costs.
        if status:
            print('Operation objective:',value(network.model.objective)-value(network.model.investment_objective))

    def fix_load_curtailment():
        """ Fix the  load curtailment to avoid propagation of VOLL to other buses due to numerical tolerances"""

        # Clear the problem solution before the rerun to reduce memory usage
        if OGEM_options["reference"] != 'expansion':
            network.model.clear_suffix_value('dual')
            for comp in network.model.component_map():
                #if comp not in ['passive_branch_p','link_p']:
                data = getattr(network.model, comp)
                if data.type() in [Var]:
                    for it in data:
                        data[it].value = None
            network.model.solutions.clear(True)
            network.results.clear()

        # Fix and set the load curtailment values with some margin to increase convergence
        for gen in network.generators[network.generators.carrier == 'Slack'].index:
            for sn in network.snapshots:
                network.model.generator_p[(gen, sn)].fixed = True
                network.model.generator_p[(gen, sn)].value = network.generators_t.p.loc[sn, gen] * 1.01

        network.model.preprocess()

        solver_run()
        convergence_status()
        # Unfix load curtailment in case of run iterations.
        for gen in network.generators[network.generators.carrier == 'Slack'].index:
            for sn in network.snapshots:
                network.model.generator_p[(gen, sn)].fixed = False

        # Manually set nodal prices to VOLL at buses and snapshot with load curtailment
        slack_prices = (network.generators_t.p.loc[:, network.generators.carrier == 'Slack'] > 1e-4)
        network.generators.loc[network.generators.carrier == 'Slack', 'marginal_cost'] = 0
        network.generators.loc[slack_prices.columns[slack_prices.any()],'marginal_cost'] = parameters['VOLL']
        slack_prices.columns = network.generators.bus[slack_prices.columns]
        slack_prices = slack_prices.mul(parameters['VOLL'] * network.snapshot_weightings, axis=0)
        network.buses_t.marginal_price.update(slack_prices.where(slack_prices > 0, network.buses_t.marginal_price))

    # Choose symbolic labels for debugging.
    if OGEM_options["reference"] == "expansion":
        symbolic_labels = False
    else:
        symbolic_labels = False

    if symbolic_labels:
        print("Warning: Using symbolic solver labels")

    def solver_run(keepfiles=keep_files, symbolic = symbolic_labels):
        """ Solver run function since solver is called multiple times """

        memory_usage('Pre-solver')
        print(solver_options)

        if 'pypsa' in free_memory:
            with empty_network(network):
                network.results = opt.solve(network.model, suffixes=["dual"],keepfiles=keepfiles, options=solver_options, warmstart=warmstart,tee=True,symbolic_solver_labels = symbolic)
        else:
            network.results = opt.solve(network.model, suffixes=["dual"],keepfiles=keepfiles, options=solver_options, warmstart=warmstart,tee=True,symbolic_solver_labels = symbolic)
        memory_usage('Post-solver')

        print("Solver finished at {:%H:%M}.".format(datetime.datetime.now()))

    def model_run(run_fixed = True):

        def KVL_duals_check():
            """ Since the disjunctive branch flow constraints big_M may be too small, check ex-post the duals of the inactive branches for any non-null values """
            upper_dual = pd.Series(list(network.model.passive_extendable_branch_p_upper.values()),index=pd.MultiIndex.from_tuples(list(network.model.passive_extendable_branch_p_upper.keys()))).map(pd.Series(list(network.model.dual.values()), index=pd.Index(list(network.model.dual.keys())))).fillna(0)
            upper_dual.name = 'upper_dual'
            lower_dual = pd.Series(list(network.model.passive_extendable_branch_p_lower.values()), index=pd.MultiIndex.from_tuples(list(network.model.passive_extendable_branch_p_lower.keys()))).map(pd.Series(list(network.model.dual.values()), index=pd.Index(list(network.model.dual.keys())))).fillna(0)
            lower_dual.name = 'lower_dual'
            inv_bin = pd.Series({l: network.model.pb_bin_inv[l].value == 0 for l in network.model.pb_bin_inv})
            inv_bin.index = inv_bin.index.droplevel(0)
            branch_dual = pd.concat([upper_dual,lower_dual],axis=1).stack().multiply(inv_bin,level=1)

            if branch_dual.abs().values.max() > 1E-1:
                print("Disjunctive KVL constraint dual of {:.2f} at {}".format(branch_dual.abs().values.max(), branch_dual.abs().idxmax()))
                print(branch_dual[branch_dual.abs()>1E-1])
                sys.exit()

        starttime = datetime.datetime.now()
        print("Free model run at {:%H:%M}...".format(starttime))

        if OGEM_options["reference"] == "expansion":
            fix_variables(False) # Unfix variables in case of solve iterations

        # First run for all problems.
        solver_run()

        convergence_status()

        network.MILP_objective = network.objective # Save the objective for use with the Pareto welfare stop criterion.

        if OGEM_options["reference"] != "expansion":
            fix_load_curtailment() # Fix load curtailment and manually set VOLL prices in load-shedding snapshots and buses.

        if logger.level > 0:
            network.results.write()

        # Fix integer and other variables to obtain marginal prices
        if OGEM_options["reference"] == "expansion":
            KVL_duals_check()

            if run_fixed:

                print("Fixed model run at {:%H:%M}...".format(datetime.datetime.now()))

                nonlocal warmstart

                warmstart = False

                fix_variables(True) # Fix variables to obtain marginal prices.

                fix_load_curtailment() # Fix load curtailment and manually set VOLL prices in load-shedding snapshots and buses.

        print("Run finished at {:%H:%M}. Duration: {}.".format(datetime.datetime.now(),datetime.datetime.now() - starttime))

    def clear_solution():
        """ After the warmstart, define values for integer variables for warm start and clear values for continuous variables to guarantee a feasible warm start """
        print('Clearing solution for MIP warm start...')
        threshold = 0.1
        for v in network.model.pb_bin_inv:
            network.model.pb_bin_inv[v].value = (network.model.pb_bin_inv[v].value >= threshold) * 1.
            if network.model.pb_bin_inv[v].value < 1.:
                network.model.pb_inv_ratio[v].value = 0
            else:
                network.model.pb_inv_ratio[v].value = max(network.model.pb_inv_ratio[v].value,network.lines.loc[v[1], 's_nom_min']/network.lines.loc[v[1], 's_nom_max'])

        for v in network.model.cb_bin_inv:
            network.model.cb_bin_inv[v].value = (network.model.cb_bin_inv[v].value >= threshold) * 1.
            if network.model.cb_bin_inv[v].value < 1.:
                network.model.cb_inv_ratio[v].value = 0
            else:
                network.model.cb_inv_ratio[v].value = max(network.model.cb_inv_ratio[v].value,network.links.loc[v, 'p_nom_min']/network.links.loc[v, 'p_nom_max'])

        for v in network.model.conv_p_nom:
            network.model.conv_p_nom[v].value = None

        for v in network.model.generator_p_nom:
            network.model.generator_p_nom[v].value = None

        for v in network.model.country_participation:
            network.model.country_participation[v].value = None

        for v in network.model.passive_branch_p:
            network.model.passive_branch_p[v].value = None

        for v in network.model.abs_passive_branch_p:
            network.model.abs_passive_branch_p[v].value = None

        for v in network.model.voltage_angles:
            network.model.voltage_angles[v].value = None

        for v in network.model.link_p:
            network.model.link_p[v].value = None

        for v in network.model.abs_link_p:
            network.model.abs_link_p[v].value = None

        for v in network.model.voltage_angles:
            network.model.voltage_angles[v].value = None

        for v in network.model.generator_p:
            if network.model.generator_p[v].fixed == False:
                network.model.generator_p[v].value = None

        for v in network.model.state_of_charge:
            network.model.state_of_charge[v].value = None

    if isinstance(free_memory, string_types):
        free_memory = {free_memory}

    if 'pyomo_hack' in free_memory:
        patch_optsolver_free_model_before_solving(opt, network.model)

    if OGEM_options["reference"] == "expansion":

            # Participation variables indicate if country built cooperative links, even if there is no cooperation limit.
            define_participation_variables(network, snapshots, base_welfare)

            fix_constrained_units(network, snapshots) # In the expansion case storage units and generators with inflows need to be fixed.

            define_minimum_ratio(network) # Establish minimum investment ratio for branches where p_nom_min/s_nom_min > 0

    # Solve problem with high integrality tolerance to provide warmstart
    def MIP_warmstart():
        # Relaxes integer variables for warmstart, except the country_participation variable

        for v in network.model.pb_bin_inv:
            network.model.pb_bin_inv[v].domain = NonNegativeReals

        for v in network.model.cb_bin_inv:
            network.model.cb_bin_inv[v].domain = NonNegativeReals

        model_run(run_fixed=False)

        for v in network.model.pb_bin_inv:
            network.model.pb_bin_inv[v].domain = Boolean

        for v in network.model.cb_bin_inv:
            network.model.cb_bin_inv[v].domain = Boolean

        clear_solution()

    if OGEM_options["reference"] == "expansion":
        solver_options.update({'preprocessing_aggregator': 5})

        MIP_warmstart()

        warmstart = True

    model_run()

    if OGEM_options["pareto"]:

        solver_options.update({'preprocessing_aggregator': 5})

        def update_margins():
            """ Since surpluses and nodal prices are model parameters, they need to be updated manually """
            prices = network.buses_t.marginal_price.divide(network.snapshot_weightings,axis=0)

            # Parameters are rounded to decimal_digits to improve the numerical properties of the problem.
            decimal_digits = 5
            for sn in snapshots:
                weight = network.snapshot_weightings[sn]
                for bus in network.buses.index:
                    network.model.marginal_prices[(bus, sn)] = (prices.loc[sn, bus] * weight).round(decimal_digits)
                for gen,(mc,bus) in network.generators[['marginal_cost','bus']].iterrows():
                    network.model.gen_margins[(gen,sn)] = max(((prices.loc[sn, bus]-mc) * weight).round(decimal_digits),0)
                for su,(mc,bus) in network.storage_units[['marginal_cost','bus']].iterrows():
                    network.model.su_margins[(su, sn)] = max(((prices.loc[sn, bus]-mc) * weight).round(decimal_digits),0)
                for pb,(bus0,bus1) in network.lines[['bus0','bus1']].iterrows():
                    network.model.pb_margins[(pb, sn)] = ((prices.loc[sn, bus1] - prices.loc[sn, bus0]) * weight)
                for cb,(bus0,bus1) in network.links[['bus0','bus1']].iterrows():
                    network.model.cb_margins[(cb, sn)] = ((prices.loc[sn, bus1] - prices.loc[sn, bus0]) * weight)
            network.model.preprocess()

        def update_expressions(country):
            # Since the welfare components are set as expressions, they need to be manually updated after price updates.

            network.model.consumer_payments[country].value = value(network.model.consumer_payments_e[country])
            network.model.producer_surplus[country].value = value(network.model.producer_surplus_e[country])
            network.model.storage_surplus[country].value = value(network.model.storage_surplus_e[country])
            network.model.pb_rent[country].value = value(network.model.pb_rent_e[country])
            network.model.ptp_rent[country].value = value(network.model.ptp_rent_e[country])
            network.model.conv_rent[country].value = value(network.model.conv_rent_e[country])
            network.model.investment_cost[country].value = value(network.model.investment_cost_e[country])
            network.model.welfare_constant[country].value = value(network.model.welfare_constant_e[country])
            network.model.preprocess()

        def update_big_M():
            # Given a solution, the big_M can be set as the minimum between a standard value and the require value to unbound each pareto_welfare constraint.

            for country in network.model.country_participation:
                participation = network.model.country_participation[country].value
                network.model.country_participation[country].value = 1.
                update_expressions(country)
                network.model.welfare_big_M[country] = max(- network.model.pareto_welfare[country].lslack() * 1.2 * network.model.welfare_big_M_scale.value, 1.5e2) * parameters['MILP_snapshot_scale'] # Update the minimum big_M value in case the big_M_duals_check fails.
                network.model.country_participation[country].value = participation
                update_expressions(country)

            network.model.welfare_big_M.display()
            network.model.preprocess()

        def update_parameters(trim=False):
            # Pareto iteration requires setting certain parameters as Pyomo parameters, so that rebuilding the model is not necessary at each iteration.
            update_margins()

            # Trimming removes welfare components at certain snapshots according to the threshold, to reduce the welfare constraint equation size and improve the solve time. The complete constraint is checked ex-post to guarantee feasibility.

            if trim:
                # Trimming removes insignificant welfare components to accelerate convergence by reducing the problem size. The actual verification of the constraints is done with all components at all snapshots.

                for country in network.model.country_participation:
                    update_expressions(country) # Update welfare components with current prices.

                print({c: (network.model.country_participation[c].value, network.model.pareto_welfare[c].lslack()) for c in network.model.pareto_welfare})

                margin_threshold = 0.3 # Used to define minimum margins for congestion rent and producer surplus.
                print('Welfare trimming margin threshold', margin_threshold)

                min_pb_margin = sum([abs(network.model.pb_margins[m].value) for m in network.model.pb_margins]) / len(network.model.pb_margins) * margin_threshold
                min_cb_margin = sum([abs(network.model.cb_margins[m].value) for m in network.model.cb_margins]) / len(network.model.cb_margins) * margin_threshold
                min_gen_margin = sum([abs(network.model.gen_margins[m].value) for m in network.model.gen_margins]) / len(network.model.gen_margins) * margin_threshold / 2

                for sn in snapshots:
                    #for gen, (mc, bus) in network.generators[['marginal_cost', 'bus']].iterrows():
                    #    network.model.gen_margins[(gen, sn)] = network.model.gen_margins[(gen,sn)].value * (abs(network.model.gen_margins[(gen,sn)].value) > min_gen_margin)
                    for pb,(bus0,bus1) in network.lines[['bus0','bus1']].iterrows():
                        network.model.pb_margins[(pb, sn)] = network.model.pb_margins[(pb, sn)].value * (abs(network.model.pb_margins[(pb, sn)].value) > min_pb_margin)
                    for cb,(bus0,bus1) in network.links[['bus0','bus1']].iterrows():
                        network.model.cb_margins[(cb, sn)] = network.model.cb_margins[(cb, sn)].value * (abs(network.model.cb_margins[(cb, sn)].value) > min_cb_margin)

            for country in network.model.country_participation:
                update_expressions(country) # Update welfare components with trimming.

            print({c: (network.model.country_participation[c].value, network.model.pareto_welfare[c].lslack()) for c in network.model.pareto_welfare})
            network.model.preprocess()

        def welfare_check():
            # Check if the welfare constraints with updated marginal prices are all satisfied
            update_parameters()

            if _pyomo_version < "5.1": # The slack sign changed in Pyomo.
                check = all([network.model.pareto_welfare[c].lslack() <= - 1/(10**1) for c in network.model.pareto_welfare])
            else:
                check = all([network.model.pareto_welfare[c].lslack() >= - 1/(10**1) for c in network.model.pareto_welfare])

            return check

        def big_M_duals_check():
            # Check the disjunctive pareto welfare constraint duals for non-cooperative countries to guarantee the big_M was large enough.
            # In case the check fails, increase the big_M in update_big_M()

            duals = pd.Series(list(network.model.pareto_welfare.values()), index=pd.Index(list(network.model.pareto_welfare.keys()))).map(
                pd.Series(list(network.model.dual.values()), index=pd.Index(list(network.model.dual.keys())))).fillna(0)
            countries =  pd.Series({l: network.model.country_participation[l].value == 0 for l in network.model.country_participation})
            dual = duals.multiply(countries)
            if dual.abs().values.max() > 1E-2:
                print("Welfare constraint dual of {:.2f} at {}".format(dual.abs().values.max(), dual.abs().idxmax()))
                print(dual[dual.abs() > 1E-2])
                sys.exit()

        print("Running Pareto-constrained model...")

        network.model.welfare_big_M_scale = Param(mutable=True, default=parameters['welfare_scale']) # Scale parameter to improve the numerical properties of the problem.
        if solver_name == 'cplex':
            pareto_tolerance = solver_options['mip_tolerances_mipgap']
        else:
            print("Solver not recognized, exiting.")
            sys.exit()

        define_pareto_welfare(network, snapshots, base_welfare, low_totex_factor, high_totex_factor, pareto_tolerance)

        update_margins()

        update_big_M()

        welfare_satisfied = welfare_check()

        big_M_duals_check()

        objective_constant = True # For the first Pareto iteration this is always true

        while not (welfare_satisfied & objective_constant):
            """ Nodal prices may change while still providing a solution with the same objective function value and respecting the Pareto welfare constraints.
                Thus the iteration stop criterion is rather whether the value is changing and if the welfare constraint is respected, instead of whether marginal prices are constant.
            """

            print("Welfare satisfied: {}, objective constant: {}, iterating...".format(welfare_satisfied,objective_constant))

            update_parameters(trim=True)

            update_big_M()

            # Save i - 1 objective and price values for comparison.
            base_objective = network.MILP_objective
            base_shadow = network.buses_t.marginal_price.copy()

            if welfare_satisfied == True:
                # If welfare was satisfied the solution is feasible for a warmstart.
                clear_solution()

                warmstart = True

            elif False: # Disabled as warmstart did not improve solution when the welfare constraint was not satisfied anyway.
            # If not, solve for warmstart relaxing investment binaries.

                warmstart = True

                MIP_warmstart()

                warmstart = True

            model_run()

            welfare_satisfied = welfare_check()

            big_M_duals_check()

            print(base_objective,network.MILP_objective)

            objective_constant = (abs((base_objective - network.MILP_objective) / network.MILP_objective) < pareto_tolerance) # The objective will never be truly constant, but can vary only withing the mip gap tolerance.

            shadow_gap = (network.buses_t.marginal_price - base_shadow)

            print("Maximum nodal price gap: ${:0.2f}".format(shadow_gap.abs().values.max()))

        else:

            print("No more iterations necessary")


