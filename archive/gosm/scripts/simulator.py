####################################################
#                   prescient                      #
####################################################
# # Aug 2015: HERE LIES THE COMBINATION OF CONFCOMP.PY, POPULATE_INPUT_DIRECTORIES.PY AND PRESCIENT_SIM.PY.
# # IT INCLUDES ALL FUNCTIONALITY. ALL OPTIONS
# # AND ALL BASH FILES USED FOR THE PREVIOUS ARE USABLE ON THIS WITH SMALL MODIFICATIONS.
# # 1) CHANGE THE PY FILE TO PRESCIENT.PY
# # 2) ADD THE OPTION --run-{x} where x = {populator,simulator,scenarios}
#
# # Note that bash files are outdated as of Sept 2016 and instead txt files paired with the runner.py.
# import random
import sys
import os
import shutil
import random
import traceback
import csv
import time
import subprocess
import datetime
import math
from optparse import OptionParser, OptionGroup

try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

try:
    import cProfile as profile
except ImportError:
    import profile

import matplotlib
# the following forces matplotlib to not use any Xwindows backend.
# taken from stackoverflow, of course.
matplotlib.use('Agg')

import numpy as np
import pandas as pd

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False
    
try:
    import dateutil
except:
    print("***Failed to import python dateutil module - try: easy_install python-dateutil")
    sys.exit(1)

from six import iterkeys, itervalues, iteritems
from pyomo.core import *
from pyomo.opt import *
from pyomo.pysp.ef import create_ef_instance
from pyomo.pysp.scenariotree import ScenarioTreeInstanceFactory, ScenarioTree
from pyomo.pysp.phutils import find_active_objective, cull_constraints_from_instance
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
import pyutilib

import prescient.sim.MasterOptions as MasterOptions

# import plotting capabilities
import prescient.sim.graphutils as graphutils
import prescient.sim.storagegraphutils as storagegraphutils

# random-but-useful global constants.
DEFAULT_MAX_LABEL_LENGTH = 15

#############################################################################################
###############################       START PRESCIENT        ################################
#############################################################################################

###############################
# Helper function definitions #
###############################

# the simulator relies on the presence of various methods in the reference 
# model module - verify that these exist.

def validate_reference_model(module):
    required_methods = ["fix_binary_variables", "free_binary_variables", "define_suffixes", "reconstruct_instance_for_t0_changes"]
    for method in required_methods:
        if not hasattr(module, method):
            raise RuntimeError("Reference model module does not have required method=%s" % method)

def call_solver(solver,instance,**kwargs):
    # # Needed to allow for persistent solver options July2015
    return solver.solve(instance, load_solutions=False, **kwargs)

def round_small_values(x, p=1e-6):
    # Rounds values that are within (-1e-6, 1e-6) to 0.
    try:
        if math.fabs(x) < p:
            return 0
        return x
    except:
        raise RuntimeError("Utility function round_small_values failed on input=%s, p=%f" % (str(x), p))
#
# a utility to create the hourly (deterministic) economic dispatch instance, given
# the prior day's RUC solution, the basic ruc model, and the ruc instance to simulate.
# TBD - we probably want to grab topology from somewhere, even if the stochastic 
#       RUC is not solved with the topology.
# a description of the input arguments:
#
# 1 - sced_model: an uninstantiated (abstract) unit commitment model. 
#                 used as the basis for the constructed and returned model.
#
# 2 - stochastic_scenario_instances: a map from scenario name to scenario instance, for the stochastic RUC.
#                              these instances are solved, with results loaded.
#
# 3 - ruc_instance_to_simulate: a single scenario / deterministic RUC instance. 
#                               (projected) realized demands and renewables output for the SCED are extracted 
#                               from this instance, as are static / topological features of the network.
#
# 4 - prior_sced_instance: the sced instance from the immediately preceding time period.
#                          provides the initial generator (T0) state information.
#                          can be None, in which case - TBD?!
#
# 5 - hour_to_simulate: the hour of the day to simulate, 0-based.
#

# NOTES: 
# 1) It is critical that the SCED be multi-period, to manage ramp-down / shut-off constraints. 
# 2) As a result, we define SCED as a weird-ish version of the SCUC - namely one in which
#    the initial conditions are taken from the prior period SCED instance time period 1 (the
#    first real period), and the binaries for all remaining subsequent time periods in the day
#    are taken from (and fixed to) the values in the stochastic RUC - as these are the commitments 
#    that must be satisfied.
# 3) We are presently taking the demand in future time periods from the instance to simulate. this
#    may sound a bit like cheating, but we can argue that the one would use the scenario from the
#    RUC that is closest to that observed. this brings to mind a matching / selection scheme, but
#    that is presently not conducted.

def daterange(start_day_date, end_day_date):
    for n in range(int ((end_day_date - start_day_date).days)+1):
        yield start_day_date + datetime.timedelta(n)

def create_sced_instance(sced_model, 
                         today_ruc_instance, today_stochastic_scenario_instances, today_scenario_tree,
                         tomorrow_ruc_instance, tomorrow_stochastic_scenario_instances, tomorrow_scenario_tree,
                         ruc_instance_to_simulate,  # providies actuals and an instance to query
                         prior_sced_instance,  # used for initial conditions if initial_from_ruc=False
                         actual_demand,  # native Python dictionary containing the actuals
                         demand_forecast_error, 
                         actual_min_renewables,  # native Python dictionary containing the actuals
                         actual_max_renewables,  # native Python dictionary containing the actuals
                         renewables_forecast_error,
                         hour_to_simulate,
                         reserve_factor, 
                         hours_in_objective=1,
                         # by default, just worry about cost minimization for the hour to simulate.
                         sced_horizon=24,
                         # by default simulate a SCED for 24 hours, to pass through the midnight boundary successfully
                         initialize_from_ruc=True,
                         use_prescient_forecast_error=True,
                         use_persistent_forecast_error=False):

    assert sced_model != None
    assert ruc_instance_to_simulate != None
    assert hour_to_simulate >= 0
    assert reserve_factor >= 0.0

    if prior_sced_instance is None:
        assert hour_to_simulate == 0

    # NOTE: if the prior SCED instance is None, then we extract the unit T0 state information from the
    #       stochastic RUC instance (solved for the preceding day).

    # the input hour is 0-based, but the time periods in our UC and SCED optimization models are one-based.
    hour_to_simulate += 1 

    #################################################################
    # compute the T0 state parameters, based on the input instances #
    #################################################################

    UnitOnT0Dict         = {}
    UnitOnT0StateDict    = {}
    PowerGeneratedT0Dict = {}

    # there are a variety of ways to accomplish this, which can result in
    # radically different simulator behavior. these are driven by keyword
    # arguments - we don't perform checking for specification of multiple
    # alternatives. 

    if initialize_from_ruc:
        # the simplest initialization method is to set the initial conditions 
        # to those found in the RUC (deterministic or stochastic) instance. this has 
        # the advantage that simulating from an in-sample scenario will yield
        # a feasible sced at each simulated hour.

        # if there isn't a set of stochastic scenario instances, then 
        # we're dealing with a deterministic RUC - use it directly.
        if today_stochastic_scenario_instances == None:
            print("")
            print("Drawing initial conditions from deterministic RUC initial conditions")
            for g in today_ruc_instance.ThermalGenerators:
                UnitOnT0Dict[g] = value(today_ruc_instance.UnitOnT0[g])
                UnitOnT0StateDict[g] = value(today_ruc_instance.UnitOnT0State[g])
                PowerGeneratedT0Dict[g] = value(today_ruc_instance.PowerGeneratedT0[g])
        else:
            print("")
            print("Drawing initial conditions from stochastic RUC initial conditions")
            arbitrary_scenario_instance = today_stochastic_scenario_instances[list(today_stochastic_scenario_instances.keys())[0]]
            for g in ruc_instance_to_simulate.ThermalGenerators:
                UnitOnT0Dict[g] = value(arbitrary_scenario_instance.UnitOnT0[g])
                UnitOnT0StateDict[g] = value(arbitrary_scenario_instance.UnitOnT0State[g])
                PowerGeneratedT0Dict[g] = value(arbitrary_scenario_instance.PowerGeneratedT0[g])

    else:
        # TBD: Clean code below up - if we get this far, shouldn't we always have a prior sched instance?
        print("")
        print("Drawing initial conditions from prior SCED solution, if available")
        today_root_node = today_scenario_tree.findRootNode()
        for g in ruc_instance_to_simulate.ThermalGenerators:

            if prior_sced_instance is None:
                # if there is no prior sced instance, then 
                # let the T0 state be equal to the unit state
                # in the initial time period of the stochastic RUC.
                unit_on = int(round(today_root_node.get_variable_value("UnitOn", (g, "Stage_1", hour_to_simulate))))
                UnitOnT0Dict[g] = unit_on
            else:
                unit_on = int(round(value(prior_sced_instance.UnitOn[g, 1])))
                UnitOnT0Dict[g] = unit_on

            # since we are dealing with a single time period model, propagate the
            # UnitOn state (which will be fixed in any case) backward into the past.
            if unit_on == 1:
                UnitOnT0StateDict[g] = 24 
            else:
                UnitOnT0StateDict[g] = -24 

            if prior_sced_instance is None:
                # TBD - this is the problem - we need to not compute the expected power generated, but rather
                #       the actual T0 level specified in the stochastic RUC (same for all scenarios) - this
                #       really should be option-driven, so we can retain the older variant.
                total_power_generated = 0.0
                for instance in today_stochastic_scenario_instances.values():
                    total_power_generated += value(instance.PowerGenerated[g, 1])
                PowerGeneratedT0Dict[g] = total_power_generated / float(len(today_stochastic_scenario_instances))
            else:
                PowerGeneratedT0Dict[g] = value(prior_sced_instance.PowerGenerated[g, 1])

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization 
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.
            min_power_output = value(ruc_instance_to_simulate.MinimumPowerOutput[g])
            max_power_output = value(ruc_instance_to_simulate.MaximumPowerOutput[g])

            candidate_power_generated = PowerGeneratedT0Dict[g]
                
            # TBD: Eventually make the 1e-5 an user-settable option.
            if math.fabs(min_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = min_power_output
            elif math.fabs(max_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = max_power_output
            
            # related to the above (and this is a case that is not caught by the above),
            # if the unit is off, then the power generated at t0 must be equal to 0 -
            # no tolerances allowed.
            if unit_on == 0:
                PowerGeneratedT0Dict[g] = 0.0

    ################################################################################
    # initialize the demand and renewables data, based on the forecast error model #
    ################################################################################

    if use_prescient_forecast_error:

        demand_dict = dict(((b, t+1), value(actual_demand[b, hour_to_simulate + t]))
                           for b in ruc_instance_to_simulate.Buses for t in range(0, sced_horizon))
        min_nondispatch_dict = dict(((g, t+1), value(actual_min_renewables[g, hour_to_simulate + t]))
                                    for g in ruc_instance_to_simulate.AllNondispatchableGenerators
                                    for t in range(0, sced_horizon))
        max_nondispatch_dict = dict(((g, t+1), value(actual_max_renewables[g, hour_to_simulate + t]))
                                    for g in ruc_instance_to_simulate.AllNondispatchableGenerators
                                    for t in range(0, sced_horizon))

    else:  # use_persistent_forecast_error:

        # there is redundancy between the code for processing the two cases below. 
        # for now, leaving this alone for clarity / debug.

        demand_dict = {}
        min_nondispatch_dict = {}
        max_nondispatch_dict = {}

        if today_stochastic_scenario_instances == None:  # we're running in deterministic mode

            # the current hour is necessarily (by definition) the actual.
            for b in ruc_instance_to_simulate.Buses:
                demand_dict[(b,1)] = value(actual_demand[b, hour_to_simulate])

            # for each subsequent hour, apply a simple persistence forecast error model to account for deviations.
            for b in ruc_instance_to_simulate.Buses:
                forecast_error_now = demand_forecast_error[(b, hour_to_simulate)]
                actual_now = value(actual_demand[b, hour_to_simulate])
                forecast_now = actual_now + forecast_error_now

                for t in range(1, sced_horizon):
                    # IMPT: forecast errors (and therefore forecasts) are relative to actual demand, 
                    #       which is the only reason that the latter appears below - to reconstruct
                    #       the forecast. thus, no presicence is involved.
                    forecast_error_later = demand_forecast_error[(b,hour_to_simulate + t)]
                    actual_later = value(actual_demand[b,hour_to_simulate + t])
                    forecast_later = actual_later + forecast_error_later
                    # 0 demand can happen, in some odd circumstances (not at the ISO level!).
                    if forecast_now != 0.0:
                        demand_dict[(b, t+1)] = (forecast_later/forecast_now)*actual_now
                    else:
                        demand_dict[(b, t+1)] = 0.0

            # repeat the above for renewables.
            for g in ruc_instance_to_simulate.AllNondispatchableGenerators:
                min_nondispatch_dict[(g, 1)] = value(actual_min_renewables[g, hour_to_simulate])
                max_nondispatch_dict[(g, 1)] = value(actual_max_renewables[g, hour_to_simulate])
                
            for g in ruc_instance_to_simulate.AllNondispatchableGenerators:
                forecast_error_now = renewables_forecast_error[(g, hour_to_simulate)]
                actual_now = value(actual_max_renewables[g, hour_to_simulate])
                forecast_now = actual_now + forecast_error_now

                for t in range(1, sced_horizon):
                    # forecast errors are with respect to the maximum - that is the actual maximum power available.
                    forecast_error_later = renewables_forecast_error[(g, hour_to_simulate + t)]
                    actual_later = value(actual_max_renewables[g, hour_to_simulate + t])
                    forecast_later = actual_later + forecast_error_later

                    if forecast_now != 0.0:
                        max_nondispatch_dict[(g, t+1)] = (forecast_later/forecast_now)*actual_now
                    else:
                        max_nondispatch_dict[(g, t+1)] = 0.0
                    # TBD - fix this - it should be non-zero!
                    min_nondispatch_dict[(g, t+1)] = 0.0

        else:  # we're running in stochastic mode

            # find the nearest scenario instance, from the current day's first time period through now.
            # this scenario will be used to extract the (point) forecast quantities for demand and renewables. 
            # this process can be viewed as identifying a dynamic forecast.

            nearest_scenario_instance = find_nearest_scenario(ruc_instance_to_simulate, today_stochastic_scenario_instances, hour_to_simulate)
            print("Nearest scenario to observations for purposes of persistence-based forecast adjustment=" + nearest_scenario_instance.name)

            # the current hour is necessarily (by definition) the actual.
            for b in ruc_instance_to_simulate.Buses:
                demand_dict[(b, 1)] = value(actual_demand[b, hour_to_simulate])

            # for each subsequent hour, apply a simple persistence forecast error model to account for deviations.
            for b in ruc_instance_to_simulate.Buses:
                actual_now = value(actual_demand[b, hour_to_simulate])
                forecast_now = value(nearest_scenario_instance.Demand[b, hour_to_simulate])

                for t in range(1, sced_horizon):
                    # the forecast later is simply the value projected by the nearest scenario.
                    forecast_later = value(nearest_scenario_instance.Demand[b, hour_to_simulate+t])
                    demand_dict[(b, t+1)] = (forecast_later/forecast_now) * actual_now

            # repeat the above for renewables.
            for g in ruc_instance_to_simulate.AllNondispatchableGenerators:
                min_nondispatch_dict[(g, 1)] = value(actual_min_renewables[g, hour_to_simulate])
                max_nondispatch_dict[(g, 1)] = value(actual_max_renewables[g, hour_to_simulate])
                
            for g in ruc_instance_to_simulate.AllNondispatchableGenerators:
                actual_now = value(actual_max_renewables[g, hour_to_simulate])
                forecast_now = value(nearest_scenario_instance.MaxNondispatchablePower[g, hour_to_simulate])

                for t in range(1, sced_horizon):
                    forecast_later = value(nearest_scenario_instance.MaxNondispatchablePower[g, hour_to_simulate+t])

                    if forecast_now != 0.0:
                        max_nondispatch_dict[(g, t+1)] = (forecast_later/forecast_now) * actual_now
                    else:
                        max_nondispatch_dict[(g, t+1)] = 0.0
                    # TBD - fix this - it should be non-zero!
                    min_nondispatch_dict[(g, t+1)] = 0.0

    ##########################################################################
    # construct the data dictionary for instance initialization from scratch #
    ##########################################################################

    ed_data = {None: { 'Buses': {None: [b for b in ruc_instance_to_simulate.Buses]},
                       'StageSet': {None: ["Stage_1", "Stage_2"]},
                       'TimePeriodLength': {None: 1.0},
                       'NumTimePeriods': {None: sced_horizon},
                       'CommitmentTimeInStage': {"Stage_1": list(range(1, sced_horizon+1)), "Stage_2": []},
                       'GenerationTimeInStage': {"Stage_1": [], "Stage_2": list(range(1, sced_horizon+1))},
                       'TransmissionLines': {None : list(ruc_instance_to_simulate.TransmissionLines)},
                       'BusFrom': dict((line, ruc_instance_to_simulate.BusFrom[line])
                                       for line in ruc_instance_to_simulate.TransmissionLines),
                       'BusTo': dict((line, ruc_instance_to_simulate.BusTo[line])
                                     for line in ruc_instance_to_simulate.TransmissionLines),
                       'Impedence': dict((line, ruc_instance_to_simulate.Impedence[line])
                                         for line in ruc_instance_to_simulate.TransmissionLines),
                       'ThermalLimit': dict((line, ruc_instance_to_simulate.ThermalLimit[line])
                                            for line in ruc_instance_to_simulate.TransmissionLines),
                       'ThermalGenerators': {None: [gen for gen in ruc_instance_to_simulate.ThermalGenerators]},
                       'ThermalGeneratorType': dict((gen, ruc_instance_to_simulate.ThermalGeneratorType[gen])
                                                    for gen in ruc_instance_to_simulate.ThermalGenerators),
                       'ThermalGeneratorsAtBus': dict((b, [gen for gen in ruc_instance_to_simulate.ThermalGeneratorsAtBus[b]])
                                                      for b in ruc_instance_to_simulate.Buses),
                       'QuickStart': dict((g, value(ruc_instance_to_simulate.QuickStart[g])) for g in ruc_instance_to_simulate.ThermalGenerators),
                       'QuickStartGenerators': {None: [g for g in ruc_instance_to_simulate.QuickStartGenerators]},
                       'AllNondispatchableGenerators': {None: [g for g in ruc_instance_to_simulate.AllNondispatchableGenerators]},
                       'NondispatchableGeneratorType': dict((gen, ruc_instance_to_simulate.NondispatchableGeneratorType[gen])
                                                            for gen in ruc_instance_to_simulate.AllNondispatchableGenerators),
                       'MustRunGenerators': {None: [g for g in ruc_instance_to_simulate.MustRunGenerators]},
                       'NondispatchableGeneratorsAtBus': dict((b, [gen for gen in ruc_instance_to_simulate.NondispatchableGeneratorsAtBus[b]])
                                                              for b in ruc_instance_to_simulate.Buses),
                       'Demand': demand_dict,
                       # TBD - for now, we are ignoring the ReserveRequirement parameters for the economic dispatch
                       # we do handle the ReserveFactor, below.
                       'MinimumPowerOutput': dict((g, value(ruc_instance_to_simulate.MinimumPowerOutput[g]))
                                                  for g in ruc_instance_to_simulate.ThermalGenerators),
                       'MaximumPowerOutput': dict((g, value(ruc_instance_to_simulate.MaximumPowerOutput[g]))
                                                  for g in ruc_instance_to_simulate.ThermalGenerators),
                       'MinNondispatchablePower': min_nondispatch_dict,
                       'MaxNondispatchablePower': max_nondispatch_dict,
                       'NominalRampUpLimit': dict((g, value(ruc_instance_to_simulate.NominalRampUpLimit[g]))
                                                  for g in ruc_instance_to_simulate.ThermalGenerators),
                       'NominalRampDownLimit': dict((g, value(ruc_instance_to_simulate.NominalRampDownLimit[g]))
                                                    for g in ruc_instance_to_simulate.ThermalGenerators),
                       'StartupRampLimit': dict((g, value(ruc_instance_to_simulate.StartupRampLimit[g]))
                                                for g in ruc_instance_to_simulate.ThermalGenerators),
                       'ShutdownRampLimit': dict((g, value(ruc_instance_to_simulate.ShutdownRampLimit[g]))
                                                 for g in ruc_instance_to_simulate.ThermalGenerators),
                       'MinimumUpTime': dict((g, value(ruc_instance_to_simulate.MinimumUpTime[g]))
                                             for g in ruc_instance_to_simulate.ThermalGenerators),
                       'MinimumDownTime': dict((g, value(ruc_instance_to_simulate.MinimumDownTime[g]))
                                               for g in ruc_instance_to_simulate.ThermalGenerators),
                       'UnitOnT0': UnitOnT0Dict,
                       'UnitOnT0State': UnitOnT0StateDict,
                       'PowerGeneratedT0': PowerGeneratedT0Dict,
                       'ProductionCostA0': dict((g, value(ruc_instance_to_simulate.ProductionCostA0[g]))
                                                for g in ruc_instance_to_simulate.ThermalGenerators),
                       'ProductionCostA1': dict((g, value(ruc_instance_to_simulate.ProductionCostA1[g]))
                                                for g in ruc_instance_to_simulate.ThermalGenerators),
                       'ProductionCostA2': dict((g, value(ruc_instance_to_simulate.ProductionCostA2[g]))
                                                for g in ruc_instance_to_simulate.ThermalGenerators),
                       'CostPiecewisePoints': dict((g, [point for point in ruc_instance_to_simulate.CostPiecewisePoints[g]])
                                                   for g in ruc_instance_to_simulate.ThermalGenerators),
                       'CostPiecewiseValues': dict((g, [value for value in ruc_instance_to_simulate.CostPiecewiseValues[g]])
                                                   for g in ruc_instance_to_simulate.ThermalGenerators),
                       'FuelCost': dict((g, value(ruc_instance_to_simulate.FuelCost[g]))
                                        for g in ruc_instance_to_simulate.ThermalGenerators),
                       'NumGeneratorCostCurvePieces': {None:value(ruc_instance_to_simulate.NumGeneratorCostCurvePieces)},
                       'StartupLags': dict((g, [point for point in ruc_instance_to_simulate.StartupLags[g]])
                                           for g in ruc_instance_to_simulate.ThermalGenerators),
                       'StartupCosts': dict((g, [point for point in ruc_instance_to_simulate.StartupCosts[g]])
                                            for g in ruc_instance_to_simulate.ThermalGenerators),
                       'ShutdownFixedCost': dict((g, value(ruc_instance_to_simulate.ShutdownFixedCost[g]))
                                                 for g in ruc_instance_to_simulate.ThermalGenerators),
                       'Storage': {None: [s for s in ruc_instance_to_simulate.Storage]},
                       'StorageAtBus': dict((b, [s for s in ruc_instance_to_simulate.StorageAtBus[b]])
                                            for b in ruc_instance_to_simulate.Buses),
                       'MinimumPowerOutputStorage': dict((s, value(ruc_instance_to_simulate.MinimumPowerOutputStorage[s]))
                                                         for s in ruc_instance_to_simulate.Storage),
                       'MaximumPowerOutputStorage': dict((s, value(ruc_instance_to_simulate.MaximumPowerOutputStorage[s]))
                                                         for s in ruc_instance_to_simulate.Storage),
                       'MinimumPowerInputStorage': dict((s, value(ruc_instance_to_simulate.MinimumPowerInputStorage[s]))
                                                        for s in ruc_instance_to_simulate.Storage),
                       'MaximumPowerInputStorage': dict((s, value(ruc_instance_to_simulate.MaximumPowerInputStorage[s]))
                                                        for s in ruc_instance_to_simulate.Storage),
                       'NominalRampUpLimitStorageOutput': dict((s, value(ruc_instance_to_simulate.NominalRampUpLimitStorageOutput[s]))
                                                               for s in ruc_instance_to_simulate.Storage),
                       'NominalRampDownLimitStorageOutput': dict((s, value(ruc_instance_to_simulate.NominalRampDownLimitStorageOutput[s]))
                                                                 for s in ruc_instance_to_simulate.Storage),
                       'NominalRampUpLimitStorageInput': dict((s, value(ruc_instance_to_simulate.NominalRampUpLimitStorageInput[s]))
                                                              for s in ruc_instance_to_simulate.Storage),
                       'NominalRampDownLimitStorageInput': dict((s, value(ruc_instance_to_simulate.NominalRampDownLimitStorageInput[s]))
                                                                for s in ruc_instance_to_simulate.Storage),
                       'MaximumEnergyStorage': dict((s, value(ruc_instance_to_simulate.MaximumEnergyStorage[s]))
                                                    for s in ruc_instance_to_simulate.Storage),
                       'MinimumSocStorage': dict((s, value(ruc_instance_to_simulate.MinimumSocStorage[s]))
                                                 for s in ruc_instance_to_simulate.Storage),
                       'InputEfficiencyEnergy': dict((s, value(ruc_instance_to_simulate.InputEfficiencyEnergy[s]))
                                                     for s in ruc_instance_to_simulate.Storage),
                       'OutputEfficiencyEnergy': dict((s, value(ruc_instance_to_simulate.OutputEfficiencyEnergy[s]))
                                                      for s in ruc_instance_to_simulate.Storage),
                       'RetentionRate': dict((s, value(ruc_instance_to_simulate.RetentionRate[s]))
                                             for s in ruc_instance_to_simulate.Storage),
                       'EndPointSocStorage': dict((s, value(ruc_instance_to_simulate.EndPointSocStorage[s]))
                                                  for s in ruc_instance_to_simulate.Storage),
                       'StoragePowerOutputOnT0': dict((s, value(ruc_instance_to_simulate.StoragePowerOutputOnT0[s]))
                                                      for s in ruc_instance_to_simulate.Storage),
                       'StoragePowerInputOnT0': dict((s, value(ruc_instance_to_simulate.StoragePowerInputOnT0[s]))
                                                     for s in ruc_instance_to_simulate.Storage),
                       'LoadMismatchPenalty': {None: value(ruc_instance_to_simulate.LoadMismatchPenalty)},
                       'ReserveShortfallPenalty': {None: value(ruc_instance_to_simulate.ReserveShortfallPenalty)}
                      }
               }
    if prior_sced_instance!=None:
        ed_data[None]["StorageSocOnT0"]=dict((s, value(prior_sced_instance.SocStorage[s, 1]))
                                             for s in ruc_instance_to_simulate.Storage)
    else:
        ed_data[None]["StorageSocOnT0"]=dict((s, value(ruc_instance_to_simulate.StorageSocOnT0[s])) for s in ruc_instance_to_simulate.Storage)

    if reserve_factor > 0.0:
        ed_data[None]["ReserveFactor"] = {None: reserve_factor}

    #######################
    # create the instance #
    #######################

    # sced_instance = sced_model.create_instance(data_dict=ed_data)
    # Node: in pyomo 4.1, the keyword argument changed from data_dict to data.
    sced_instance = sced_model.create_instance(data=ed_data)

    ##################################################################
    # set the unit on variables in the sced instance to those values #
    # found in the stochastic ruc instance for the input time period #
    ##################################################################

    # NOTE: the values coming back from the RUC solves can obviously
    #       be fractional, due to numerical tolerances on integrality.
    #       we could enforce integrality at the solver level, but are
    #       not presently. instead, we round, to force integrality.
    #       this has the disadvantage of imposing a disconnect between
    #       the stochastic RUC solution and the SCED, but for now,
    #       we will live with it.
    for t in sorted(sced_instance.TimePeriods):
        # the input t and hour_to_simulate are both 1-based => so is the translated_t
        translated_t = t + hour_to_simulate - 1
#        print "T=",t
#        print "TRANSLATED T=",translated_t

        for g in sorted(sced_instance.ThermalGenerators):
            # CRITICAL: today's ruc instance and tomorrow's ruc instance are not guaranteed
            #           to be consistent, in terms of the value of the binaries in the time 
            #           periods in which they overlap, nor the projected power levels for 
            #           time units in which they overlap. originally, we were trying to be
            #           clever and using today's ruc instance for hours <= 24, and tomorrow's
            #           ruc instance for hours > 24, but we didn't think this carefully 
            #           enough through. this issue should be revisited. at the moment, we 
            #           are relying on the consistency between our projections for the unit
            #           states at midnight and the values actually observed. these should not
            #           be too disparate, given that the projection is only 3 hours out.
            # NEW: this is actually a problem 3 hours out - especially if the initial state
            #      projections involving PowerGeneratedT0 is incorrect. and these are often
            #      divergent. 
            if translated_t > 24:
                if tomorrow_scenario_tree != None:
                    if tomorrow_stochastic_scenario_instances != None:
                        new_value = int(round(tomorrow_scenario_tree.findRootNode().get_variable_value("UnitOn", (g, translated_t - 24))))
                    else:
                        new_value = int(round(value(tomorrow_ruc_instance.UnitOn[g, translated_t - 24])))
#                    print "T=",t,"G=",g," TAKING UNIT ON FROM TOMORROW RUC - VALUE=",new_value,"HOUR TO TAKE=",translated_t - 24
                else:
                    if today_stochastic_scenario_instances != None:
                        new_value = int(round(today_scenario_tree.findRootNode().get_variable_value("UnitOn", (g, translated_t))))
                    else:
                        new_value = int(round(value(today_ruc_instance.UnitOn[g, translated_t])))
#                    print "T=",t,"G=",g," TAKING UNIT ON FROM TODAY RUC - VALUE=",new_value,"HOUR TO TAKE=",translated_t
            else:
                if today_stochastic_scenario_instances != None:
                    new_value = int(round(today_scenario_tree.findRootNode().get_variable_value("UnitOn", (g, translated_t))))
                else:
                    new_value = int(round(value(today_ruc_instance.UnitOn[g, translated_t])))
#                print "T=",t,"G=",g," TAKING UNIT ON FROM TODAY RUC - VALUE=",new_value,"HOUR TO TAKE=",translated_t
            sced_instance.UnitOn[g, t] = new_value


    # before fixing all of the UnitOn variables, make sure they
    # have legimitate values - otherwise, an unintelligible 
    # error from preprocessing will result.

    # all models should have UnitOn variables. some models have
    # other binaries, e.g., UnitStart and UnitStop, but presolve
    # (at least CPLEX and Gurobi's) should be able to eliminate 
    # those easily enough.
    for var_data in itervalues(sced_instance.UnitOn):
        if value(var_data) is None:
            raise RuntimeError("The index=" + str(var_data.index()) +
                               " of the UnitOn variable in the SCED instance is None - "
                               "fixing and subsequent preprocessing will fail")
        var_data.fix()

    # establish the objective function for the hour to simulate - which is simply to 
    # minimize production costs during this time period. no fixed costs to worry about.
    # however, we do need to penalize load shedding across all time periods - otherwise,
    # very bad things happen.
    objective = find_active_objective(sced_instance)    
    expr = sum(sced_instance.ProductionCost[g, i]
               for g in sced_instance.ThermalGenerators
               for i in range(1,hours_in_objective+1)) \
           + (sced_instance.LoadMismatchPenalty *
              sum(sced_instance.posLoadGenerateMismatch[b, t] + sced_instance.negLoadGenerateMismatch[b, t]
                  for b in sced_instance.Buses for t in sced_instance.TimePeriods)) + \
           (sced_instance.ReserveShortfallPenalty * sum(sced_instance.ReserveShortfall[t]
                                                        for t in sced_instance.TimePeriods))

    objective.expr = expr

    # preprocess after all the mucking around.
    sced_instance.preprocess()

    return sced_instance

# 
# a utility to determine if there are any ramp rate violations in the input SCED.
# if there are, they are optionally reported. intended to diagnose which SCED relaxations
# are required to achieve feasibility.
# returns true/false, plus sets of generators with various types of violations.
#

def sced_violates_ramp_rates(sced_instance, report_violations=True):

    violations_reported = False

    nominal_up_gens_violated = set()
    nominal_down_gens_violated = set()
    startup_gens_violated = set()
    shutdown_gens_violated = set()
    value_missing = False

    for g in sced_instance.ThermalGenerators:

        for t in sced_instance.TimePeriods:

            this_period_unit_on = value(sced_instance.UnitOn[g, t])

            if t == 1:
                prior_period = 0
                prior_period_unit_on = value(sced_instance.UnitOnT0[g])
                generated_delta = value(sced_instance.PowerGenerated[g, t]) - value(sced_instance.PowerGeneratedT0[g])
            else:
                prior_period = t-1
                prior_period_unit_on = value(sced_instance.UnitOn[g, t-1])
                generated_delta = value(sced_instance.PowerGenerated[g, t]) - \
                                  value(sced_instance.PowerGenerated[g, t-1])

            if (prior_period_unit_on == 1) and (this_period_unit_on == 1):
                if generated_delta < 0.0:
                    if math.fabs(generated_delta) > value(sced_instance.ScaledNominalRampDownLimit[g]):
                        violations_reported = True
                        print("***WARNING: Nominal ramp down rate violated for g=" +
                              str(g) + " from time period=" + str(prior_period) + " to time period=" + str(t))
                        print("            delta=" + str(math.fabs(generated_delta)) +
                              " exceeds the nominal limit=" + str(value(sced_instance.ScaledNominalRampDownLimit[g])))
                        nominal_down_gens_violated.add(g)
                else:
                    if generated_delta > value(sced_instance.ScaledNominalRampUpLimit[g]):
                        violations_reported = True
                        print("***WARNING: Nominal ramp up rate violated for g="
                              + str(g) + " from time period=" + str(prior_period) + " to time period=" + str(t))
                        print("            delta=" + str(math.fabs(generated_delta)) +
                              " exceeds the nominal limit=" + str(value(sced_instance.ScaledNominalRampUpLimit[g])))
                        nominal_up_gens_violated.add(g)

            elif (prior_period_unit_on == 0) and (this_period_unit_on == 1):
                if generated_delta > value(sced_instance.ScaledStartupRampLimit[g]) :
                    violations_reported = True
                    print("***WARNING: Startup ramp rate violated for g=" +
                          str(g) + " from time period=" + str(prior_period) + " to time period=" + str(t))
                    print("            rate=" + str(generated_delta) +
                          " exceeds the limit=" + str(value(sced_instance.ScaledStartupRampLimit[g])))
                    startup_gens_violated.add(g)

            elif (prior_period_unit_on == 1) and (this_period_unit_on == 0):
                if -generated_delta > value(sced_instance.ScaledShutdownRampLimit[g]) :
                    violations_reported = True
                    print("***WARNING: Startdown ramp rate violated for g=" +
                          str(g) + " from time period=" + str(prior_period) + " to time period=" + str(t))
                    print("            rate=" + str(-generated_delta) +
                          " exceeds the limit=" + str(value(sced_instance.ScaledShutdownRampLimit[g])))
                    startup_gens_violated.add(g)

    return violations_reported, nominal_up_gens_violated, nominal_down_gens_violated, startup_gens_violated, shutdown_gens_violated

#
# a utility to inflate the ramp rates for those units in the input instance with violations.
# re-preprocesses the instance as necessary, so it is ready to solve. the input
# factor must be >= 0 and < 1. The new limits are obtained via the scale factor 1 + scale_factor.
# NOTE: We could and probably should just relax the limits by the degree to which they are 
#       violated - the present situation is more out of legacy, and the fact that we would have
#       to restructure what is returned in the reporting / computation method above.
#

def relax_sced_ramp_rates(sced_instance, scale_factor): 
    #                          nominal_up_gens_violated, nominal_down_gens_violated,
    #                          startup_gens_violated, shutdown_gens_violated):

    #    for g in nominal_up_gens_violated:
    #        sced_instance.NominalRampUpLimit[g] = value(sced_instance.NominalRampUpLimit[g]) * (1.0 + scale_factor)

    #    for g in nominal_down_gens_violated:
    #        sced_instance.NominalRampDownLimit[g] = value(sced_instance.NominalRampDownLimit[g]) * (1.0 + scale_factor)

    #    for g in startup_gens_violated:
    #        sced_instance.StartupRampLimit[g] = value(sced_instance.StartupRampLimit[g]) * (1.0 + scale_factor)

    #    for g in shutdown_gens_violated:
    #        sced_instance.ShutdownRampLimit[g] = value(sced_instance.ShutdownRampLimit[g]) * (1.0 + scale_factor)

    # COMMENT ON THE FACT THAT RAMP-DOWN AND SHUT-DOWN ARE THE ONLY ISSUES
    for g in sced_instance.ThermalGenerators:
        # sced_instance.NominalRampUpLimit[g] = value(sced_instance.NominalRampUpLimit[g]) * (1.0 + scale_factor)
        sced_instance.NominalRampDownLimit[g] = value(sced_instance.NominalRampDownLimit[g]) * (1.0 + scale_factor)        
        # sced_instance.StartupRampLimit[g] = value(sced_instance.StartupRampLimit[g]) * (1.0 + scale_factor)
        sced_instance.ShutdownRampLimit[g] = value(sced_instance.ShutdownRampLimit[g]) * (1.0 + scale_factor)

        # doing more work than strictly necessary here, but SCED isn't a bottleneck.
    # sced_instance.ScaledNominalRampUpLimit.reconstruct()
    sced_instance.ScaledNominalRampDownLimit.reconstruct()
    # sced_instance.ScaledStartupRampLimit.reconstruct()
    sced_instance.ScaledShutdownRampLimit.reconstruct()

    # sced_instance.EnforceMaxAvailableRampUpRates.reconstruct()
    # sced_instance.EnforceMaxAvailableRampDownRates.reconstruct()
    sced_instance.EnforceScaledNominalRampDownLimits.reconstruct()
    
    sced_instance.preprocess()


def take_generator_offline(model, instance, generator):
    if generator not in offline_generators:
        offline_generators.append(generator)
        for t in instance.TimePeriods:
            instance.UnitOn[generator,t] = 0
            instance.UnitOn[generator,t].fixed = True
            instance.PowerGenerated[generator,t] = 0
            instance.PowerGenerated[generator,t].fixed = True
            instance.UnitOnT0[generator] = 0
            instance.PowerGeneratedT0[generator] = 0
            if model == "multiSUSD":
                instance.UnitOff[generator, t] = 1
                instance.UnitOff[generator, t].fixed = True
                instance.UnitOffT0[generator] = 1
            else:
                instance.MaximumPowerAvailable[generator,t] = 0
                instance.MaximumPowerAvailable[generator,t].fixed = True


def reset_offline_elements(model, instance):
    for g in offline_generators:
        for t in instance.TimePeriods:
            #print "Resetting UnitOn and PowerGenerated for",g,t
            instance.UnitOn[g,t].fixed = False
            instance.PowerGenerated[g,t].fixed = False
            if model == "multiSUSD":
                instance.UnitOff[g, t].fixed = False
            else:
                instance.MaximumPowerAvailable[g,t].fixed = False

########################################################
# utility functions for reporting various aspects of a #
# multi-period SCED solution.                          #
########################################################

def report_costs_for_deterministic_sced(instance):

    # only worry about two-stage models for now..
    fixed_cost = value(sum(instance.StartupCost[g,1] + instance.ShutdownCost[g,1] for g in instance.ThermalGenerators) + \
                       sum(instance.UnitOn[g,1] * instance.MinimumProductionCost[g] * instance.TimePeriodLength for g in instance.ThermalGenerators))
    variable_cost = value(instance.TotalProductionCost[1])
    print("Fixed costs:    %12.2f" % fixed_cost)
    print("Variable costs: %12.2f" % variable_cost)
    return fixed_cost, variable_cost

def report_mismatches_for_deterministic_sced(instance):

    issues_found = False

    load_generation_mismatch_value = round_small_values(sum(value(instance.LoadGenerateMismatch[b, 1])
                                                            for b in instance.Buses))
    if load_generation_mismatch_value != 0.0:
        issues_found = True
        load_shedding_value = round_small_values(sum(value(instance.posLoadGenerateMismatch[b, 1])
                                                     for b in instance.Buses))
        over_generation_value = round_small_values(sum(value(instance.negLoadGenerateMismatch[b, 1])
                                                       for b in instance.Buses))
        if load_shedding_value != 0.0:
            print("Load shedding reported at t=%d -     total=%12.2f" % (1, load_shedding_value))
        if over_generation_value != 0.0:
            print("Over-generation reported at t=%d -   total=%12.2f" % (1, over_generation_value))
    else:
        load_shedding_value = 0.0
        over_generation_value = 0.0

    available_quick_start = available_quick_start_for_deterministic_sced(instance)

    reserve_shortfall_value = round_small_values(value(instance.ReserveShortfall[1]))
    if reserve_shortfall_value != 0.0:
        issues_found = True
        print("Reserve shortfall reported at t=%2d: %12.2f" % (1, reserve_shortfall_value))
        # report if quick start generation is available during reserve shortfalls in SCED
        print("Quick start generation capacity available at t=%2d: %12.2f" % (1, available_quick_start))
        print("")

    available_reserve = sum(value(instance.MaximumPowerAvailable[g, 1]) - value(instance.PowerGenerated[g, 1])
                            for g in instance.ThermalGenerators)

    if issues_found:
        pass
#        print "***ISSUES FOUND***"
#        instance.ReserveRequirement.pprint()
#        instance.LoadGenerateMismatch.pprint()
#        instance.ReserveMismatch.pprint()
#        instance.MaximumPowerAvailable.pprint()
#        instance.PowerGenerated.pprint()
    
    # order of return values is: load-shedding, over-generation, reserve-shortfall, available_reserve
    return load_shedding_value, over_generation_value, reserve_shortfall_value, available_reserve, available_quick_start

def report_lmps_for_deterministic_sced(instance, 
                                       max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print(("%-"+str(max_bus_label_length)+"s %14s") % ("Bus", "Computed LMP"))
    for bus in instance.Buses:
        print(("%-"+str(max_bus_label_length)+"s %14.6f") % (bus, instance.dual[instance.PowerBalance[bus,1]]))    

def report_at_limit_lines_for_deterministic_sced(instance,
                                                 max_line_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")

    lines_at_limit = []

    for l in sorted(instance.TransmissionLines):
        if value(instance.LinePower[l,1]) < 0.0: 
            if instance.ThermalLimit[l] - -1.0 * value(instance.LinePower[l,1]) <= 1e-5:
                lines_at_limit.append((l, value(instance.LinePower[l,1])))
        else:
            if instance.ThermalLimit[l] - value(instance.LinePower[l,1]) <= 1e-5:
                lines_at_limit.append((l, value(instance.LinePower[l,1])))

    if len(lines_at_limit) == 0:
        print("No lines were at thermal limits")
    else:
        print(("%-"+str(max_line_label_length)+"s %14s") % ("Line at thermal limit","Flow"))
        for line, flow in lines_at_limit:
            print(("%-"+str(max_line_label_length)+"s %14.6f") % (line, flow))

def available_quick_start_for_deterministic_sced(instance):
    """Given a SCED instance with commitments from the RUC,
    determine how much quick start capacity is available 
    """
    available_quick_start_capacity = 0.0 
    for g in instance.QuickStartGenerators:
        available = True  # until proven otherwise
        if int(round(value(instance.UnitOn[g, 1]))) == 1:
            available = False  # unit was already committed in the RUC
        elif instance.MinimumDownTime[g] > 1:
            # minimum downtime should be 1 or less, by definition of a quick start
            available = False
        elif (value(instance.UnitOnT0[g]) - int(round(value(instance.UnitOn[g, 1])))) == 1:
            # there cannot have been a a shutdown in the previous hour 
            available = False
 
        if available:  # add the amount of power that can be accessed in the first hour
            # use the min() because scaled startup ramps can be larger than the generator limit
            available_quick_start_capacity += min(value(instance.ScaledStartupRampLimit[g]), value(instance.MaximumPowerOutput[g]))
        
    return available_quick_start_capacity


def report_renewables_curtailment_for_deterministic_sced(instance):

    # we only extract curtailment statistics for time period 1

    total_curtailment = round_small_values(sum((value(instance.MaxNondispatchablePower[g, 1]) -
                                                value(instance.NondispatchablePowerUsed[g, 1]))
                                               for g in instance.AllNondispatchableGenerators))

    if total_curtailment > 0:
        print("")
        print("Renewables curtailment reported at t=%d - total=%12.2f" % (1, total_curtailment))

    return total_curtailment



def report_on_off_and_ramps_for_deterministic_sced(instance):

    num_on_offs = 0
    sum_on_off_ramps = 0.0
    sum_nominal_ramps = 0.0  # this is the total ramp change for units not switching on/off
    
    for g in instance.ThermalGenerators:
        unit_on = int(round(value(instance.UnitOn[g, 1])))
        power_generated = value(instance.PowerGenerated[g, 1])
        if value(instance.UnitOnT0State[g]) > 0:
            # unit was on in previous time period
            if unit_on:
                # no change in state
                sum_nominal_ramps += math.fabs(power_generated - value(instance.PowerGeneratedT0[g]))
            else:
                num_on_offs += 1
                sum_on_off_ramps += power_generated
        else: # value(instance.UnitOnT0State[g]) < 0
            # unit was off in previous time period
            if not unit_on:
                # no change in state
                sum_nominal_ramps += math.fabs(power_generated - value(instance.PowerGeneratedT0[g]))
            else:
                num_on_offs += 1
                sum_on_off_ramps += power_generated

    print("")
    print("Number on/offs:       %12d" % num_on_offs)
    print("Sum on/off ramps:     %12.2f" % sum_on_off_ramps)
    print("Sum nominal ramps:    %12.2f" % sum_nominal_ramps) 

    return num_on_offs, sum_on_off_ramps, sum_nominal_ramps

###################################################################
# utility functions for computing and reporting various aspects   #
# of a deterministic unit commitment solution.                    #
###################################################################

def output_solution_for_deterministic_ruc(ruc_instance, 
                                          this_date,
                                          max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Generator Commitments:")
    for g in sorted(ruc_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(1, 37):
            print("%2d"% int(round(value(ruc_instance.UnitOn[g, t]))), end=' ')
            if t == 24: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Dispatch Levels:")
    for g in sorted(ruc_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(1, 37):
            print("%7.2f"% value(ruc_instance.PowerGenerated[g,t]), end=' ')
            if t == 24: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Reserve Headroom:")
    total_headroom = [0.0 for i in range(0, 37)]  # add the 0 in for simplicity of indexing
    for g in sorted(ruc_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(1, 37):
            headroom = math.fabs(value(ruc_instance.MaximumPowerAvailable[g, t]) -
                                 value(ruc_instance.PowerGenerated[g, t]))
            print("%7.2f" % headroom, end=' ')
            total_headroom[t] += headroom
            if t == 24: 
                print(" |", end=' ')
        print("")
    print(("%-"+str(max_thermal_generator_label_length)+"s: ") % "Total", end=' ')
    for t in range(1, 37):
        print("%7.2f" % total_headroom[t], end=' ')
    print("")

    if len(ruc_instance.Storage) > 0:
        
        directory = "./deterministic_simple_storage"
        if not os.path.exists(directory):
            os.makedirs(directory)
            csv_hourly_output_filename = os.path.join(directory, "Storage_summary_for_" + str(this_date) + ".csv")
        else:
            csv_hourly_output_filename = os.path.join(directory, "Storage_summary_for_" + str(this_date) + ".csv")

        csv_hourly_output_file = open(csv_hourly_output_filename, "w")
        
        print("Storage Input levels")
        for s in sorted(ruc_instance.Storage):
            print("%30s: " % s, end=' ')
            print(s, end=' ', file=csv_hourly_output_file)
            print("Storage Input:", file=csv_hourly_output_file)
            for t in range(1, 27):
                print("%7.2f"% value(ruc_instance.PowerInputStorage[s,t]), end=' ')
                print("%7.2f"% value(ruc_instance.PowerInputStorage[s,t]), end=' ', file=csv_hourly_output_file)
                if t == 24: 
                    print(" |", end=' ')
                    print(" |", end=' ', file=csv_hourly_output_file)
            print("", file=csv_hourly_output_file)
            print("")

        print("Storage Output levels")
        for s in sorted(ruc_instance.Storage):
            print("%30s: " % s, end=' ')
            print(s, end=' ', file=csv_hourly_output_file)
            print("Storage Output:", file=csv_hourly_output_file)
            for t in range(1,27):
                print("%7.2f"% value(ruc_instance.PowerOutputStorage[s,t]), end=' ', file=csv_hourly_output_file)
                print("%7.2f"% value(ruc_instance.PowerOutputStorage[s,t]), end=' ')
                if t == 24: 
                    print(" |", end=' ')
                    print(" |", end=' ', file=csv_hourly_output_file)
            print("", file=csv_hourly_output_file)
            print("")

        print("Storage SOC levels")
        for s in sorted(ruc_instance.Storage):
            print("%30s: " % s, end=' ')
            print(s, end=' ', file=csv_hourly_output_file)
            print("Storage SOC:", file=csv_hourly_output_file)
            for t in range(1,27):
                print("%7.2f"% value(ruc_instance.SocStorage[s,t]), end=' ', file=csv_hourly_output_file)
                print("%7.2f"% value(ruc_instance.SocStorage[s,t]), end=' ')
                if t == 24: 
                    print(" |", end=' ')
                    print(" |", end=' ', file=csv_hourly_output_file)
            print("", file=csv_hourly_output_file)
            print("")


def report_demand_for_deterministic_ruc(ruc_instance,
                                        max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print("Projected Demand:")
    for b in sorted(ruc_instance.Buses):
        print(("%-"+str(max_bus_label_length)+"s: ") % b, end=' ')
        for t in range(1,37):
            print("%8.2f"% value(ruc_instance.Demand[b,t]), end=' ')
            if t == 24: 
                print(" |", end=' ')
        print("")

def report_initial_conditions_for_deterministic_ruc(deterministic_instance,
                                                    max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated must-run):")

    for g in sorted(deterministic_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s %5d %5d %12.2f %6d") % 
              (g, 
               value(deterministic_instance.UnitOnT0[g]),
               value(deterministic_instance.UnitOnT0State[g]), 
               value(deterministic_instance.PowerGeneratedT0[g]),
               value(deterministic_instance.MustRun[g])))

    # it is generally useful to know something about the bounds on output capacity
    # of the thermal fleet from the initial condition to the first time period. 
    # one obvious use of this information is to aid analysis of infeasibile
    # initial conditions, which occur rather frequently when hand-constructing
    # instances.

    # output the total amount of power generated at T0
    total_t0_power_output = 0.0
    for g in sorted(deterministic_instance.ThermalGenerators):    
        total_t0_power_output += value(deterministic_instance.PowerGeneratedT0[g])
    print("")
    print("Power generated at T0=%8.2f" % total_t0_power_output)
    
    # compute the amount of new generation that can be brought on-line the first period.
    total_new_online_capacity = 0.0
    for g in sorted(deterministic_instance.ThermalGenerators):
        t0_state = value(deterministic_instance.UnitOnT0State[g])
        if t0_state < 0: # the unit has been off
            if int(math.fabs(t0_state)) >= value(deterministic_instance.MinimumDownTime[g]):
                total_new_online_capacity += min(value(deterministic_instance.StartupRampLimit[g]), value(deterministic_instance.MaximumPowerOutput[g]))
    print("")
    print("Total capacity at T=1 available to add from newly started units=%8.2f" % total_new_online_capacity)

    # compute the amount of generation that can be brough off-line in the first period
    # (to a shut-down state)
    total_new_offline_capacity = 0.0
    for g in sorted(deterministic_instance.ThermalGenerators):
        t0_state = value(deterministic_instance.UnitOnT0State[g])
        if t0_state > 0: # the unit has been on
            if t0_state >= value(deterministic_instance.MinimumUpTime[g]):
                if value(deterministic_instance.PowerGeneratedT0[g]) <= value(deterministic_instance.ShutdownRampLimit[g]):
                    total_new_offline_capacity += value(deterministic_instance.PowerGeneratedT0[g])
    print("")
    print("Total capacity at T=1 available to drop from newly shut-down units=%8.2f" % total_new_offline_capacity)

# NOTE: CommitmentStageCost variables track: startup, shutdown, and minimum production costs.

def compute_fixed_costs_for_deterministic_ruc(deterministic_instance):

    return sum(value(deterministic_instance.CommitmentStageCost[stage]) for stage in deterministic_instance.StageSet)

def report_fixed_costs_for_deterministic_ruc(deterministic_instance):

    second_stage = "Stage_1" # TBD - data-drive this - maybe get it from the scenario tree? StageSet should also be ordered in the UC models.

    print("Fixed costs:    %12.2f" % value(deterministic_instance.CommitmentStageCost[second_stage]))

def report_generation_costs_for_deterministic_ruc(deterministic_instance):

    # only worry about two-stage models for now..
    second_stage = "Stage_2" # TBD - data-drive this - maybe get it from the scenario tree? StageSet should also be ordered in the UC models.

    print("Variable costs: %12.2f" % value(deterministic_instance.GenerationStageCost[second_stage]))

def report_load_generation_mismatch_for_deterministic_ruc(ruc_instance):

    for t in sorted(ruc_instance.TimePeriods):
        mismatch_reported = False
        sum_mismatch = round_small_values(sum(value(ruc_instance.LoadGenerateMismatch[b, t])
                                              for b in ruc_instance.Buses))
        if sum_mismatch != 0.0:
            posLoadGenerateMismatch = round_small_values(sum(value(ruc_instance.posLoadGenerateMismatch[b, t])
                                                             for b in ruc_instance.Buses))
            negLoadGenerateMismatch = round_small_values(sum(value(ruc_instance.negLoadGenerateMismatch[b, t])
                                                             for b in ruc_instance.Buses))
            if negLoadGenerateMismatch != 0.0:
                print("Projected over-generation reported at t=%d -   total=%12.2f" % (t, negLoadGenerateMismatch))
                mismatch_reported = True
            if posLoadGenerateMismatch != 0.0:
                print("Projected load shedding reported at t=%d -     total=%12.2f" % (t, posLoadGenerateMismatch))
                mismatch_reported = True

        reserve_shortfall_value = round_small_values(value(ruc_instance.ReserveShortfall[t]))
        if reserve_shortfall_value != 0.0:
            print("Projected reserve shortfall reported at t=%d - total=%12.2f" % (t, reserve_shortfall_value))
            mismatch_reported = True

        if mismatch_reported:

            print("")
            print("Dispatch detail for time period=%d" % t)
            total_generated = 0.0
            for g in sorted(ruc_instance.ThermalGenerators):
                unit_on = int(round(value(ruc_instance.UnitOn[g, t])))
                print("%-30s %2d %12.2f %12.2f" % (g, 
                                                   unit_on, 
                                                   value(ruc_instance.PowerGenerated[g,t]),
                                                   value(ruc_instance.MaximumPowerAvailable[g,t]) - value(ruc_instance.PowerGenerated[g,t])), end=' ')
                if (unit_on == 1) and (math.fabs(value(ruc_instance.PowerGenerated[g,t]) -
                                                 value(ruc_instance.MaximumPowerOutput[g])) <= 1e-5): 
                    print(" << At max output", end=' ')
                elif (unit_on == 1) and (math.fabs(value(ruc_instance.PowerGenerated[g,t]) -
                                                   value(ruc_instance.MinimumPowerOutput[g])) <= 1e-5): 
                    print(" << At min output", end=' ')
                if value(ruc_instance.MustRun[g]):
                    print(" ***", end=' ')
                print("")
                if unit_on == 1:
                    total_generated += value(ruc_instance.PowerGenerated[g,t])
            print("")
            print("Total power dispatched=%7.2f" % total_generated)

###################################################################
# utility functions for computing and reporting various aspects   #
# of a stochastic extensive form unit commitment solution.        #
###################################################################

def output_solution_for_stochastic_ruc(scenario_instances, scenario_tree, output_scenario_dispatches=False):

    # we can grab solution information either from the scenario tree, or 
    # from the instances themselves. by convention, we are grabbing values
    # from the scenario tree in cases where multiple instances are involved
    # (e.g., at a root node, where variables are blended), and otherwise
    # from the instances themselves.

    print("Generator Commitments:")
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]
    root_node = scenario_tree.findRootNode()
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        print("%30s: " % g, end=' ')
        for t in range(1, 37):
            raw_value = root_node.get_variable_value("UnitOn", (g, t))
            if raw_value is None:
                raise RuntimeError("***Failed to extract value for variable UnitOn, index=" + str((g, t)) +
                                   " from scenario tree.")
            print("%2d"% int(round(raw_value)), end=' ')
            if t == 24: 
                print(" |", end=' ')
        print("")

    if output_scenario_dispatches:
        scenario_names = sorted(scenario_instances.keys())
        for scenario_name in scenario_names:
            scenario_instance = scenario_instances[scenario_name]
            print("")
            print("Generator Dispatch Levels for Scenario=" + scenario_name)
            for g in sorted(scenario_instance.ThermalGenerators):
                print("%-30s: " % g, end=' ')
                for t in range(1, 25):
                    print("%6.2f"% value(scenario_instance.PowerGenerated[g, t]), end=' ')
                print("")

            print("")
            print("Generator Production Costs Scenario=" + scenario_name)
            for g in sorted(scenario_instance.ThermalGenerators):
                print("%30s: " % g, end=' ')
                for t in range(1, 25):
                    print("%8.2f"% value(scenario_instance.ProductionCost[g,t]), end=' ')
                print("")
            print("%30s: " % "Total", end=' ')
            for t in range(1, 25):
                sum_production_costs = sum(value(scenario_instance.ProductionCost[g, t])
                                           for g in scenario_instance.ThermalGenerators)
                print("%8.2f"% sum_production_costs, end=' ')
            print("")


def compute_fixed_costs_for_stochastic_ruc(stochastic_instance, scenario_instances):

    # we query on an arbitrary scenario instance, because the 
    # input instance / solution is assumed to be non-anticipative.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]
    
    return sum(value(arbitrary_scenario_instance.CommitmentStageCost[stage])
               for stage in arbitrary_scenario_instance.StageSet)


def report_initial_conditions_for_stochastic_ruc(scenario_instances):

    # we query on an arbitrary scenario instance, because the 
    # initial conditions are assumed to be non-anticipative.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]

    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated):")
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        print("%-30s %5d %5d %12.2f" % (g, 
                                        value(arbitrary_scenario_instance.UnitOnT0[g]), 
                                        value(arbitrary_scenario_instance.UnitOnT0State[g]), 
                                        value(arbitrary_scenario_instance.PowerGeneratedT0[g]))) 


    # it is generally useful to know something about the bounds on output capacity
    # of the thermal fleet from the initial condition to the first time period. 
    # one obvious use of this information is to aid analysis of infeasibile
    # initial conditions, which occur rather frequently when hand-constructing
    # instances.

    # output the total amount of power generated at T0
    total_t0_power_output = 0.0
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):    
        total_t0_power_output += value(arbitrary_scenario_instance.PowerGeneratedT0[g])
    print("")
    print("Power generated at T0=%8.2f" % total_t0_power_output)
    
    # compute the amount of new generation that can be brought on-line the first period.
    total_new_online_capacity = 0.0
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        t0_state = value(arbitrary_scenario_instance.UnitOnT0State[g])
        if t0_state < 0: # the unit has been off
            if int(math.fabs(t0_state)) >= value(arbitrary_scenario_instance.MinimumDownTime[g]):
                total_new_online_capacity += min(value(arbitrary_scenario_instance.StartupRampLimit[g]),
                                                 value(arbitrary_scenario_instance.MaximumPowerOutput[g]))
    print("")
    print("Total capacity at T=1 available to add from newly started units=%8.2f" % total_new_online_capacity)

    # compute the amount of generation that can be brough off-line in the first period
    # (to a shut-down state)
    total_new_offline_capacity = 0.0
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        t0_state = value(arbitrary_scenario_instance.UnitOnT0State[g])
        if t0_state > 0: # the unit has been on
            if t0_state >= value(arbitrary_scenario_instance.MinimumUpTime[g]):
                if value(arbitrary_scenario_instance.PowerGeneratedT0[g]) <= \
                        value(arbitrary_scenario_instance.ShutdownRampLimit[g]):
                    total_new_offline_capacity += value(arbitrary_scenario_instance.PowerGeneratedT0[g])
    print("")
    print("Total capacity at T=1 available to drop from newly shut-down units=%8.2f" % total_new_offline_capacity)


def report_fixed_costs_for_stochastic_ruc(scenario_instances):

    # we query on an arbitrary scenario instance, because the 
    # input instance / solution is assumed to be non-anticipative.
    instance = scenario_instances[list(scenario_instances.keys())[0]]

    # no fixed costs to output for stage 2 as of yet.
    stage = "Stage_1"
    print("Fixed cost for stage %s:     %10.2f" % (stage, value(instance.CommitmentStageCost[stage])))

    # break out the startup, shutdown, and minimum-production costs, for reporting purposes.
    startup_costs = sum(value(instance.StartupCost[g,t])
                        for g in instance.ThermalGenerators for t in instance.CommitmentTimeInStage[stage])
    shutdown_costs = sum(value(instance.ShutdownCost[g,t])
                         for g in instance.ThermalGenerators for t in instance.CommitmentTimeInStage[stage])
    minimum_generation_costs = sum(sum(value(instance.UnitOn[g,t])
                                       for t in instance.CommitmentTimeInStage[stage]) *
                                   value(instance.MinimumProductionCost[g]) *
                                   value(instance.TimePeriodLength) for g in instance.ThermalGenerators)

    print("   Startup costs:                 %10.2f" % startup_costs)
    print("   Shutdown costs:                %10.2f" % shutdown_costs)
    print("   Minimum generation costs:      %10.2f" % minimum_generation_costs)


def report_generation_costs_for_stochastic_ruc(scenario_instances):

    # load shedding is scenario-dependent, so we have to loop through each scenario instance.
    # but we will in any case use an arbitrary scenario instance to access index sets.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]

    # only worry about two-stage models for now..
    second_stage = "Stage_2"  # TBD - data-drive this - maybe get it from the scenario tree? StageSet should also be ordered in the UC models.

    print("Generation costs (Scenario, Cost):")
    for scenario_name in sorted(scenario_instances.keys()):
        scenario_instance = scenario_instances[scenario_name]
        # print("%-15s %12.2f" % (scenario_name, scenario_instance.GenerationStageCost[second_stage]))
        print("%-15s %12.2f" % (scenario_name, value(scenario_instance.GenerationStageCost[second_stage])))


def report_load_generation_mismatch_for_stochastic_ruc(scenario_instances):

    # load shedding is scenario-dependent, so we have to loop through each scenario instance.
    # but we will in any case use an arbitrary scenario instance to access index sets.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]

    for t in sorted(arbitrary_scenario_instance.TimePeriods):
        for scenario_name, scenario_instance in scenario_instances.items():

            sum_mismatch = round_small_values(sum(value(scenario_instance.LoadGenerateMismatch[b,t])
                                                  for b in scenario_instance.Buses))
            if sum_mismatch != 0.0:
                posLoadGenerateMismatch = round_small_values(sum(value(scenario_instance.posLoadGenerateMismatch[b, t])
                                                                 for b in scenario_instance.Buses))
                negLoadGenerateMismatch = round_small_values(sum(value(scenario_instance.negLoadGenerateMismatch[b, t])
                                                                 for b in scenario_instance.Buses))
                if posLoadGenerateMismatch != 0.0:
                    print("Projected load shedding reported at t=%d -     total=%12.2f - scenario=%s"
                          % (t, posLoadGenerateMismatch, scenario_name))
                if negLoadGenerateMismatch != 0.0:
                    print("Projected over-generation reported at t=%d -   total=%12.2f - scenario=%s"
                          % (t, negLoadGenerateMismatch, scenario_name))

            reserve_shortfall_value = round_small_values(value(scenario_instance.ReserveShortfall[t]))
            if reserve_shortfall_value != 0.0:
                print("Projected reserve shortfall reported at t=%d - total=%12.2f - scenario=%s"
                      % (t, reserve_shortfall_value, scenario_name))

#####################################################################
# utility functions for pretty-printing solutions, for SCED and RUC #
#####################################################################

def output_sced_initial_condition(sced_instance, hour=1, max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated t1-unit-on must-run):")
    for g in sorted(sced_instance.ThermalGenerators):
        if hour == 1:
            print(("%-"+str(max_thermal_generator_label_length)+"s %5d %12.2f %5d %6d") % 
                  (g, 
                   value(sced_instance.UnitOnT0[g]), 
                   value(sced_instance.PowerGeneratedT0[g]),
                   value(sced_instance.UnitOn[g,hour]), 
                   value(sced_instance.MustRun[g])))
        else:
            print(("%-"+str(max_thermal_generator_label_length)+"s %5d %12.2f %5d %6d") % 
                  (g, 
                   value(sced_instance.UnitOn[g,hour-1]), 
                   value(sced_instance.PowerGenerated[g,hour-1]),
                   value(sced_instance.UnitOn[g,hour]), 
                   value(sced_instance.MustRun[g])))

def output_sced_demand(sced_instance, max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Demand detail:")
    for b in sorted(sced_instance.Buses):
        print(("%-"+str(max_bus_label_length)+"s %12.2f") % 
              (b, 
               value(sced_instance.Demand[b, 1])))

    print("")
    print(("%-"+str(max_bus_label_length)+"s %12.2f") % 
          ("Reserve requirement:", 
           value(sced_instance.ReserveRequirement[1])))

    print("")
    print("Maximum non-dispatachable power available:")
    for b in sorted(sced_instance.Buses):
        total_max_nondispatchable_power = sum(value(sced_instance.MaxNondispatchablePower[g, 1])
                                              for g in sced_instance.NondispatchableGeneratorsAtBus[b])
        print("%-30s %12.2f" % (b, total_max_nondispatchable_power))

    print("")
    print("Minimum non-dispatachable power available:")
    for b in sorted(sced_instance.Buses):
        total_min_nondispatchable_power = sum(value(sced_instance.MinNondispatchablePower[g, 1])
                                              for g in sced_instance.NondispatchableGeneratorsAtBus[b])
        print("%-30s %12.2f" % (b, total_min_nondispatchable_power))

def output_sced_solution(sced_instance, max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Solution detail:")
    print("")
    print("Dispatch Levels (unit-on, power-generated, reserve-headroom)")
    for g in sorted(sced_instance.ThermalGenerators):
        unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
        print(("%-"+str(max_thermal_generator_label_length)+"s %2d %12.2f %12.2f") % 
              (g, 
               unit_on, 
               value(sced_instance.PowerGenerated[g, 1]),
               math.fabs(value(sced_instance.MaximumPowerAvailable[g,1]) -
                         value(sced_instance.PowerGenerated[g,1]))), 
              end=' ')
        if (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                         value(sced_instance.MaximumPowerOutput[g])) <= 1e-5): 
            print(" << At max output", end=' ')
        elif (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                         value(sced_instance.MinimumPowerOutput[g])) <= 1e-5): 
            print(" << At min output", end=' ')
        if value(sced_instance.MustRun[g]):
            print(" ***", end=' ')
        print("")

    print("")
    print("Total power dispatched      = %12.2f"
          % sum(value(sced_instance.PowerGenerated[g,1]) for g in sced_instance.ThermalGenerators))
    print("Total reserve available     = %12.2f"
          % sum(value(sced_instance.MaximumPowerAvailable[g,1]) - value(sced_instance.PowerGenerated[g,1])
                for g in sced_instance.ThermalGenerators))
    print("Total quick start available = %12.2f"
          % available_quick_start_for_deterministic_sced(sced_instance))
    print("")

    print("Cost Summary (unit-on production-cost no-load-cost startup-cost)")
    total_startup_costs = 0.0
    for g in sorted(sced_instance.ThermalGenerators):
        unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
        unit_on_t0 = int(round(value(sced_instance.UnitOnT0[g])))
        startup_cost = 0.0
        if unit_on_t0 == 0 and unit_on == 1:
            startup_cost = value(sced_instance.StartupCost[g,1])
        total_startup_costs += startup_cost
        print(("%-"+str(max_thermal_generator_label_length)+"s %2d %12.2f %12.2f %12.2f") % 
              (g, 
               unit_on, 
               value(sced_instance.ProductionCost[g, 1]), 
               unit_on * value(sced_instance.MinimumProductionCost[g]), 
               startup_cost), 
              end=' ')
        if (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                         value(sced_instance.MaximumPowerOutput[g])) <= 1e-5): 
            print(" << At max output", end=' ')
        elif (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                 value(sced_instance.MinimumPowerOutput[g])) <= 1e-5):  # TBD - still need a tolerance parameter
            print(" << At min output", end=' ')
        print("")
        
    print("")
    print("Total cost = %12.2f" % (value(sced_instance.TotalNoLoadCost[1]) + value(sced_instance.TotalProductionCost[1]) +
                                   total_startup_costs))

# useful in cases where ramp rate constraints have been violated.
# ramp rates are taken from the original sced instance. unit on
# values can be taken from either instance, as they should be the 
# same. power generated values must be taken from the relaxed instance.

def output_sced_ramp_violations(original_sced_instance, relaxed_sced_instance):

    # we are assuming that there are only a handful of violations - if that
    # is not the case, we should shift to some kind of table output format.
    for g in original_sced_instance.ThermalGenerators:
        for t in original_sced_instance.TimePeriods:

            current_unit_on = int(round(value(original_sced_instance.UnitOn[g, t])))

            if t == 1:
                previous_unit_on = int(round(value(original_sced_instance.UnitOnT0[g])))
            else:
                previous_unit_on = int(round(value(original_sced_instance.UnitOn[g, t-1])))                                                                        

            if current_unit_on == 0:
                if previous_unit_on == 0:
                    # nothing is on, nothing to worry about!
                    pass
                else:
                    # the unit is switching off.
                    # TBD - eventually deal with shutdown ramp limits
                    pass

            else:
                if previous_unit_on == 0:
                    # the unit is switching on.
                    # TBD - eventually deal with startup ramp limits
                    pass
                else:
                    # the unit is remaining on.
                    if t == 1:
                        delta_power = value(relaxed_sced_instance.PowerGenerated[g, t]) - value(relaxed_sced_instance.PowerGeneratedT0[g])
                    else:
                        delta_power = value(relaxed_sced_instance.PowerGenerated[g, t]) - value(relaxed_sced_instance.PowerGenerated[g, t-1])
                    if delta_power > 0.0:
                        # the unit is ramping up
                        if delta_power > value(original_sced_instance.NominalRampUpLimit[g]):
                            print("Thermal unit=%s violated nominal ramp up limits from time=%d to time=%d - observed delta=%f, nominal limit=%f"
                                  % (g, t-1, t, delta_power, value(original_sced_instance.NominalRampUpLimit[g])))
                    else:
                        # the unit is ramping down
                        if math.fabs(delta_power) > value(original_sced_instance.NominalRampDownLimit[g]):
                            print("Thermal unit=%s violated nominal ramp down limits from time=%d to time=%d - observed delta=%f, nominal limit=%f"
                                  % (g, t-1, t, math.fabs(delta_power), value(original_sced_instance.NominalRampDownLimit[g])))


def solve_extensive_form(solver, stochastic_instance, options, solve_options, tee=False):

    results = call_solver(solver, stochastic_instance, tee=tee, **solve_options[solver])
    stochastic_instance.solutions.load_from(results)

# utiltiies for creating a deterministic RUC instance, and a standard way to solve them.

def create_deterministic_ruc(options, 
                             this_date, 
                             prior_deterministic_ruc, # UnitOn T0 state should come from here
                             projected_sced, # PowerGenerated T0 state should come from here
                             output_initial_conditions,
                             sced_midnight_hour=4,
                             max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH,
                             max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    # the (1-based) time period in the sced corresponding to midnight

    instance_directory_name = os.path.join(options.data_directory, "pyspdir_twostage") 

    # first, create the scenario tree - this is still needed, for various reasons, in the
    # master simulation routine.

    # IMPORTANT: "scenario_tree_model" is a *function* that returns an abstract model - the
    #            naming convention does not necessarily reflect this fact...
    from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
    scenario_tree_instance_filename = os.path.join(os.path.expanduser(instance_directory_name), 
                                                   this_date, 
                                                   "ScenarioStructure.dat")

    # ignoring bundles for now - for the deterministic case, we don't care.
    scenario_tree_instance = CreateAbstractScenarioTreeModel().create_instance(filename=scenario_tree_instance_filename)

    new_scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_instance)

    # next, construct the RUC instance. for data, we always look in the pysp directory
    # for the forecasted value instance.
    reference_model_filename = os.path.expanduser(options.model_directory) + os.sep + "ReferenceModel.py"
    reference_model_module = pyutilib.misc.import_file(reference_model_filename)
    reference_model = reference_model_module.model
    new_deterministic_ruc = reference_model.create_instance(os.path.join(os.path.expanduser(instance_directory_name), 
                                                                         this_date, 
                                                                         "Scenario_forecasts.dat"))

    if max_thermal_generator_label_length == DEFAULT_MAX_LABEL_LENGTH:
        if len(new_deterministic_ruc.ThermalGenerators) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max((len(this_generator) for this_generator in new_deterministic_ruc.ThermalGenerators))        

    if max_bus_label_length == DEFAULT_MAX_LABEL_LENGTH:
        max_bus_label_length = max((len(this_bus) for this_bus in new_deterministic_ruc.Buses))    

    # set up the initial conditions for the deterministic RUC, based on all information
    # available as to the conditions at the initial day. if this is the first simulation
    # day, then we have nothing to go on other than the initial conditions already
    # specified in the forecasted value instance. however, if this is not the first simulation
    # day, use the projected solution state embodied (either explicitly or implicitly)
    # in the prior date RUC to establish the initial conditions.
    if (prior_deterministic_ruc != None) and (projected_sced != None):

        for g in sorted(new_deterministic_ruc.ThermalGenerators):

            final_unit_on_state = int(round(value(prior_deterministic_ruc.UnitOn[g, 24])))
            state_duration = 1
            hours = list(range(1, 24))
            hours.reverse()
            for i in hours:
                this_unit_on_state = int(round(value(prior_deterministic_ruc.UnitOn[g,i])))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration

            # power generated is the projected output at midnight.
            power_generated_at_midnight = value(projected_sced.PowerGenerated[g, sced_midnight_hour])

            # on occasion, the average power generated across scenarios for a single generator
            # can be a very small negative number, due to MIP tolerances allowing it. if this
            # is the case, simply threshold it to 0.0. similarly, the instance validator will
            # fail if the average power generated is small-but-positive (e.g., 1e-14) and the
            # UnitOnT0 state is Off. in the latter case, just set the average power to 0.0.
            if power_generated_at_midnight < 0.0:
                power_generated_at_midnight = 0.0
            elif final_unit_on_state == 0:
                power_generated_at_midnight = 0.0

            # propagate the initial conditions to the deterministic ruc being constructed.

            new_deterministic_ruc.UnitOnT0[g] = final_unit_on_state
            new_deterministic_ruc.UnitOnT0State[g] = state_duration

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization 
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.
            min_power_output = value(new_deterministic_ruc.MinimumPowerOutput[g])
            max_power_output = value(new_deterministic_ruc.MaximumPowerOutput[g])

            # TBD: Eventually make the 1e-5 an user-settable option.
            if math.fabs(min_power_output - power_generated_at_midnight) <= 1e-5: 
                new_deterministic_ruc.PowerGeneratedT0[g] = min_power_output
            elif math.fabs(max_power_output - power_generated_at_midnight) <= 1e-5: 
                new_deterministic_ruc.PowerGeneratedT0[g] = max_power_output
            else:
                new_deterministic_ruc.PowerGeneratedT0[g] = power_generated_at_midnight

    if output_initial_conditions:
        report_initial_conditions_for_deterministic_ruc(new_deterministic_ruc,
                                                        max_thermal_generator_label_length=max_thermal_generator_label_length)
        
        report_demand_for_deterministic_ruc(new_deterministic_ruc,
                                            max_bus_label_length=max_bus_label_length)

    if options.reserve_factor > 0.0:
        new_deterministic_ruc.ReserveFactor = options.reserve_factor

    # there are a number of "derived" parameters and related rules
    # that need to be re-fired / re-populated.
    reference_model_module.reconstruct_instance_for_t0_changes(new_deterministic_ruc)

    # preprocess the RUC, to pick up the simulation T0 initial conditions.
    new_deterministic_ruc.preprocess()

    return new_deterministic_ruc, new_scenario_tree

def solve_deterministic_ruc(deterministic_ruc_instance,
                            solver, 
                            options, 
                            solve_options):

    if options.deterministic_ruc_solver_plugin != None:
        try:
            solver_plugin_module = pyutilib.misc.import_file(options.deterministic_ruc_solver_plugin)
        except:
            raise RuntimeError("Could not locate deterministic ruc solver plugin module=%s" % options.deterministic_ruc_solver_plugin)

        solve_function = getattr(solver_plugin_module, "solve_deterministic_ruc", None)
        if solve_function is None:
            raise RuntimeError("Could not find function 'solve_deterministic_ruc' in deterministic ruc solver plugin module=%s" % options.deterministic_ruc_solver_plugin)

        solve_function(deterministic_ruc_instance, solver, options, solve_options)

    else:
        results = call_solver(solver,
                              deterministic_ruc_instance, 
                              tee=options.output_solver_logs,
                              **solve_options[solver])

        if results.solution.status.key != "optimal":
            print("Failed to solve deterministic RUC instance - no feasible solution exists!")        
            output_filename = "bad_ruc.lp"
            print("Writing failed RUC model to file=" + output_filename)
            lp_writer = ProblemWriter_cpxlp()            
            lp_writer(deterministic_ruc_instance, output_filename, 
                      lambda x: True, {"symbolic_solver_labels" : True})

            if options.error_if_infeasible:
                raise RuntimeError("Halting due to infeasibility")

        deterministic_ruc_instance.solutions.load_from(results)

# utility to resolve the scenarios in the extensive form by fixing the unit on binaries. 
# mirrors what ISOs do when solving the RUC. snapshots the solution into the scenario tree.

def resolve_stochastic_ruc_with_fixed_binaries(stochastic_ruc_instance, scenario_instances, scenario_tree,
                                               options, solver, solver_options):

    # the scenarios are independent once we fix the unit on binaries.
    for scenario_name, scenario_instance in iteritems(scenario_instances):
        print("Processing scenario name=",scenario_name)
        for index, var_data in iteritems(scenario_instance.UnitOn):
            var_data.fix()
        scenario_instance.preprocess()

        results = call_solver(solver,scenario_instance, tee=options.output_solver_logs,
                              keepfiles=options.keep_solver_files,**solve_options[solver])
        if results.solution.status.key != "optimal":
            print("Writing failed scenario=" + scenario_name + " to file=" + output_filename)
            lp_writer = ProblemWriter_cpxlp()            
            output_filename = "bad_resolve_scenario.lp"
            lp_writer(scenario_instance, output_filename, lambda x: True, True)
            raise RuntimeError("Halting - failed to solve scenario=" + scenario_name +
                               " when re-solving stochastic RUC with fixed binaries")

        for index, var_data in iteritems(scenario_instance.UnitOn):
            var_data.unfix()
        scenario_instance.preprocess()

    scenario_tree.pullScenarioSolutionsFromInstances()
    scenario_tree.snapshotSolutionFromScenarios()

#
# a utility function to write generator initial state information to a text file, for use with solver
# callbacks when exeucting runph. presently writes the file to "ic.txt", for "initial conditions".
#

def write_stochastic_initial_conditions(yesterday_hour_invoked, yesterday_ruc_simulation,
                                        yesterday_stochastic_ruc_scenarios, yesterday_scenario_tree,
                                        today_stochastic_ruc_scenarios, 
                                        output_initial_conditions, options,
                                        prior_sced_instance, projected_sced_instance):

    # the sced model and the prior sced instance are used to construct the SCED for projections
    # of generator output at time period t=24. they are only used if there is a stochastic RUC
    # from yesterday. the h hour is the hour at which the RUC is invoked.

    output_filename = "ic.txt"
    output_file = open(output_filename, "w")

    # set up the initial conditions for the stochastic RUC, based on all information
    # available as to the conditions on the initial day. if this is the first simulation
    # day, then we have nothing to go on other than the initial conditions already
    # specified in the scenario instances. however, if this is not the first simulation
    # day, use the projected solution state embodied (either explicitly or implicitly)
    # in the prior date stochastic RUC to establish the initial conditions.
    if yesterday_stochastic_ruc_scenarios != None:

        yesterday_root_node = yesterday_scenario_tree.findRootNode()
        arbitrary_scenario_instance = yesterday_stochastic_ruc_scenarios[
            list(yesterday_stochastic_ruc_scenarios.keys())[0]]

        for g in sorted(arbitrary_scenario_instance.ThermalGenerators):

            final_unit_on_state = int(round(yesterday_root_node.get_variable_value("UnitOn", (g, 24))))
            state_duration = 1
            hours = list(range(1, 24))
            hours.reverse()
            for i in hours:
                this_unit_on_state = int(round(yesterday_root_node.get_variable_value("UnitOn", (g, i))))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration

            projected_power_generated = value(projected_sced_instance.PowerGenerated[
                                                  g, 23 - yesterday_hour_invoked + 1])
            # the input h hours are 0-based - 23 is the last hour of the day

            # on occasion, the average power generated across scenarios for a single generator
            # can be a very small negative number, due to MIP tolerances allowing it. if this
            # is the case, simply threshold it to 0.0. similarly, the instance validator will
            # fail if the average power generated is small-but-positive (e.g., 1e-14) and the
            # UnitOnT0 state is Off. in the latter case, just set the average power to 0.0.
            if projected_power_generated < 0.0:
                projected_power_generated = 0.0
            elif final_unit_on_state == 0:
                projected_power_generated = 0.0                

            print(g, state_duration, projected_power_generated, file=output_file)

            # propagate the initial conditions to each scenario - the initial conditions
            # must obviously be non-anticipative. 
            for instance in today_stochastic_ruc_scenarios.values():
                instance.UnitOnT0[g] = final_unit_on_state
                instance.UnitOnT0State[g] = state_duration

                # the validators are rather picky, in that tolerances are not acceptable.
                # given that the average power generated comes from an optimization 
                # problem solve, the average power generated can wind up being less
                # than or greater than the bounds by a small epsilon. touch-up in this
                # case.
                min_power_output = value(instance.MinimumPowerOutput[g])
                max_power_output = value(instance.MaximumPowerOutput[g])
                
                # TBD: Eventually make the 1e-5 an user-settable option.
                if math.fabs(min_power_output - projected_power_generated) <= 1e-5: 
                    instance.PowerGeneratedT0[g] = min_power_output
                elif math.fabs(max_power_output - projected_power_generated) <= 1e-5: 
                    instance.PowerGeneratedT0[g] = max_power_output
                else:
                    instance.PowerGeneratedT0[g] = projected_power_generated

    else:
        print("")
        print("***WARNING: No prior stochastic RUC instance available "
              "=> There is no solution from which to draw initial conditions from; running with defaults.")

        arbitrary_scenario_instance = today_stochastic_ruc_scenarios[list(today_stochastic_ruc_scenarios.keys())[0]]

        for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
            unit_on_t0_state = value(arbitrary_scenario_instance.UnitOnT0State[g])
            t0_power_generated = value(arbitrary_scenario_instance.PowerGeneratedT0[g])
            print(g, unit_on_t0_state, t0_power_generated, file=output_file)

    # always output the reserve factor.
    print(options.reserve_factor, file=output_file)

    if output_initial_conditions:
        print("")
        report_initial_conditions_for_stochastic_ruc(today_stochastic_ruc_scenarios)
        
    print("")
    print('Finished writing initial conditions (stochastic)')

    output_file.close()

# a utility to create a stochastic RUC instance and solve it via the extensive form.


def create_and_solve_stochastic_ruc_via_ef(solver, sced_solver, solve_options, options, this_date,
                                           yesterday_hour_invoked, yesterday_ruc_simulation,
                                           yesterday_stochastic_ruc_scenarios, yesterday_scenario_tree,
                                           output_initial_conditions,
                                           sced_model, prior_sced_instance, projected_sced_instance,
                                           use_prescient_forecast_error_in_sced=True,
                                           use_persistent_forecast_error_in_sced=False):

    # NOTE: the "yesterday" technically is relative to whatever day this stochastic RUC is being constructed/solved for.
    #       so in the context of the simulator, yesterday actually means the RUC executing for today.

    print("Constructing scenario tree and scenario instances...")

    instance_directory_name = os.path.join(options.data_directory, "pyspdir_twostage",str(this_date)) 
    if not os.path.exists(instance_directory_name):
        raise RuntimeError("Stochastic RUC instance data directory=%s either does not exist or cannot be read"
                           % instance_directory_name)

    scenario_tree_instance_factory = ScenarioTreeInstanceFactory(os.path.expanduser(options.model_directory),
                                                                 os.path.expanduser(instance_directory_name))

    scenario_tree = scenario_tree_instance_factory.generate_scenario_tree()

    print("")
    print("Number of scenarios=%d" % len(scenario_tree._scenarios))
    print("")

    scenario_instances = scenario_tree_instance_factory.construct_instances_for_scenario_tree(scenario_tree)
    
    print("Done constructing scenario instances")
    
    scenario_tree_instance_factory.close()

    scenario_tree.linkInInstances(scenario_instances)

    stochastic_ruc_instance = create_ef_instance(scenario_tree,
                                                 verbose_output=options.verbose)

    # set up the initial conditions for the stochastic RUC, based on all information
    # available as to the conditions at the initial day. if this is the first simulation
    # day, then we have nothing to go on other than the initial conditions already
    # specified in the scenario instances. however, if this is not the first simulation
    # day, use the projected solution state embodied (either explicitly or implicitly)
    # in the prior date stochastic RUC to establish the initial conditions.
    if yesterday_stochastic_ruc_scenarios != None:

        assert yesterday_ruc_simulation != None

        yesterday_root_node = yesterday_scenario_tree.findRootNode()
        arbitrary_scenario_instance = yesterday_stochastic_ruc_scenarios[list(yesterday_stochastic_ruc_scenarios.keys())[0]]

        for g in sorted(arbitrary_scenario_instance.ThermalGenerators):

            final_unit_on_state = int(round(yesterday_root_node.get_variable_value("UnitOn", (g, 24))))
            state_duration = 1
            hours = list(range(1, 24))
            hours.reverse()
            for i in hours:
                this_unit_on_state = int(round(yesterday_root_node.get_variable_value("UnitOn", (g, i))))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration

            projected_power_generated = value(projected_sced_instance.PowerGenerated[
                                                  g, 23 - yesterday_hour_invoked + 1])
            # the input h hours are 0-based - 23 is the last hour of the day

            # on occasion, the average power generated across scenarios for a single generator
            # can be a very small negative number, due to MIP tolerances allowing it. if this
            # is the case, simply threshold it to 0.0. similarly, the instance validator will
            # fail if the average power generated is small-but-positive (e.g., 1e-14) and the
            # UnitOnT0 state is Off. in the latter case, just set the average power to 0.0.
            if projected_power_generated < 0.0:
                projected_power_generated = 0.0
            elif final_unit_on_state == 0:
                projected_power_generated = 0.0                

            # propagate the initial conditions to each scenario - the initial condition
            # must obviously be non-anticipative.
            for instance in scenario_instances.values():
                instance.UnitOnT0[g] = final_unit_on_state
                instance.UnitOnT0State[g] = state_duration

                # the validators are rather picky, in that tolerances are not acceptable.
                # given that the average power generated comes from an optimization 
                # problem solve, the average power generated can wind up being less
                # than or greater than the bounds by a small epsilon. touch-up in this
                # case.
                min_power_output = value(instance.MinimumPowerOutput[g])
                max_power_output = value(instance.MaximumPowerOutput[g])
                
                # TBD: Eventually make the 1e-5 an user-settable option.
                if math.fabs(min_power_output - projected_power_generated) <= 1e-5: 
                    instance.PowerGeneratedT0[g] = min_power_output
                elif math.fabs(max_power_output - projected_power_generated) <= 1e-5: 
                    instance.PowerGeneratedT0[g] = max_power_output
                else:
                    instance.PowerGeneratedT0[g] = projected_power_generated

    if output_initial_conditions:
        print("")
        report_initial_conditions_for_stochastic_ruc(scenario_instances)

    if options.reserve_factor > 0.0:
        for instance in scenario_instances.values():
            instance.ReserveFactor = options.reserve_factor

    reference_model_filename = os.path.expanduser(options.model_directory) + os.sep + "ReferenceModel.py"
    reference_model_module = pyutilib.misc.import_file(reference_model_filename)

    for instance in scenario_instances.values():

        reference_model_module.reconstruct_instance_for_t0_changes(instance)

        # and we need to re-process those constraints that reference
        # the modified derived parameters. 

        instance.preprocess()

    # preprocess the RUC extensive form, to pick up the simulation T0 initial conditions.
    # TBD - not sure if this is recursive - if so, nix the per-instance preprocess.
    stochastic_ruc_instance.preprocess()

    # solve the extensive form - which includes loading of results
    # (but not into the scenario tree)
    solve_extensive_form(solver, stochastic_ruc_instance, options,solve_options, tee=options.output_solver_logs)

    # snapshot the solution into the scenario tree.
    scenario_tree.pullScenarioSolutionsFromInstances()
    scenario_tree.snapshotSolutionFromScenarios()
    print('create_and_solve complete')

    return scenario_instances, scenario_tree

# 
# a utility to load a solution from a PH or EF-generated CSV solution file into
# a stochastic RUC instance.
#

def load_stochastic_ruc_solution_from_csv(scenario_instances, scenario_tree, csv_filename):

    csv_file = open(csv_filename, "r")
    csv_reader = csv.reader(csv_file, delimiter=",", quotechar='|')
    for line in csv_reader:
        stage_name = line[0].strip()
        node_name = line[1].strip()
        variable_name = line[2].strip()
        index = line[3].strip().split(":")
        index = tuple(i.strip() for i in index)
        quantity = float(line[4])

        if len(index) == 1:
            index = index[0]
            try:
                index = int(index)
            except ValueError:
                pass
        else:
            transformed_index = ()
            for piece in index:
                piece = piece.strip()
                piece = piece.lstrip('\'')
                piece = piece.rstrip('\'')
                transformed_component = None
                try:
                    transformed_component = int(piece)
                except ValueError:
                    transformed_component = piece
                transformed_index = transformed_index + (transformed_component,)
            index = transformed_index

        tree_node = scenario_tree.get_node(node_name)
        try:
            variable_id = tree_node._name_index_to_id[(variable_name, index)]
            for var_data, probability in tree_node._variable_datas[variable_id]:
                var_data.stale = False
                var_data.value = quantity
        except:
            # we will assume that we're dealing with a stage cost variable -
            # which is admittedly dangerous.
            for cost_vardata, probability in tree_node._cost_variable_datas:
                cost_vardata.stale = False
                cost_vardata.value = quantity

    csv_file.close()

# a utility to create a stochastic RUC instance and solve it via progressive hedging.

def create_and_solve_stochastic_ruc_via_ph(solver, sced_solver, options, this_date, 
                                           yesterday_hour_invoked, yesterday_ruc_simulation,
                                           yesterday_stochastic_ruc_scenarios, yesterday_scenario_tree,
                                           output_initial_conditions,
                                           sced_model, prior_sced_instance, projected_sced_instance):

    print("Constructing scenario tree and scenario instances...")

    instance_directory_name = os.path.join(options.data_directory, "pyspdir_twostage", this_date)
    if not os.path.exists(instance_directory_name):
        raise RuntimeError("Stochastic RUC instance data directory=%s either does not exist or cannot be read"
                           % instance_directory_name)

    scenario_tree_instance_factory = ScenarioTreeInstanceFactory(os.path.expanduser(options.model_directory),
                                                                 os.path.expanduser(instance_directory_name))

    scenario_tree = scenario_tree_instance_factory.generate_scenario_tree()

    print("")
    print("Number of scenarios=%d" % len(scenario_tree._scenarios))
    print("")

    # before we construct the instances, cull the constraints from the model - PH is solved outside of 
    # the simulator process, and we only need to instantiate variables in order to store solutions.
    cull_constraints_from_instance(scenario_tree_instance_factory._model_object, ["BuildBusZone","CreatePowerGenerationPiecewisePoints"])

    scenario_instances = scenario_tree_instance_factory.construct_instances_for_scenario_tree(scenario_tree)

    scenario_tree_instance_factory.close()

    scenario_tree.linkInInstances(scenario_instances)

    write_stochastic_initial_conditions(yesterday_hour_invoked, yesterday_ruc_simulation,
                                        yesterday_stochastic_ruc_scenarios, yesterday_scenario_tree,
                                        scenario_instances, 
                                        output_initial_conditions, options,
                                        prior_sced_instance, projected_sced_instance)

    ph_output_filename = options.output_directory + os.sep + str(this_date) + os.sep + "ph.out"

    # TBD: We currently don't link from the option --symbolic-solver-labels to propagate to the below,
    #      when constructing PH command-lines. This is a major omission. For now, it is enabled always -
    #      infeasibilities are impossible to diangose when numeric labels are employed.
    if options.ph_mode == "serial":
        command_line = ("runph -m %s -i %s --max-iterations=%d --solver=%s --solver-manager=serial "
                        "--solution-writer=pyomo.pysp.plugins.csvsolutionwriter "
                        "--user-defined-extension=initialstatesetter.py %s >& %s"
                        % (options.model_directory, instance_directory_name, options.ph_max_iterations,
                           options.ph_ruc_subproblem_solver_type, options.ph_options, ph_output_filename))
    elif options.ph_mode == "localmpi":
        if options.pyro_port == None:
            raise RuntimeError("A pyro port must be specified when running PH in localmpi mode")
        command_line = ("runph -m %s -i %s "
                        "--disable-gc "
                        "--max-iterations=%d --solver-manager=phpyro "
                        "--solver=%s --pyro-host=%s --pyro-port=%d "
                        " --scenario-solver-options=\"threads=2\" "
                        "--solution-writer=pyomo.pysp.plugins.csvsolutionwriter "
                        "--user-defined-extension=initialstatesetter.py %s >& %s"
                        % (options.model_directory, instance_directory_name,
                           options.ph_max_iterations, options.ph_ruc_subproblem_solver_type,
                           options.pyro_host,
                           options.pyro_port, 
                           options.ph_options, ph_output_filename))
    else:
        raise RuntimeError("Unknown PH execution mode=" + str(options.ph_mode) +
                           " specified - legal values are: 'serial' and 'localmpi'")
    
    print("")
    print("Executing PH to solve stochastic RUC - command line=" + command_line)
    sys.stdout.flush()
    
    process = subprocess.Popen(command_line,shell=True, executable='/bin/bash',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # os.system(command_line)
    process.wait()

    # validate that all expected output file were generated.
    if not os.path.exists("ph.csv"):
        raise RuntimeError("No results file=ph.csv generated.")
    if not os.path.exists("ph_StageCostDetail.csv"):
        raise RuntimeError("No result file=ph_StageCostDetail.csv")
    
    # move any files generated by PH to the appropriate output directory for that day.
    shutil.move("ph.csv", options.output_directory + os.sep + str(this_date))
    shutil.move("ph_StageCostDetail.csv", options.output_directory + os.sep + str(this_date))
    shutil.move("ic.txt", options.output_directory + os.sep + str(this_date))

    # load the PH solution into the corresponding scenario instances. 
    load_stochastic_ruc_solution_from_csv(scenario_instances, 
                                          scenario_tree, 
                                          options.output_directory + os.sep + str(this_date) + os.sep + "ph.csv")

    # snapshot the solution into the scenario tree.
    scenario_tree.pullScenarioSolutionsFromInstances()

    scenario_tree.snapshotSolutionFromScenarios()

    return scenario_instances, scenario_tree

########################################################################################
# a utility to find the "nearest" - quantified via Euclidean distance - scenario among #
# a candidate set relative to the input scenario, up through and including the         #
# specified simulation hour.                                                           #
########################################################################################

def find_nearest_scenario(reference_scenario, candidate_scenarios, simulation_hour):

    min_distance_scenario = None
    min_distance = 0.0
    alternative_min_distance_scenarios = [] # if there is more than one at the minimum distance

    # NOTE: because the units of demand and renewables are identical (e.g., MWs) there is
    #       no immediately obvious first-order reason to weight the quantities differently.

    # look through these in sorted order, to maintain sanity when tie-breaking - 
    # we always return the first in this case.
    for candidate_scenario_name in sorted(candidate_scenarios.keys()):
        candidate_scenario = candidate_scenarios[candidate_scenario_name]

        this_distance = 0.0
        
        for t in reference_scenario.TimePeriods:
            if t <= simulation_hour:
                for b in candidate_scenario.Buses:
                    reference_demand = value(reference_scenario.Demand[b, t])
                    candidate_demand = value(candidate_scenario.Demand[b, t])
                    diff = reference_demand - candidate_demand
                    this_distance += diff * diff

                for g in candidate_scenario.AllNondispatchableGenerators:
                    reference_power = value(reference_scenario.MaxNondispatchablePower[g, t])
                    candidate_power = value(candidate_scenario.MaxNondispatchablePower[g, t])
                    diff = reference_power - candidate_power
                    this_distance += diff * diff

        this_distance = math.sqrt(this_distance)
        this_distance /= (simulation_hour * len(reference_scenario.Buses)) # normalize to per-hour, per-bus

        if min_distance_scenario is None:
            min_distance_scenario = candidate_scenario
            min_distance = this_distance
        elif this_distance < min_distance:
            min_distance_scenario = candidate_scenario
            min_distance = this_distance        
            alternative_min_distance_scenarios = []
        elif this_distance == min_distance: # eventually put a tolerance on this
            alternative_min_distance_scenarios.append(candidate_scenario)

    if len(alternative_min_distance_scenarios) > 0:
        print("")
        print("***WARNING: Multiple scenarios exist at the minimum distance="+str(min_distance)+" - additional candidates include:")
        for scenario in alternative_min_distance_scenarios:
            print(scenario.name)

    return min_distance_scenario

################################################################################
# a simple utility to construct the scenario tree representing the data in the #
# given instance input directory.
################################################################################


def construct_scenario_tree(instance_directory):
    # create and populate the scenario tree model

    from pyomo.pysp.util.scenariomodels import scenario_tree_model
    scenario_tree_instance = None

    try:
        scenario_tree_instance_filename = os.path.expanduser(instance_directory) + os.sep + "ScenarioStructure.dat"
        scenario_tree_instance = scenario_tree_model.clone()
        instance_data = DataPortal(model=scenario_tree_instance)
        instance_data.load(filename=scenario_tree_instance_filename)
        scenario_tree_instance.load(instance_data)
    except IOError:
        exception = sys.exc_info()[1]
        print(("***ERROR: Failed to load scenario tree instance data from file=" + scenario_tree_instance_filename +
               "; Source error="+str(exception)))
        return None

    # construct the scenario tree
    scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_instance)

    return scenario_tree

#######################################################################
# a utility to determine the name of the file in which simulated data #
# is to be drawn, given a specific input date and run-time options.   #
#######################################################################


def compute_simulation_filename_for_date(a_date, options):

    simulated_dat_filename = ""

    if options.simulate_out_of_sample > 0:
        # assume the input directories have been set up with a Scenario_actuals.dat file in each day.
        simulated_dat_filename = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date), "Scenario_actuals.dat")
    else:
        if options.run_deterministic_ruc:
            print("")
            print("***WARNING: Simulating the forecast scenario when running deterministic RUC - "
                  "time consistency across midnight boundaries is not guaranteed, and may lead to threshold events.")
            simulated_dat_filename = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date),
                                                  "Scenario_forecasts.dat")
        else:
            simulation_data_directory_name = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date))
            scenario_tree = construct_scenario_tree(simulation_data_directory_name)

            selected_index = random.randint(0, len(scenario_tree._scenarios)-1)
            selected_scenario = scenario_tree._scenarios[selected_index]
            simulated_dat_filename = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date),
                                                  selected_scenario._name + ".dat")

    return simulated_dat_filename



######################################
# main simulation routine starts now #
######################################

def simulate(options):

    # echo the key outputs for the simulation - for replicability.
    print("Initiating simulation...")
    print("")
    print("Model directory:", options.model_directory)
    print("Data directory:", options.data_directory)
    print("Output directory:", options.output_directory)
    print("Random seed:", options.random_seed)
    print("Reserve factor:", options.reserve_factor)
    print("")

    # do some simple high-level checking on the existence
    # of the model and input data directories.
    if not os.path.exists(options.model_directory):
        raise RuntimeError("Model directory=%s does not exist or cannot be read" % options.model_directory)

    if not os.path.exists(options.data_directory):
        raise RuntimeError("Data directory=%s does not exist or cannot be read" % options.data_directory)

    # echo the solver configuration
    if not options.run_deterministic_ruc:
        if not options.solve_with_ph:
            print("Solving stochastic RUC instances using the extensive form")
        else:
            print("Solving stochastic RUC instances using progressive hedging - mode=%s" % options.ph_mode)
        print("")

    # not importing the time module here leads to a weird error message -
    # "local variable 'time' referenced before assignment"
    import time
    simulation_start_time = time.time()

    # seed the generator if a user-supplied seed is provided. otherwise,
    # python will seed from the current system time.
    if options.random_seed > 0:
        random.seed(options.random_seed)

    # validate the PH execution mode options.
    if options.ph_mode != "serial" and options.ph_mode != "localmpi":
        raise RuntimeError("Unknown PH execution mode=" + str(options.ph_mode) +
                           " specified - legal values are: 'serial' and 'localmpi'")

    # Model choice control
    run_ruc = (options.disable_ruc == False)
    run_sced = (options.disable_sced == False)

    model_filename = os.path.join(options.model_directory, "ReferenceModel.py")
    if not os.path.exists(model_filename):
        raise RuntimeError("The model %s either does not exist or cannot be read" % model_filename)

    from pyutilib.misc import import_file
    reference_model_module = import_file(model_filename)

    # make sure all utility methods required by the simulator are defined in the reference model module.
    validate_reference_model(reference_model_module)
    
    ruc_model = reference_model_module.model
    sced_model = reference_model_module.model

    # state variables
    offline_generators = []
    offline_lines_B = {}

    # NOTE: We eventually want specific solver options for the various types below.
    if options.python_io:
        deterministic_ruc_solver = SolverFactory(options.deterministic_ruc_solver_type, solver_io="python")
        ef_ruc_solver = SolverFactory(options.ef_ruc_solver_type, solver_io="python")
        sced_solver = SolverFactory(options.sced_solver_type, solver_io="python")
    else:
        deterministic_ruc_solver = SolverFactory(options.deterministic_ruc_solver_type)
        ef_ruc_solver = SolverFactory(options.ef_ruc_solver_type)
        sced_solver = SolverFactory(options.sced_solver_type)
        
    solve_options = {}
    solve_options[sced_solver] = {}
    solve_options[deterministic_ruc_solver] = {}
    solve_options[ef_ruc_solver] = {}
    
    for s in solve_options:
        solve_options[s]['symbolic_solver_labels'] = options.symbolic_solver_labels

    if options.deterministic_ruc_solver_type == "cplex" or options.deterministic_ruc_solver_type == "cplex_persistent":
        deterministic_ruc_solver.options.mip_tolerances_mipgap = options.ruc_mipgap
    elif options.deterministic_ruc_solver_type == "gurobi" or options.deterministic_ruc_solver_type == "gurobi_persistent":
        deterministic_ruc_solver.options.MIPGap = options.ruc_mipgap
    else:
        raise RuntimeError("Unknown solver type=%s specified" % options.deterministic_ruc_solver_type)
    
    if options.ef_ruc_solver_type == "cplex" or options.ef_ruc_solver_type == "cplex_persistent":
        ef_ruc_solver.options.mip_tolerances_mipgap = options.ef_mipgap
    elif options.ef_ruc_solver_type == "gurobi" or options.ef_ruc_solver_type == "gurobi_persistent":
        ef_ruc_solver.options.MIPGap = options.ef_mipgap
    else:
        raise RuntimeError("Unknown solver type=%s specified" % options.ef_ruc_solver_type)
    
    ef_ruc_solver.set_options("".join(options.stochastic_ruc_ef_solver_options))
    deterministic_ruc_solver.set_options("".join(options.deterministic_ruc_solver_options))
    sced_solver.set_options("".join(options.sced_solver_options))
    
    # validate the start date
    try:
        start_date = dateutil.parser.parse(options.start_date)
    except ValueError:
        print("***ERROR: Illegally formatted start date=" + options.start_date + " supplied!")
        sys.exit(1)

    # Initialize nested dictionaries for holding output
    dates_to_simulate = [str(a_date)
                         for a_date in daterange(start_date.date(),
                                                 start_date.date() +
                                                 datetime.timedelta(options.num_days-1))]
    end_date = dates_to_simulate[-1]
    list_hours = list(range(0, 24))



    # TBD: eventually option-drive the interval (minutely) within an hour that
    #      economic dispatch is executed. for now, because we don't have better
    #      data, drive it once per hour.
    list_minutes = list(range(0, 60, 60))

    print("Dates to simulate:",dates_to_simulate)

    # we often want to write SCED instances, for diagnostics.
    lp_writer = ProblemWriter_cpxlp()

    # before the simulation starts, delete the existing contents of the output directory.
    if os.path.exists(options.output_directory):
        shutil.rmtree(options.output_directory)    
    os.mkdir(options.output_directory)
    
    # create an output directory for plot data.
    # for now, it's only a single directory - not per-day.
    os.mkdir(os.path.join(options.output_directory,"plots"))

    if run_ruc:
        last_ruc_date = dates_to_simulate[-1]
        print("")
        print("Last RUC date:", last_ruc_date)

    scenario_instances_for_tomorrow = None
    scenario_tree_for_tomorrow = None

    deterministic_ruc_instance_for_tomorrow = None

    # build all of the per-date output directories up-front, for simplicity.
    for this_date in dates_to_simulate:
        simulation_directory_for_this_date = options.output_directory + os.sep + str(this_date) 
        os.mkdir(options.output_directory + os.sep + str(this_date))

    ########################################################################################
    # we need to create the "yesterday" deterministic or stochastic ruc instance, to kick  #
    # off the simulation process. for now, simply solve RUC for the first day, as          #
    # specified in the instance files. in practice, this may not be the best idea but it   #
    # will at least get us a solution to start from.                                       #
    ########################################################################################

    print("")
    first_date = dates_to_simulate[0]

    # not importing the time module here leads to a weird error message -
    # "local variable 'time' referenced before assignment"
    import time
    start_time = time.time()

    # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
    # prescient processing is identical in the case of deterministic and stochastic RUC. 
    # persistent processing differs, as there is no point forecast for stochastic RUC.
    use_prescient_forecast_error_in_sced = True # always the default
    use_persistent_forecast_error_in_sced = False
    if options.run_sced_with_persistent_forecast_errors:
        print("Using persistent forecast error model when projecting demand and renewables in SCED")
        use_persistent_forecast_error_in_sced = True
        use_prescient_forecast_error_in_sced = False
    else:
        print("Using prescient forecast error model when projecting demand and renewables in SCED")
    print("")

    if options.run_deterministic_ruc:
        print("Creating and solving deterministic RUC instance for date:", first_date)
        deterministic_ruc_instance_for_tomorrow, \
        scenario_tree_for_tomorrow = create_deterministic_ruc(options,
                                                              first_date,
                                                              None,
                                                              # no prior deterministic RUC
                                                              None,
                                                              # to projected SCED for midnight conditions
                                                              options.output_ruc_initial_conditions)
        solve_deterministic_ruc(deterministic_ruc_instance_for_tomorrow,
                                deterministic_ruc_solver,
                                options,
                                solve_options)

        total_cost = deterministic_ruc_instance_for_tomorrow.TotalCostObjective()
        print("")
        print("Deterministic RUC Cost: {0:.2f}".format(total_cost))

        if options.output_ruc_solutions:
            print("")
            output_solution_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow, first_date)

        print("")
        report_fixed_costs_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow)
        report_generation_costs_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow)
        print("")
        report_load_generation_mismatch_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow)


    else:
        print("Creating and solving stochastic RUC instance for date:", first_date)
        if options.solve_with_ph == False:
            scenario_instances_for_tomorrow, \
            scenario_tree_for_tomorrow = create_and_solve_stochastic_ruc_via_ef(ef_ruc_solver, 
                                                                                sced_solver,solve_options,
                                                                                options, 
                                                                                first_date, 
                                                                                -1,
                                                                                None,
                                                                                # no yesterday ruc was simulated
                                                                                None,
                                                                                # no yesterday stochastic ruc scenarios
                                                                                None,
                                                                                # no yesterday scenario tree
                                                                                options.output_ruc_initial_conditions,
                                                                                sced_model,
                                                                                None,
                                                                                # no prior SCED instance
                                                                                None) # no projected SCED instance
        else:
            scenario_instances_for_tomorrow, \
            scenario_tree_for_tomorrow = create_and_solve_stochastic_ruc_via_ph(solver, 
                                                                                sced_solver,
                                                                                options, 
                                                                                first_date, 
                                                                                None,
                                                                                None,
                                                                                # no yesterday ruc was simulated
                                                                                None,
                                                                                # no yesterday stochastic ruc scenarios
                                                                                None,
                                                                                # no yesterday scenario tree
                                                                                options.output_ruc_initial_conditions,
                                                                                sced_model,
                                                                                None,
                                                                                # no prior SCED instance
                                                                                None)  # no projected SCED instance

        # now report the solution that we have generated.
        expected_cost = scenario_tree_for_tomorrow.findRootNode().computeExpectedNodeCost()
        if expected_cost == None:
            scenario_tree_for_tomorrow.pprintCosts()
            raise RuntimeError("Could not computed expected cost - one or more stage"
                               " costs is undefined. This is likely due to not all"
                               " second stage variables being specified in the"
                               " scenario tree structure definition file.")
        print("Stochastic RUC expected cost: %8.2f" % expected_cost)

        if options.output_ruc_solutions:
            print("")
            output_solution_for_stochastic_ruc(scenario_instances_for_tomorrow, 
                                               scenario_tree_for_tomorrow, 
                                               output_scenario_dispatches=options.output_ruc_dispatches)

        print("")
        report_fixed_costs_for_stochastic_ruc(scenario_instances_for_tomorrow)
        print("")
        report_generation_costs_for_stochastic_ruc(scenario_instances_for_tomorrow)
        print("")
        report_load_generation_mismatch_for_stochastic_ruc(scenario_instances_for_tomorrow)

        print("")
        print(("Construction and solve time=%.2f seconds" % (time.time() - start_time)))

    ##################################################################################
    # variables to track the aggregate simulation costs and load shedding quantities #
    ##################################################################################
    
    total_overall_fixed_costs = 0.0
    total_overall_generation_costs = 0.0
    total_overall_load_shedding = 0.0
    total_overall_over_generation = 0.0
    total_overall_reserve_shortfall = 0.0
    total_overall_renewables_curtailment = 0.0
    total_on_offs = 0
    total_sum_on_off_ramps = 0.0
    total_sum_nominal_ramps = 0.0
    total_quick_start_additional_costs = 0.0
    total_quick_start_additional_power_generated = 0.0

    ###############################################################################
    # variables to track daily statistics, for plotting or summarization purposes #
    # IMPT: these are the quantities that are also written to a CSV file upon     #
    #       completion of the simulation.                                         #
    ###############################################################################

    daily_total_costs = []
    daily_fixed_costs = []
    daily_generation_costs = []
    daily_load_shedding = []
    daily_over_generation = []
    daily_reserve_shortfall = []
    daily_renewables_available = []
    daily_renewables_used = []
    daily_renewables_curtailment = []
    daily_demand = []
    daily_average_price = []
    daily_on_offs = [] 
    daily_sum_on_off_ramps = [] 
    daily_sum_nominal_ramps = []
    daily_quick_start_additional_costs = []
    daily_quick_start_additional_power_generated = []

    # statistics accumulated over the entire simulation, to compute effective 
    # renewables penetration rate across the time horizon of the simulation.
    cumulative_demand = 0.0
    cumulative_renewables_used = 0.0

    ############################
    # create output dataframes #
    ############################

    daily_summary_df = pd.DataFrame(columns=['Date','Demand','Renewables available', 'Renewables used', 'Renewables penetration rate','Average price','Fixed costs','Generation costs','Load shedding','Over generation','Reserve shortfall','Renewables curtailment','On/off','Sum on/off ramps','Sum nominal ramps'])
    
    options_df = pd.DataFrame(columns=['Date', 'Model directory', 'Data directory', 'Output directory', 'Random seed', 'Reserve factor', 'Ruc mipgap', 'Solver type','Num days', 'Sart Date', 'Run deterministic ruc', 'Run sced with persistend forecast errors', 'Output ruc solutions', 'Options ruc dispatches', 'Output solver log', 'Relax ramping if infeasible', 'Output sced solutions', 'Plot individual generators', 'Output sced initial conditions', 'Output sced demands', 'Simulate out of sample', 'Output ruc initial conditions', 'Sced horizon', 'Traceback', 'Run simulator'])
    
    thermal_generator_dispatch_df = pd.DataFrame(columns=['Date', 'Hour', 'Generator', 'Dispatch', 'Headroom', 'Unit State'])

    renewables_production_df  = pd.DataFrame(columns=['Date', 'Hour', 'Generator', 'Output', 'Curtailment'])

    line_df = pd.DataFrame(columns=['Date', 'Hour', 'Line', 'Flow'])

    bus_df = pd.DataFrame(columns=['Date', 'Hour', 'Bus', 'Shortfall', 'Overgeneration', 'LMP'])

    overall_simulation_output_df = pd.DataFrame(columns=['Total demand','Total fixed costs','Total generation costs','Total costs','Total load shedding','Total over generation','Total reserve shortfall','Total renewables curtialment','Total on/offs','Total sum on/off ramps','Total sum nominal ramps','Maximum observed demand','Overall renewables penetration rate','Cumulative average price'])
    
    quickstart_summary_df = pd.DataFrame(columns=['Date', 'Hour', 'Generator', 'Used as quickstart', 'Dispatch level of quick start generator'])
    
    hourly_gen_summary_df = pd.DataFrame(columns=['Date', 'Hour', 'Load shedding', 'Reserve shortfall', 'Available reserves', 'Over generation'])

    runtime_df = pd.DataFrame(columns=['Date','Hour','Type', 'Solve Time'])
    
    ###################################
    # prep the hourly output CSV file #
    ###################################

    csv_hourly_output_filename = os.path.join(options.output_directory,"hourly_summary.csv")
    csv_hourly_output_file = open(csv_hourly_output_filename,"w")
    sim_dates = [this_date for this_date in dates_to_simulate]
    print("Date", ",", "Hour", ",", "TotalCosts", ",", "FixedCosts", ",", "VariableCosts", ",", "LoadShedding", \
          ",", "OverGeneration", ",", "ReserveShortfall", ",", "RenewablesUsed", ",", "RenewablesCurtailed", ",",
          "Demand", ",", "Price", file=csv_hourly_output_file)

    #################################################################
    # construct the simiulation data associated with the first date #
    #################################################################

    # identify the .dat file from which (simulated) actual load data will be drawn.
    simulated_dat_filename = compute_simulation_filename_for_date(dates_to_simulate[0], options)

    print("")
    print("Actual simulation data drawn from file=" + simulated_dat_filename)

    if not os.path.exists(simulated_dat_filename):
        raise RuntimeError("The file " + simulated_dat_filename + " does not exist or cannot be read.")

    # the RUC instance to simulate only exists to store the actual demand and renewables outputs
    # to be realized during the course of a day. it also serves to provide a concrete instance,
    # from which static data and topological features of the system can be extracted.
    # IMPORTANT: This instance should *not* be passed to any method involved in the creation of
    #            economic dispatch instances, as that would enable those instances to be 
    #            prescient. 
    print("")
    print("Creating RUC instance to simulate")
    ruc_instance_to_simulate_tomorrow = ruc_model.create_instance(simulated_dat_filename)

    #####################################################################################################
    # the main simulation engine starts here - loop through each day, performing ISO-related activities #
    #####################################################################################################

    if options.simulate_out_of_sample != False:
        print("")
        print("Executing simulation using out-of-sample scenarios")

    for this_date in dates_to_simulate:

        # preliminaries 
        print("")
        print(">>>>Simulating date: "+this_date)
        if this_date == dates_to_simulate[-1]:
            next_date = None
        else:
            next_date = dates_to_simulate[dates_to_simulate.index(this_date)+1]
        print("")

        # transfer over the ruc instance to simulate
        ruc_instance_to_simulate_today = ruc_instance_to_simulate_tomorrow

        # initialize the actual demand and renewables vectors - these will be incrementally 
        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
        actual_demand = dict(((b, t), value(ruc_instance_to_simulate_today.Demand[b, t]))
                             for b in ruc_instance_to_simulate_today.Buses
                             for t in ruc_instance_to_simulate_today.TimePeriods)
        actual_min_renewables = dict(((g, t), value(ruc_instance_to_simulate_today.MinNondispatchablePower[g, t]))
                                     for g in ruc_instance_to_simulate_today.AllNondispatchableGenerators
                                     for t in ruc_instance_to_simulate_today.TimePeriods)
        actual_max_renewables = dict(((g, t), value(ruc_instance_to_simulate_today.MaxNondispatchablePower[g, t]))
                                     for g in ruc_instance_to_simulate_today.AllNondispatchableGenerators
                                     for t in ruc_instance_to_simulate_today.TimePeriods)

        # the thermal fleet capacity is necessarily a static quantity, so it 
        # technically doesn't have to be computed each iteration. however, we
        # don't have an instance outside of the date loop, and it doesn't cost
        # anything to compute. this quantity is primarily used as a normalization
        # term for stackgraph generation.
        thermal_fleet_capacity = sum(value(ruc_instance_to_simulate_today.MaximumPowerOutput[g])
                                     for g in ruc_instance_to_simulate_today.ThermalGenerators)

        # track the peak demand in any given hour of the simulation, to report 
        # at the end. the primary use of this data is to appropriate scale
        # the stack graph plots in subsequent runs.
        max_hourly_demand = 0.0

        # create a dictionary of observed dispatch values during the course of the 
        # day - for purposes of generating stack plots and related outputs.
        observed_thermal_dispatch_levels = {}
        for g in ruc_instance_to_simulate_today.ThermalGenerators:
            observed_thermal_dispatch_levels[g] = np.array([0.0 for r in range(24)])

        # useful to know the headroom for thermal generators as well.
        # headroom - max-available-power minus actual dispatch.
        observed_thermal_headroom_levels = {}
        for g in ruc_instance_to_simulate_today.ThermalGenerators:
            observed_thermal_headroom_levels[g] = np.array([0.0 for r in range(24)])

        # do the above, but for renewables power used (*not* available)
        observed_renewables_levels = {}
        for g in ruc_instance_to_simulate_today.AllNondispatchableGenerators:
            observed_renewables_levels[g] = np.array([0.0 for r in range(24)])

        # and curtailment...
        observed_renewables_curtailment = {}
        for g in ruc_instance_to_simulate_today.AllNondispatchableGenerators:
            observed_renewables_curtailment[g] = np.array([0.0 for r in range(24)])

        # dictionary for on/off states of thermal generators in real-time dispatch.
        observed_thermal_states = {}
        for g in ruc_instance_to_simulate_today.ThermalGenerators:
            observed_thermal_states[g] = np.array([-1 for r in range(24)])        

        # dictionary for line flows.
        observed_flow_levels = {}
        for l in ruc_instance_to_simulate_today.TransmissionLines:
            observed_flow_levels[l] = np.array([0.0 for r in range(24)])

        # dictionary for bus load-generate mismatch.
        observed_bus_mismatches = {}
        for b in ruc_instance_to_simulate_today.Buses:
            observed_bus_mismatches[b] = np.array([0.0 for r in range(24)])

        # dictionary for bus LMPs.
        observed_bus_LMPs = {}
        for b in ruc_instance_to_simulate_today.Buses:
            observed_bus_LMPs[b] = np.array([0.0 for r in range(24)])            

        # dictionary of input and output levels for storage units
        storage_input_dispatchlevelsdict = {}
        for s in ruc_instance_to_simulate_today.Storage:
            storage_input_dispatchlevelsdict[s] = np.array([0.0 for r in range(24)])

        storage_output_dispatchlevelsdict = {}
        for s in ruc_instance_to_simulate_today.Storage:
            storage_output_dispatchlevelsdict[s] = np.array([0.0 for r in range(24)])

        storage_soc_dispatchlevelsdict = {}
        for s in ruc_instance_to_simulate_today.Storage:
            storage_soc_dispatchlevelsdict[s] = np.array([0.0 for r in range(24)])

        # SCED run-times, in seconds.
        sced_runtimes = []

        # keep track of any events that are worth annotating on daily generation plots.
        # the entries in this list should be (x,y) pairs, where x is the event hour and
        # and y is the associated text label.
        event_annotations = []

        # track the total curtailment across the day, for output / plot generation purposes.
        curtailments_by_hour = []

        # track the total load shedding across the day, for output / plot generation purposes.
        load_shedding_by_hour = []

        # track the total over generation across the day, for output / plot generation purposes.
        over_generation_by_hour = []

        # the reserve requirements as induced by the SCED.
        reserve_requirements_by_hour = []

        # shortfalls in reserve requirements as induced by the SCED.
        reserve_shortfalls_by_hour = []

        # total available reserve as computed by the SCED.
        available_reserves_by_hour = []

        # available quickstart by hour as computed by the SCED.
        available_quickstart_by_hour = []

        # quick start generators committed before unfixing the SCED
        fixed_quick_start_generators_committed = []

        # quick start generators committed after unfixing the SCED
        unfixed_quick_start_generators_committed = []

        # quick start additional costs by hour as computed by the SCED
        quick_start_additional_costs_by_hour = []

        # quick start additional power generated by hour as computed by the SCED
        quick_start_additional_power_generated_by_hour = []

        # generators used as quick start by hour as computed by the SCED
        used_as_quick_start = {}

        if options.enable_quick_start_generator_commitment:
            for g in ruc_instance_to_simulate_today.QuickStartGenerators:
                used_as_quick_start[g]=[]

        # establish the stochastic ruc instance for today - we use this instance to track,
        # for better or worse, the projected and actual UnitOn states through the day.
        if options.run_deterministic_ruc:
            deterministic_ruc_instance_for_today = deterministic_ruc_instance_for_tomorrow
            scenario_tree_for_today = scenario_tree_for_tomorrow
        else:
            scenario_instances_for_today = scenario_instances_for_tomorrow
            scenario_tree_for_today = scenario_tree_for_tomorrow

        # now that we've established the stochastic ruc for today, null out the one for tomorrow.
        scenario_instances_for_tomorrow = None
        scenario_tree_for_tomorrow = None

        # NOTE: Not sure if the demand should be demand desired - or satisfied? Right now, it's the former.
        this_date_demand = 0.0
        this_date_fixed_costs = 0.0
        this_date_variable_costs = 0.0
        this_date_over_generation = 0.0
        this_date_load_shedding = 0.0
        this_date_reserve_shortfall = 0.0
        this_date_renewables_available = 0.0
        this_date_renewables_used = 0.0
        this_date_renewables_curtailment = 0.0
        this_date_on_offs = 0
        this_date_sum_on_off_ramps = 0.0
        this_date_sum_nominal_ramps = 0.0
        this_date_quick_start_additional_costs = 0.0
        this_date_quick_start_additional_power_generated = 0.0
        
        # compute a demand and renewables forecast error for all time periods being simulated 
        # today. this is used when creating SCED instances, where it is useful / necessary
        # (which depends on the simulation option) in order to adjust the projected quantities
        # that was originally used in solving the RUC.

        if options.run_deterministic_ruc:
            print("")
            print("NOTE: Positive forecast errors indicate projected values higher than actuals")
            demand_forecast_error = {}  # maps (bus,time-period) pairs to an error, defined as forecast minus actual
            for b in deterministic_ruc_instance_for_today.Buses:
                for t in range(1, 49):  # TBD - not sure how to option-drive the upper bound on the time period value.
                    demand_forecast_error[b, t] = value(deterministic_ruc_instance_for_today.Demand[b, t]) - \
                                                  value(ruc_instance_to_simulate_today.Demand[b,t])
                    print("Demand forecast error for bus=%s at t=%2d: %12.2f" % (b, t, demand_forecast_error[b,t]))

            print("")
            renewables_forecast_error = {}
            # maps (generator,time-period) pairs to an error, defined as forecast minus actual
            for g in deterministic_ruc_instance_for_today.AllNondispatchableGenerators:
                for t in range(1, 49):
                    renewables_forecast_error[g,t] = value(deterministic_ruc_instance_for_today.MaxNondispatchablePower[g, t]) - \
                                                     value(ruc_instance_to_simulate_today.MaxNondispatchablePower[g, t])
                    print("Renewables forecast error for generator=%s at t=%2d: %12.2f" % (g, t, renewables_forecast_error[g, t]))

        for h in list_hours:

            print("")
            print(">>>>Simulating hour: " + str(h+1) + " (date: " + str(this_date) + ")")

            if run_sced:
                
                # establish the previous sced instance, which is used as the basis for 
                # the initial conditions (T0 state, power generated) for the next 
                # sced instance. 
                
                if (h == 0) and (this_date == first_date):
                    # there is to prior sced instance - we'll have to take the initial
                    # conditions from the stochastic RUC initial conditions.
                    prior_sced_instance = None
                else:
                    # take the sced instance from the prior iteration of the by-hour simulation loop.
                    prior_sced_instance = current_sced_instance

            # run RUC at D-X (except on the last day of the simulation), where X is the 
            # user-specified hour at which RUC is to be executed each day.
            if run_ruc and (this_date != end_date) and (h == options.ruc_execution_hour):

                if this_date != last_ruc_date:

                    start_time = time.time()

                    print("")
                    print("Creating and solving SCED to determine UC initial conditions for date:", next_date)

                    # to create the RUC for tomorrow, we need to estimate initial conditions
                    # at midnight. we can do this most accurately by solving a sced, taking into account 
                    # our current forecast error model. note that there is a bit of a cart-before-the-horse
                    # problem here, as we must estimate the midnight conditions prior to knowing the 
                    # unit commitments for the next day (which are necessarily taken as a repeat of the 
                    # current day commitments, lacking an alternative). we could in principle compute 
                    # updated forecast errors for the next day, but that would seem to be of minimal use
                    # in that the binaries would not be correct - and it is a mouthful to explain. and
                    # probably not realistic.

                    if options.run_deterministic_ruc:
                        scenario_instances_for_today = None
                    else:
                        deterministic_ruc_instance_for_today = None
                        demand_forecast_error = None
                        renewables_forecast_error = None

                    # NOTE: the projected sced probably doesn't have to be run for a full 24 hours - just enough 
                    #       to get you to midnight and a few hours beyond (to avoid end-of-horizon effects).
                    projected_sced_instance = create_sced_instance(sced_model, 
                                                                   deterministic_ruc_instance_for_today,
                                                                   scenario_instances_for_today,
                                                                   scenario_tree_for_today,
                                                                   None, None, None,
                                                                   # we're setting up for tomorrow - it's not here yet!
                                                                   ruc_instance_to_simulate_today, prior_sced_instance, 
                                                                   actual_demand,
                                                                   demand_forecast_error,
                                                                   actual_min_renewables,
                                                                   actual_max_renewables,
                                                                   renewables_forecast_error,
                                                                   h,
                                                                   options.reserve_factor, 
                                                                   hours_in_objective=23-h+1,
                                                                   initialize_from_ruc=initialize_from_ruc,
                                                                   use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                                   use_persistent_forecast_error=use_persistent_forecast_error_in_sced)

                    sced_results = call_solver(sced_solver,projected_sced_instance, 
                                               tee=options.output_solver_logs,
                                               keepfiles=options.keep_solver_files,
                                               **solve_options[sced_solver])
                    if sced_results.solution.status.key != "optimal":
                        raise RuntimeError("Failed to solve initial condition SCED")
                    projected_sced_instance.solutions.load_from(sced_results)

                    if options.run_deterministic_ruc:

                        print("")
                        print("Creating and solving deterministic RUC instance for date:", next_date)

                        deterministic_ruc_instance_for_tomorrow, \
                        scenario_tree_for_tomorrow = create_deterministic_ruc(options,
                                                                              next_date,
                                                                              deterministic_ruc_instance_for_today,
                                                                              projected_sced_instance,
                                                                              options.output_ruc_initial_conditions,
                                                                              sced_midnight_hour=23-h+1)

                        solve_deterministic_ruc(deterministic_ruc_instance_for_tomorrow,
                                                deterministic_ruc_solver,
                                                options,
                                                solve_options)

                        total_cost = deterministic_ruc_instance_for_tomorrow.TotalCostObjective()
                        print("")
                        print("Deterministic RUC Cost: {0:.2f}".format(total_cost))

                        if options.output_ruc_solutions:

                            print("")                            
                            output_solution_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow, this_date)

                        print("")
                        report_fixed_costs_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow)
                        report_generation_costs_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow)
                        print("")
                        report_load_generation_mismatch_for_deterministic_ruc(deterministic_ruc_instance_for_tomorrow)                        
                    else:

                        print("")
                        print("Solving stochastic RUC for date:", next_date)
                        print()
                        print("Creating and solving stochastic RUC instance for date:", next_date)
                        ##SOLVER CALL##
                        if options.solve_with_ph == False:
                            scenario_instances_for_tomorrow, \
                            scenario_tree_for_tomorrow = \
                                create_and_solve_stochastic_ruc_via_ef(ef_ruc_solver,
                                                                       sced_solver,solve_options,
                                                                       options,
                                                                       next_date,
                                                                       h,
                                                                       ruc_instance_to_simulate_today,
                                                                       scenario_instances_for_today,
                                                                       scenario_tree_for_today,
                                                                       options.output_ruc_initial_conditions,
                                                                       sced_model,
                                                                       prior_sced_instance,
                                                                       projected_sced_instance)
                        else:
                            scenario_instances_for_tomorrow, \
                            scenario_tree_for_tomorrow = \
                                create_and_solve_stochastic_ruc_via_ph(solver,
                                                                       sced_solver,
                                                                       options,
                                                                       next_date,
                                                                       h,
                                                                       ruc_instance_to_simulate_today,
                                                                       scenario_instances_for_today,
                                                                       scenario_tree_for_today,
                                                                       options.output_ruc_initial_conditions,
                                                                       sced_model,
                                                                       prior_sced_instance,
                                                                       projected_sced_instance)

                        expected_cost = scenario_tree_for_tomorrow.findRootNode().computeExpectedNodeCost()
                        print("Stochastic RUC expected cost: {0:.2f}".format(expected_cost))

                        print("")
                        report_fixed_costs_for_stochastic_ruc(scenario_instances_for_tomorrow)
                        print("")
                        report_generation_costs_for_stochastic_ruc(scenario_instances_for_tomorrow)
                        print("")
                        report_load_generation_mismatch_for_stochastic_ruc(scenario_instances_for_tomorrow)

                        if options.output_ruc_solutions:
                            print("")
                            output_solution_for_stochastic_ruc(scenario_instances_for_tomorrow, 
                                                               scenario_tree_for_tomorrow, 
                                                               output_scenario_dispatches=options.output_ruc_dispatches)
   


                        print("")
                        print(("Construction and solve time=%.2f seconds" % (time.time() - start_time)))

                        print("")

                # we assume that the RUC solution time coincides with the availability of any new forecasted quantities
                # for the next day, i.e., they are "released". these can and should be leveraged in all subsequent
                # SCED solves, for both prescient and persistent modes. in principle, we could move release of the 
                # forecast into a separate point of the code, e.g., its own hour. 

                simulated_dat_filename = compute_simulation_filename_for_date(next_date, options)
                print("")
                print("Actual simulation data for date=" + str(next_date) + " drawn from file=" +
                      simulated_dat_filename)

                if not os.path.exists(simulated_dat_filename):
                    raise RuntimeError("The file " + simulated_dat_filename + " does not exist or cannot be read.")

                print("")
                print("Creating RUC instance to simulate")
                ruc_instance_to_simulate_tomorrow = ruc_model.create_instance(simulated_dat_filename)

                # demand and renewables forecast errors only make sense in the context of deterministic RUC.
                if options.run_deterministic_ruc:

                    # update the demand and renewables forecast error dictionaries, using the recently released forecasts.
                    print("")
                    print("Updating forecast errors")
                    print("")
                    for b in deterministic_ruc_instance_for_today.Buses:
                        for t in range(1,25): 
                            demand_forecast_error[b, t+24] = \
                                value(deterministic_ruc_instance_for_tomorrow.Demand[b, t]) - \
                                value(ruc_instance_to_simulate_tomorrow.Demand[b, t])
                            print("Demand forecast error for bus=%s at t=%2d: %12.2f"
                                  % (b, t, demand_forecast_error[b, t+24]))

                    print("")

                    for g in deterministic_ruc_instance_for_today.AllNondispatchableGenerators:
                        for t in range(1, 25):
                            renewables_forecast_error[g, t+24] = \
                                value(deterministic_ruc_instance_for_tomorrow.MaxNondispatchablePower[g, t]) -\
                                value(ruc_instance_to_simulate_tomorrow.MaxNondispatchablePower[g, t])
                            print("Renewables forecast error for generator=%s at t=%2d: %12.2f"
                                  % (g, t, renewables_forecast_error[g, t+24]))

                # update the second 24 hours of the current actual demand/renewables vectors
                for t in range(1,25):
                    for b in ruc_instance_to_simulate_tomorrow.Buses:
                        actual_demand[b, t+24] = value(ruc_instance_to_simulate_tomorrow.Demand[b,t])
                    for g in ruc_instance_to_simulate_tomorrow.AllNondispatchableGenerators:
                        actual_min_renewables[g, t+24] = \
                            value(ruc_instance_to_simulate_tomorrow.MinNondispatchablePower[g, t])
                        actual_max_renewables[g, t+24] = \
                            value(ruc_instance_to_simulate_tomorrow.MaxNondispatchablePower[g, t])

            if run_sced:

                for m in list_minutes:

                    start_time = time.time()

                    print("")
                    print("Creating SCED optimization instance for day:", str(this_date), " hour:", str(h),
                          "minute:", str(m))

                    # if this is the first hour of the day, we might (often) want to establish initial conditions from
                    # something other than the prior sced instance. these reasons are not for purposes of realism, but
                    # rather pragmatism. for example, there may be discontinuities in the "projected" initial condition
                    # at this time (based on the stochastic RUC schedule being executed during the day) and that of the
                    # actual sced instance.
                    if (h == 0) and (m == list_minutes[0]):
                        if this_date == first_date:
                            # if this is the first date to simulate, we don't have anything
                            # else to go off of when it comes to initial conditions - use those
                            # found in the RUC.
                            initialize_from_ruc = True
                        else:
                            # if we're not in the first day, we should always simulate from the
                            # state of the prior SCED.
                            initialize_from_ruc = False

                    else:
                        initialize_from_ruc = False

                    if options.run_deterministic_ruc:
                        current_sced_instance = \
                            create_sced_instance(sced_model,
                                                 deterministic_ruc_instance_for_today, None, scenario_tree_for_today,
                                                 deterministic_ruc_instance_for_tomorrow, None, scenario_tree_for_tomorrow,
                                                 ruc_instance_to_simulate_today, prior_sced_instance,
                                                 actual_demand,
                                                 demand_forecast_error,
                                                 actual_min_renewables,
                                                 actual_max_renewables,
                                                 renewables_forecast_error,
                                                 h,
                                                 options.reserve_factor,
                                                 sced_horizon=options.sced_horizon,
                                                 initialize_from_ruc=initialize_from_ruc,
                                                 use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                 use_persistent_forecast_error=use_persistent_forecast_error_in_sced)

                    else:
                        current_sced_instance = \
                            create_sced_instance(sced_model,
                                                 None, scenario_instances_for_today, scenario_tree_for_today,
                                                 None, scenario_instances_for_tomorrow, scenario_tree_for_tomorrow,
                                                 ruc_instance_to_simulate_today, prior_sced_instance,
                                                 actual_demand,
                                                 None,
                                                 actual_min_renewables,
                                                 actual_max_renewables,
                                                 None,
                                                 h,
                                                 options.reserve_factor,
                                                 sced_horizon=options.sced_horizon,
                                                 initialize_from_ruc=initialize_from_ruc,
                                                 use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                 use_persistent_forecast_error=use_persistent_forecast_error_in_sced)

                    # for pretty-printing purposes, compute the maximum bus and generator label lengths.
                    max_bus_label_length = max((len(this_bus) for this_bus in current_sced_instance.Buses))

                    if len(current_sced_instance.TransmissionLines) == 0:
                        max_line_label_length = None
                    else:
                        max_line_label_length = max((len(this_line) for this_line in current_sced_instance.TransmissionLines))

                    if len(current_sced_instance.ThermalGenerators) == 0:
                        max_thermal_generator_label_length = None
                    else:
                        max_thermal_generator_label_length = max((len(this_generator) for this_generator in current_sced_instance.ThermalGenerators))

                    if len(current_sced_instance.AllNondispatchableGenerators) == 0:
                        max_nondispatchable_generator_label_length = None
                    else:
                        max_nondispatchable_generator_label_length = max((len(this_generator) for this_generator in current_sced_instance.AllNondispatchableGenerators))

                    if options.write_sced_instances:
                        current_sced_filename = options.output_directory + os.sep + str(this_date) + \
                                                os.sep + "sced_hour_" + str(h) + ".lp"
                        lp_writer(current_sced_instance, current_sced_filename, lambda x: True,
                                  {"symbolic_solver_labels" : True})
                        print("SCED instance written to file=" + current_sced_filename)

                    if options.output_sced_initial_conditions:
                        print("")
                        output_sced_initial_condition(current_sced_instance, 
                                                      max_thermal_generator_label_length=max_thermal_generator_label_length)

                    if options.output_sced_demands:
                        print("")
                        output_sced_demand(current_sced_instance,
                                           max_bus_label_length=max_bus_label_length)
                
                    print("")
                    print("Solving SCED instance")
                    infeasibilities_detected_and_corrected = False

                    if options.output_solver_logs:
                        print("")
                        print("------------------------------------------------------------------------------")

                    sced_results = call_solver(sced_solver, 
                                               current_sced_instance, 
                                               tee=options.output_solver_logs, 
                                               keepfiles=options.keep_solver_files,
                                               **solve_options[sced_solver])

                    sced_runtimes.append(sced_results.solver.time)

                    if options.output_solver_logs:
                        print("")
                        print("------------------------------------------------------------------------------")
                        print("")

                    if sced_results.solution.status.key != "optimal":
                        print("SCED RESULTS STATUS=",sced_results.solution.status.key)
                        print("")
                        print("Failed to solve SCED optimization instance - no feasible solution exists!")
                        print("SCED RESULTS:", sced_results)


                        # for diagnostic purposes, save the failed SCED instance.
                        infeasible_sced_filename = options.output_directory + os.sep + str(this_date) + os.sep + \
                                                   "failed_sced_hour_" + str(h) + ".lp"
                        lp_writer(current_sced_instance, infeasible_sced_filename, lambda x: True,
                                  {"symbolic_solver_labels" : True})
                        print("Infeasible SCED instance written to file=" + infeasible_sced_filename)

                        # create a relaxed SCED instance, for manipulation purposes.
                        if options.run_deterministic_ruc:
                            relaxed_sced_instance = \
                                create_sced_instance(sced_model,
                                                     deterministic_ruc_instance_for_today, None,
                                                     scenario_tree_for_today,
                                                     deterministic_ruc_instance_for_tomorrow, None,
                                                     scenario_tree_for_tomorrow,
                                                     ruc_instance_to_simulate_today, prior_sced_instance,
                                                     actual_demand,
                                                     demand_forecast_error,
                                                     actual_min_renewables,
                                                     actual_max_renewables,
                                                     renewables_forecast_error,
                                                     h,
                                                     options.reserve_factor,
                                                     sced_horizon=options.sced_horizon,
                                                     initialize_from_ruc=initialize_from_ruc,
                                                     use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                     use_persistent_forecast_error=use_persistent_forecast_error_in_sced)
                        else:  # changed Mar 2015, GCS: Some arguments were missing.
                            relaxed_sced_instance = \
                                create_sced_instance(sced_model,
                                                     None, scenario_instances_for_today, scenario_tree_for_today,
                                                     None, scenario_instances_for_tomorrow, scenario_tree_for_tomorrow,
                                                     ruc_instance_to_simulate_today, prior_sced_instance,
                                                     actual_demand,
                                                     None,
                                                     actual_min_renewables,
                                                     actual_max_renewables,
                                                     None,
                                                     h,
                                                     options.reserve_factor,
                                                     sced_horizon=options.sced_horizon,
                                                     initialize_from_ruc=initialize_from_ruc,
                                                     use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                     use_persistent_forecast_error=use_persistent_forecast_error_in_sced)

                        if options.relax_ramping_if_infeasible:
                            print("")
                            print("Trying to solve SCED optimization instance with relaxed ramping constraints")

                            if True:  # ramp_rate_violations:

                                relax_iteration = 0
                                current_inflation_factor = 1.0

                                while True:

                                    relax_iteration += 1
                                    current_inflation_factor = current_inflation_factor * \
                                                               (1.0 + options.relax_ramping_factor)

                                    print("Relaxing nominal ramp rates - applying scale factor=" +
                                          str(options.relax_ramping_factor) + "; iteration=" + str(relax_iteration))
                                    relax_sced_ramp_rates(relaxed_sced_instance, options.relax_ramping_factor)
                                    print("Cumulative inflation factor=" + str(current_inflation_factor))

                                    # for diagnostic purposes, save the failed SCED instance.
                                    ramp_relaxed_sced_filename = options.output_directory + os.sep + str(this_date) + \
                                                                 os.sep + "ramp_relaxed_sced_hour_" + str(h) + \
                                                                 "_iter_" + str(relax_iteration) + ".lp"
                                    lp_writer(relaxed_sced_instance, ramp_relaxed_sced_filename, lambda x: True,
                                              {"symbolic_solver_labels" : True})
                                    print("Ramp-relaxed SCED instance written to file=" + ramp_relaxed_sced_filename)

                                    sced_results = call_solver(sced_solver,
                                                               relaxed_sced_instance, 
                                                               tee=options.output_solver_logs, 
                                                               keepfiles=options.keep_solver_files,
                                                               **solve_options[sced_solver])

                                    relaxed_sced_instance.load_from(sced_results)  # load so that we can report later...




                                    if sced_results.solution.status.key != "optimal":
                                        print("Failed to solve ramp-rate-relaxed SCED optimization instance - "
                                              "still no feasible solution exists!")
                                    else:
                                        break

                                    # the "20" is arbitrary, but the point is that there are situations - if the T0 power is hopelessly messed
                                    # up, where ramp rate relaxations may do nothing...
                                    if relax_iteration >= 20:
                                        raise RuntimeError("Halting - failed to solve ramp-rate-relaxed SCED optimization instance after 20 iterations")

                                print("Successfully solved ramp-rate-relaxed SCED - proceeding with simulation")
                                infeasibilities_detected_and_corrected = True

                            else:

                                raise RuntimeError("Halting - unknown root cause of SCED infeasibility, so no correction can be initiated.")

                        elif options.error_if_infeasible:
                            # TBD - modify this to generate the LP file automatically, for debug purposes.
                            # relaxed_sced_instance.pprint()
                            raise RuntimeError("Halting - option --halt-on-failed-solve enabled")
                        else:
                            # TBD - need to do something with respect to establishing the prior_sced_instance
                            #     - probably need to set the current instance to the previous instance,
                            #       to ensure the prior instance is always feasible, or at least has a solution.
                            print("WARNING: Continuing simulation despite infeasible SCED solution - watch out!")
                
    
                    # IMPORTANT: The sced results may yield infeasibilities in the current sced, due to relaxing of 
                    #            ramping constraints. It depends on which logic branch above was taken. 
                    current_sced_instance.solutions.load_from(sced_results)
                    
                    if options.enable_quick_start_generator_commitment:
                        
                        for g in current_sced_instance.QuickStartGenerators:
                            if current_sced_instance.UnitOn[g, 1]==1:
                                fixed_quick_start_generators_committed.append(g)

                        # check for load shedding
                        if round_small_values(sum(value(current_sced_instance.posLoadGenerateMismatch[b, 1]) for b in current_sced_instance.Buses))>0:
                        
                            # report solution/load shedding before unfixing Quick Start Generators
                            print("")
                            print("SCED Solution before unfixing Quick Start Generators")
                            print("")
                        
                            if infeasibilities_detected_and_corrected:
                                output_sced_ramp_violations(current_sced_instance, relaxed_sced_instance)
                                            
                            this_sced_without_quick_start_fixed_costs, \
                            this_sced_without_quick_start_variable_costs = report_costs_for_deterministic_sced(current_sced_instance)
                            print("")
                        
                            report_mismatches_for_deterministic_sced(current_sced_instance)
                            print("")
                        
                            report_renewables_curtailment_for_deterministic_sced(current_sced_instance)
                        
                            report_on_off_and_ramps_for_deterministic_sced(current_sced_instance)
                            print("")
                            
                            this_sced_without_quick_start_power_generated=sum(value(current_sced_instance.PowerGenerated[g,1]) for g in current_sced_instance.ThermalGenerators)

                            output_sced_solution(current_sced_instance, max_thermal_generator_label_length=max_thermal_generator_label_length)
                            print("")
                        
                            # re-solve the sced after unfixing Quick Start Generators that have not already been committed
                            print("")
                            print("Re-solving SCED after unfixing Quick Start Generators")
                            for t in sorted(current_sced_instance.TimePeriods):
                                for g in current_sced_instance.QuickStartGenerators:
                                    if current_sced_instance.UnitOn[g, t]==0:
                                        current_sced_instance.UnitOn[g,t].unfix()
                            load_shed_sced_results = call_solver(sced_solver,
                                                         current_sced_instance,
                                                         tee=options.output_solver_logs,
                                                         keepfiles=options.keep_solver_files,
                                                         **solve_options[sced_solver])
                            current_sced_instance.solutions.load_from(load_shed_sced_results)
                                
                        for g in current_sced_instance.QuickStartGenerators:
                            if current_sced_instance.UnitOn[g, 1]==1:
                                unfixed_quick_start_generators_committed.append(g)

                    print("Fixing binaries and solving for LMPs")
                    reference_model_module.fix_binary_variables(current_sced_instance)

                    reference_model_module.define_suffixes(current_sced_instance)

                    lmp_sced_results = call_solver(sced_solver,current_sced_instance,
                                                   tee=options.output_solver_logs,
                                                   keepfiles=options.keep_solver_files,
                                                   **solve_options[sced_solver])

                    if lmp_sced_results.solution.status.key != "optimal":
                        raise RuntimeError("Failed to solve LMP SCED")

                    current_sced_instance.solutions.load_from(lmp_sced_results)

                    reference_model_module.free_binary_variables(current_sced_instance)

                    print("")
                    print("Results:")
                    print("")

                    # if we had to correct infeasibilities by relaxing constraints, output diagnostics to
                    # help identify where the violations occurred.
                    # NOTE: we are not yet dealing with the scaled ramp limits - just the nominal ones.
                    if infeasibilities_detected_and_corrected:
                        output_sced_ramp_violations(current_sced_instance, relaxed_sced_instance)

                    this_sced_demand = value(current_sced_instance.TotalDemand[1])

                    max_hourly_demand = max(max_hourly_demand, this_sced_demand)

                    this_sced_fixed_costs, this_sced_variable_costs = report_costs_for_deterministic_sced(current_sced_instance)
                    print("")

                    this_sced_power_generated=sum(value(current_sced_instance.PowerGenerated[g,1]) for g in current_sced_instance.ThermalGenerators)

                    this_sced_load_shedding, \
                    this_sced_over_generation, \
                    this_sced_reserve_shortfall, \
                    this_sced_available_reserve, \
                    this_sced_available_quickstart = report_mismatches_for_deterministic_sced(current_sced_instance)

                    if len(current_sced_instance.TransmissionLines) > 0:
                        report_at_limit_lines_for_deterministic_sced(current_sced_instance,
                                                                     max_line_label_length=max_line_label_length)

                    report_lmps_for_deterministic_sced(current_sced_instance, max_bus_label_length=max_bus_label_length)

                    this_sced_renewables_available = sum(value(current_sced_instance.MaxNondispatchablePower[g, 1]) 
                                                         for g in current_sced_instance.AllNondispatchableGenerators)

                    this_sced_renewables_used = sum(value(current_sced_instance.NondispatchablePowerUsed[g, 1])
                                                    for g in current_sced_instance.AllNondispatchableGenerators)
                    this_sced_renewables_curtailment = \
                        report_renewables_curtailment_for_deterministic_sced(current_sced_instance)
                    this_sced_on_offs, this_sced_sum_on_off_ramps, this_sced_sum_nominal_ramps = \
                        report_on_off_and_ramps_for_deterministic_sced(current_sced_instance)
                                                
                    curtailments_by_hour.append(this_sced_renewables_curtailment)

                    reserve_requirements_by_hour.append(value(current_sced_instance.ReserveRequirement[1]))

                    load_shedding_by_hour.append(this_sced_load_shedding)
                    over_generation_by_hour.append(this_sced_over_generation)
                    reserve_shortfalls_by_hour.append(this_sced_reserve_shortfall)
                    available_reserves_by_hour.append(this_sced_available_reserve)

                    available_quickstart_by_hour.append(this_sced_available_quickstart)

                    if this_sced_over_generation > 0.0:
                        event_annotations.append((h, 'Over Generation'))
                    if this_sced_load_shedding > 0.0:
                        event_annotations.append((h, 'Load Shedding'))
                    if this_sced_reserve_shortfall > 0.0:
                        event_annotations.append((h, 'Reserve Shortfall'))

                    # 0 demand can happen, in some odd circumstances (not at the ISO level!).
                    if this_sced_demand != 0.0:
                        this_sced_price = (this_sced_fixed_costs + this_sced_variable_costs) / this_sced_demand
                    else:
                        this_sced_price = 0.0

                    # Track difference in costs/power generated with and without quick start generators
                    if options.enable_quick_start_generator_commitment:
                        if round_small_values(sum(value(current_sced_instance.posLoadGenerateMismatch[b, 1]) for b in current_sced_instance.Buses))>0:
                            this_sced_quick_start_additional_costs=round_small_values(this_sced_fixed_costs+this_sced_variable_costs-this_sced_without_quick_start_fixed_costs-this_sced_without_quick_start_variable_costs)
                            this_sced_quick_start_additional_power_generated=this_sced_power_generated-this_sced_without_quick_start_power_generated
                        else:
                            this_sced_quick_start_additional_costs=0.0
                            this_sced_quick_start_additional_power_generated=0.0
                    else:
                        this_sced_quick_start_additional_costs=0.0
                        this_sced_quick_start_additional_power_generated=0.0

                    quick_start_additional_costs_by_hour.append(this_sced_quick_start_additional_costs)
                    quick_start_additional_power_generated_by_hour.append(this_sced_quick_start_additional_power_generated)
                    
                    # Track quick start generators used, if any
                    if options.enable_quick_start_generator_commitment:
                        for g in current_sced_instance.QuickStartGenerators:
                            if g in unfixed_quick_start_generators_committed and g not in fixed_quick_start_generators_committed:
                                used_as_quick_start[g].append(1)
                            else:
                                used_as_quick_start[g].append(0)
                        fixed_quick_start_generators_committed=[]
                        unfixed_quick_start_generators_committed=[]
                    
                    
                    # write summary statistics to the hourly CSV file.
                    print(this_date, ",", h+1, ",", this_sced_fixed_costs + this_sced_variable_costs, ",",
                          this_sced_fixed_costs, ",", this_sced_variable_costs, ",", this_sced_load_shedding,
                          ",", this_sced_over_generation, ",", this_sced_reserve_shortfall, ",",
                          this_sced_renewables_used, ",", this_sced_renewables_curtailment, ",",
                          this_sced_demand, ",", this_sced_price, file=csv_hourly_output_file)

                    # # Feb 2015, GCS # #
                    # to plot the demand
                    try:
                        if len(demand_list) == 25:
                            foo
                        demand_list.append(this_sced_demand)
                    except:
                        demand_list = [this_sced_demand]
                    #####################
                    
                    this_date_demand += this_sced_demand
                    this_date_fixed_costs += this_sced_fixed_costs
                    this_date_variable_costs += this_sced_variable_costs
                    this_date_over_generation += this_sced_over_generation
                    this_date_load_shedding += this_sced_load_shedding
                    this_date_reserve_shortfall += this_sced_reserve_shortfall
                    this_date_renewables_available += this_sced_renewables_available
                    this_date_renewables_used += this_sced_renewables_used
                    this_date_renewables_curtailment += this_sced_renewables_curtailment
                    this_date_on_offs += this_sced_on_offs
                    this_date_sum_on_off_ramps += this_sced_sum_on_off_ramps
                    this_date_sum_nominal_ramps += this_sced_sum_nominal_ramps
                    this_date_quick_start_additional_costs += this_sced_quick_start_additional_costs
                    this_date_quick_start_additional_power_generated += this_sced_quick_start_additional_power_generated

                    if options.output_sced_solutions:
                        print("")
                        output_sced_solution(current_sced_instance, max_thermal_generator_label_length=max_thermal_generator_label_length)

                    # Create new row for hourly dataframe, all generators

                    for g in current_sced_instance.ThermalGenerators:
                        observed_thermal_dispatch_levels[g][h] = value(current_sced_instance.PowerGenerated[g, 1])

                    for g in current_sced_instance.ThermalGenerators:
                        observed_thermal_headroom_levels[g][h] = max(value(current_sced_instance.MaximumPowerAvailable[g,1]) - value(current_sced_instance.PowerGenerated[g, 1]),0.0)

                    for g in current_sced_instance.ThermalGenerators:
                        observed_thermal_states[g][h] = value(current_sced_instance.UnitOn[g, 1])

                    for g in current_sced_instance.AllNondispatchableGenerators:
                        observed_renewables_levels[g][h] = value(current_sced_instance.NondispatchablePowerUsed[g, 1])

                    for g in current_sced_instance.AllNondispatchableGenerators:
                        observed_renewables_curtailment[g][h] = value(current_sced_instance.MaxNondispatchablePower[g, 1]) - value(current_sced_instance.NondispatchablePowerUsed[g, 1])

                    for l in current_sced_instance.TransmissionLines:
                        observed_flow_levels[l][h] = value(current_sced_instance.LinePower[l, 1])

                    for b in current_sced_instance.Buses:
                        if value(current_sced_instance.LoadGenerateMismatch[b, 1]) >= 0.0:
                            observed_bus_mismatches[b][h] = value(current_sced_instance.posLoadGenerateMismatch[b, 1])
                        else:
                            observed_bus_mismatches[b][h] = -1.0 * value(current_sced_instance.negLoadGenerateMismatch[b, 1])

                    for b in current_sced_instance.Buses:
                        observed_bus_LMPs[b][h] = value(current_sced_instance.dual[current_sced_instance.PowerBalance[b, 1]])

                    for s in current_sced_instance.Storage:
                        storage_input_dispatchlevelsdict[s][h] = \
                            np.array([value(current_sced_instance.PowerInputStorage[s, 1])])
                        
                    for s in current_sced_instance.Storage:
                        storage_output_dispatchlevelsdict[s][h] = \
                            np.array([value(current_sced_instance.PowerOutputStorage[s, 1])])
                    
                    for s in current_sced_instance.Storage:
                        storage_soc_dispatchlevelsdict[s][h] = \
                            np.array([value(current_sced_instance.SocStorage[s, 1])])
                        print("")
                        print("Current State-Of-Charge Value: ", value(current_sced_instance.SocStorage[s,1]))
                        print("SOC Value at T0:", value(current_sced_instance.StorageSocOnT0[s]))
                    print("")
                    print(("Construction and solve time=%.2f seconds" % (time.time() - start_time)))

        this_date_renewables_penetration_rate = (float(this_date_renewables_used) / float(this_date_demand) * 100.0)

        this_date_average_price = (this_date_fixed_costs + this_date_variable_costs) / this_date_demand
    
        cumulative_demand += this_date_demand
        cumulative_renewables_used += this_date_renewables_used
        
        overall_renewables_penetration_rate = (float(cumulative_renewables_used) / float(cumulative_demand) * 100.0)
        
        cumulative_average_price = (total_overall_fixed_costs + total_overall_generation_costs) / cumulative_demand
        
        new_quickstart_df_entries = []
        for h in range (0,24):
            for g in current_sced_instance.ThermalGenerators:
                if g in used_as_quick_start:
                    this_gen_used_as_quick_start=used_as_quick_start[g][h]
                    this_gen_quick_start_dispatch=observed_thermal_dispatch_levels[g][h]
                    new_quickstart_df_entries.append({'Date':this_date,
                                                      'Hour':h+1,
                                                      'Generator':g,
                                                      'Used as quickstart':this_gen_used_as_quick_start,
                                                      'Dispatch level of quick start generator':this_gen_quick_start_dispatch})
        quickstart_summary_df = pd.concat([quickstart_summary_df, pd.DataFrame.from_records(new_quickstart_df_entries)])

        new_thermal_generator_dispatch_entries = []
        for h in range(0, 24):
            for g in current_sced_instance.ThermalGenerators:
                new_thermal_generator_dispatch_entries.append({'Date':this_date,
                                                               'Hour':h+1,
                                                               'Generator':g,
                                                               'Dispatch':observed_thermal_dispatch_levels[g][h],
                                                               'Headroom':observed_thermal_headroom_levels[g][h],
                                                               'Unit State': observed_thermal_states[g][h]})
        thermal_generator_dispatch_df = pd.concat([thermal_generator_dispatch_df, pd.DataFrame.from_records(new_thermal_generator_dispatch_entries)])                                                         

        new_renewables_production_entries = []
        for h in range(0, 24):
            for g in current_sced_instance.AllNondispatchableGenerators:
                new_renewables_production_entries.append({'Date':this_date,
                                                          'Hour':h+1,
                                                          'Generator':g,
                                                          'Output':observed_renewables_levels[g][h],
                                                          'Curtailment':observed_renewables_curtailment[g][h]})
        renewables_production_df = pd.concat([renewables_production_df, pd.DataFrame.from_records(new_renewables_production_entries)])

        new_line_entries = []
        for h in range(0, 24):
            for l in current_sced_instance.TransmissionLines:
                new_line_entries.append({'Date':this_date,
                                         'Hour':h+1,
                                         'Line':l,
                                         'Flow':observed_flow_levels[l][h]})
        line_df = pd.concat([line_df, pd.DataFrame.from_records(new_line_entries)])

        new_bus_entries = []
        for h in range(0, 24):
            for b in current_sced_instance.Buses:
                this_mismatch = observed_bus_mismatches[b][h]
                if this_mismatch >= 0.0:
                    shortfall = this_mismatch
                    overgeneration = 0.0
                else:
                    shortfall = 0.0
                    overgeneration = -1.0 * this_mismatch
                new_bus_entries.append({'Date':this_date,
                                        'Hour':h+1,
                                        'Bus':b,
                                        'Shortfall':shortfall,
                                        'Overgeneration':overgeneration,
                                        'LMP':observed_bus_LMPs[b][h]})
        bus_df = pd.concat([bus_df, pd.DataFrame.from_records(new_bus_entries)])
        
        new_hourly_gen_summary_entries = []
        for h in range(0, 24):
            new_hourly_gen_summary_entries.append({'Date':this_date,
                                                   'Hour':h+1,
                                                   'Load shedding':load_shedding_by_hour[h],
                                                   'Reserve shortfall':reserve_shortfalls_by_hour[h],
                                                   'Available reserves':available_reserves_by_hour[h],
                                                   'Over generation':over_generation_by_hour[h]})
        hourly_gen_summary_df = pd.concat([hourly_gen_summary_df, pd.DataFrame.from_records(new_hourly_gen_summary_entries)])

        runtime_df = runtime_df.append({'Date':this_date,
                                        'Hour':-1,
                                        'Type':'RUC',
                                        'Solve Time':0.0},
                                       ignore_index = True)

        for h in range(0, 24):
            runtime_df = runtime_df.append({'Date':this_date,
                                            'Hour':h+1,
                                            'Type':'SCED',
                                            'Solve Time':sced_runtimes[h]},
                                           ignore_index = True)

        # summarize daily costs / statistics
        print("")
        print("Date %s total demand:                                 %12.2f" % (str(this_date), this_date_demand))
        print("Date %s total renewables available:                   %12.2f" % (str(this_date), this_date_renewables_available))
        print("Date %s total renewables used:                        %12.2f" % (str(this_date), this_date_renewables_used))
        print("Date %s renewables penetration rate:                  %12.2f" % (str(this_date), this_date_renewables_penetration_rate))
        print("Date %s average price:                                %12.6f" % (str(this_date), this_date_average_price))

        print("")

        print("Date %s total fixed costs:                            %12.2f" % (str(this_date), this_date_fixed_costs))
        print("Date %s total generation costs:                       %12.2f" % (str(this_date), this_date_variable_costs))
        print("Date %s total load shedding:                          %12.2f" % (str(this_date), this_date_load_shedding))
        print("Date %s total over generation:                        %12.2f" % (str(this_date), this_date_over_generation))
        print("Date %s total reserve shortfall                       %12.2f" % (str(this_date), this_date_reserve_shortfall))
        print("Date %s total renewables curtailment:                 %12.2f" % (str(this_date), this_date_renewables_curtailment))
        print("Date %s total on/offs:                                %12d"   % (str(this_date), this_date_on_offs))
        print("Date %s total sum on/off ramps:                       %12.2f" % (str(this_date), this_date_sum_on_off_ramps))
        print("Date %s total sum nominal ramps:                      %12.2f" % (str(this_date), this_date_sum_nominal_ramps))
        print("Date %s total quick start additional costs:           %12.2f" % (str(this_date), this_date_quick_start_additional_costs))
        print("Date %s total quick start additional power generated: %12.2f" % (str(this_date), this_date_quick_start_additional_power_generated))
        

        # update overall simulation costs / statistics
        total_overall_fixed_costs += this_date_fixed_costs
        total_overall_generation_costs += this_date_variable_costs
        total_overall_load_shedding += this_date_load_shedding
        total_overall_over_generation += this_date_over_generation
        total_overall_reserve_shortfall += this_date_reserve_shortfall

        total_overall_renewables_curtailment += this_date_renewables_curtailment
        total_on_offs += this_date_on_offs
        total_sum_on_off_ramps += this_date_sum_on_off_ramps
        total_sum_nominal_ramps += this_date_sum_nominal_ramps
        total_quick_start_additional_costs += this_date_quick_start_additional_costs
        total_quick_start_additional_power_generated += this_date_quick_start_additional_power_generated
        
        daily_total_costs.append(this_date_fixed_costs+this_date_variable_costs)
        daily_fixed_costs.append(this_date_fixed_costs)
        daily_generation_costs.append(this_date_variable_costs)
        daily_load_shedding.append(this_date_load_shedding)
        daily_over_generation.append(this_date_over_generation)
        daily_reserve_shortfall.append(this_date_reserve_shortfall)

        daily_renewables_available.append(this_date_renewables_available)
        daily_renewables_used.append(this_date_renewables_used)
        daily_renewables_curtailment.append(this_date_renewables_curtailment)
        daily_on_offs.append(this_date_on_offs)
        daily_sum_on_off_ramps.append(this_date_sum_on_off_ramps)
        daily_sum_nominal_ramps.append(this_date_sum_nominal_ramps)
        daily_quick_start_additional_costs.append(this_date_quick_start_additional_costs)
        daily_quick_start_additional_power_generated.append(this_date_quick_start_additional_power_generated)

        daily_average_price.append(this_date_average_price)

        daily_demand.append(this_date_demand)

        #print out the dataframe/generator detail

        if len(offline_generators) > 0:
            print("Generators knocked offline today:", offline_generators)
            reset_offline_elements(sced_inst)
            for g in offline_generators:
                sced_inst.UnitOnT0[g] = int(round(ruc_inst.UnitOn[g, 1]))
                sced_inst.PowerGeneratedT0[g] = ruc_inst.PowerGenerated[g, 1]
            offline_generators = []


        # we always generate plots - but we optionally display them in the simulation loop.
        generator_dispatch_levels = {}

        # we need a map between generator name and type, where the latter is a single
        # character, like 'C' or 'N'.
        generator_types = {}

        if not hasattr(current_sced_instance, "ThermalGeneratorType"):
            print("***SCED instance does not have \"ThermalGeneratorType\" attribute - "
                  "required for stack graph plot generation")
            sys.exit(1)

        if not hasattr(current_sced_instance, "NondispatchableGeneratorType"):
            print("***SCED instance does not have \"NondispatchableGeneratorType\" attribute - "
                  "required for stack graph plot generation")
            sys.exit(1)

        for g in current_sced_instance.ThermalGenerators:
            generator_dispatch_levels[g] = observed_thermal_dispatch_levels[g]
            generator_types[g] = current_sced_instance.ThermalGeneratorType[g]

        for g in current_sced_instance.AllNondispatchableGenerators:
            generator_dispatch_levels[g] = observed_renewables_levels[g]
            generator_types[g] = current_sced_instance.NondispatchableGeneratorType[g]

        plot_peak_demand = thermal_fleet_capacity
        if options.plot_peak_demand > 0.0:
            plot_peak_demand = options.plot_peak_demand

        for x in range(0, 1):
            daily_summary_df = daily_summary_df.append({'Date':this_date,
                             'Demand':this_date_demand,
                             'Renewables available':this_date_renewables_available,
                             'Renewables used':this_date_renewables_used,
                             'Renewables penetration rate':this_date_renewables_penetration_rate,
                             'Average price':this_date_average_price,
                             'Fixed costs':this_date_fixed_costs,
                             'Generation costs':this_date_variable_costs,
                             'Load shedding':this_date_load_shedding,
                             'Over generation':this_date_over_generation,
                             'Reserve shortfall':this_date_reserve_shortfall,
                             'Renewables curtailment':this_date_renewables_curtailment,
                             'Number on/offs':this_date_on_offs,
                             'Sum on/off ramps':this_date_sum_on_off_ramps,
                             'Sum nominal ramps':this_date_sum_nominal_ramps},
                             ignore_index=True)



        graphutils.generate_stack_graph(plot_peak_demand,  # for scale of the plot
                                        generator_types,
                                        generator_dispatch_levels,
                                        reserve_requirements_by_hour,
                                        this_date, 
                                        curtailments_by_hour,  # is not used, Feb 2015
                                        load_shedding_by_hour,
                                        reserve_shortfalls_by_hour,
                                        available_reserves_by_hour,
                                        available_quickstart_by_hour,
                                        over_generation_by_hour,
                                        max_hourly_demand,
                                        quick_start_additional_power_generated_by_hour,
                                        annotations=event_annotations, 
                                        display_plot=options.display_plots, 
                                        show_plot_legend=(not options.disable_plot_legend),
                                        savetofile=True, 
                                        output_directory=os.path.join(options.output_directory, "plots"),
                                        plot_individual_generators=options.plot_individual_generators,
                                        renewables_penetration_rate=this_date_renewables_penetration_rate,
                                        fixed_costs=this_date_fixed_costs,
                                        variable_costs=this_date_variable_costs,
                                        demand=demand_list)  # Feb 2015, GCS: To plot the demand

        if len(storage_soc_dispatchlevelsdict) > 0:
            storagegraphutils.generate_storage_graph(storage_input_dispatchlevelsdict, 
                                                     storage_output_dispatchlevelsdict, 
                                                     storage_soc_dispatchlevelsdict, 
                                                     this_date, 
                                                     save_to_file=True, 
                                                     display_plot=options.display_plots,
                                                     plot_individual_generators=False,
                                                     output_directory=os.path.join(options.output_directory, "plots"))



    print("")
    print("Simulation complete!")




    print("")
    print("Total demand:                        %12.2f" % (cumulative_demand))
    print("")
    print("Total fixed costs:                   %12.2f" % (total_overall_fixed_costs))
    print("Total generation costs:              %12.2f" % (total_overall_generation_costs))
    print("Total costs:                         %12.2f" % (total_overall_fixed_costs + total_overall_generation_costs))
    print("")
    print("Total load shedding:                 %12.2f" % (total_overall_load_shedding))
    print("Total over generation:               %12.2f" % (total_overall_over_generation))
    print("Total reserve shortfall:             %12.2f" % (total_overall_reserve_shortfall))
    print("")
    print("Total renewables curtailment:        %12.2f" % (total_overall_renewables_curtailment))
    print("")
    print("Total on/offs:                       %12d"   % (total_on_offs))
    print("Total sum on/off ramps:              %12.2f" % (total_sum_on_off_ramps))
    print("Total sum nominal ramps:             %12.2f" % (total_sum_nominal_ramps))
    print("")
    print("Total quick start additional costs   %12.2f" % (total_quick_start_additional_costs))
    print("Total quick start additional power   %12.2f" % (total_quick_start_additional_power_generated))
    print("")
    print("Maximum observed demand:             %12.2f" % (max_hourly_demand))
    print("")
    print("Overall renewables penetration rate: %12.2f" % (overall_renewables_penetration_rate))
    print("")
    print("Cumulative average price:            %12.6f" % (cumulative_average_price))
    
    overall_simulation_output_df = overall_simulation_output_df.append(
        {'Total demand':cumulative_demand,
         'Total fixed costs':total_overall_fixed_costs,
         'Total generation costs':total_overall_generation_costs,
         'Total costs':total_overall_fixed_costs + total_overall_generation_costs,
         'Total load shedding':total_overall_load_shedding,
         'Total over generation':total_overall_over_generation,
         'Total reserve shortfall':total_overall_reserve_shortfall,
         'Total renewables curtialment':total_overall_renewables_curtailment,
         'Total on/offs':total_on_offs,
         'Total sum on/off ramps':total_sum_on_off_ramps,
         'Total sum nominal ramps':total_sum_nominal_ramps,
         'Maximum observed demand':max_hourly_demand,
         'Overall renewables penetration rate':overall_renewables_penetration_rate,
         'Cumulative average price':cumulative_average_price},
        ignore_index=True)

    #Create csv files for dataframes

    daily_summary_df.to_csv(os.path.join(options.output_directory, 'Daily_summary.csv'), index = False)
    
    options_df.to_csv(os.path.join(options.output_directory, 'Options.csv'), index = False)
    
    thermal_generator_dispatch_df.to_csv(os.path.join(options.output_directory, 'thermal_detail.csv'), index = False)

    renewables_production_df.to_csv(os.path.join(options.output_directory, 'renewables_detail.csv'), index = False)

    line_df.to_csv(os.path.join(options.output_directory, 'line_detail.csv'), index = False)

    bus_df.to_csv(os.path.join(options.output_directory, 'bus_detail.csv'), index = False)
    
    overall_simulation_output_df.to_csv(os.path.join(options.output_directory, 'Overall_simulation_output.csv'), index = False)
    
    quickstart_summary_df.to_csv(os.path.join(options.output_directory, 'Quickstart_summary.csv'), index = False)
    
    hourly_gen_summary_df.to_csv(os.path.join(options.output_directory, 'Hourly_gen_summary.csv'), index = False)

    runtime_df.to_csv(os.path.join(options.output_directory, 'runtimes.csv'), index = False)    

    graphutils.generate_cost_summary_graph(daily_fixed_costs, daily_generation_costs,
                                           daily_load_shedding, daily_over_generation,
                                           daily_reserve_shortfall, 
                                           daily_renewables_curtailment,
                                           display_plot=options.display_plots,
                                           save_to_file=True,
                                           output_directory=os.path.join(options.output_directory, "plots"))

    simulation_end_time = time.time()
    print("")
    print(("Total simulation run time=%.2f seconds" % (simulation_end_time - simulation_start_time)))

    # create a movie from the set of individual daily png files
    pyutilib.services.register_executable("ffmpeg")
    ffmpeg_executable = pyutilib.services.registered_executable("ffmpeg")
    if ffmpeg_executable == None:
        print("The executable ffmpeg is not installed - could not create movie of stack graphs")
    else:
        movie_filename = os.path.join(options.output_directory, "plots", "stackgraph_movie.mp4")
        if os.path.exists(movie_filename):
            os.remove(movie_filename)
        if os.path.exists("out.mp4"):
            os.remove("out.mp4")
    
        execution_string = "%s -r 1/2 -pattern_type glob -i " % str(ffmpeg_executable.get_path())
        execution_string += "'" + str(os.path.join(options.output_directory, "plots")) + os.sep + "stackgraph*.png" + "'"
        execution_string += " out.mp4"
        os.system(execution_string)
        shutil.move("out.mp4",movie_filename)

        print("Stackgraph movie written to file=" + movie_filename)

    ###################################################################
    # # output the daily summary statistics to a CSV file           # #
    ###################################################################

    csv_daily_output_filename = os.path.join(options.output_directory, "daily_summary.csv")
    csv_daily_output_file = open(csv_daily_output_filename, "w")
    sim_dates = [this_date for this_date in dates_to_simulate]
    print("Date", ",", "TotalCosts", ",", "FixedCosts", ",", "VariableCosts", ",", "LoadShedding", ",",
          "OverGeneration", ",", "ReserveShortfall", ",", "RenewablesAvailable", ",", "RenewablesUsed", ",", "RenewablesCurtailed", ",",
          "Demand", ",", "AveragePrice", ",", "OnOffs", ",", "SumOnOffRamps", ",", "SumNominalRamps",
          file=csv_daily_output_file)
    for i in range(0,len(sim_dates)):
        this_date = sim_dates[i]
        print(this_date, ",", daily_total_costs[i], ",", daily_fixed_costs[i], ",", daily_generation_costs[i], ",",
              daily_load_shedding[i], ",", daily_over_generation[i], ",", daily_reserve_shortfall[i], ",",
              daily_renewables_available[i], ",", daily_renewables_used[i], ",", daily_renewables_curtailment[i], ",", daily_demand[i], ",",
              daily_average_price[i], ",", daily_on_offs[i], ",", daily_sum_on_off_ramps[i], ",",
              daily_sum_nominal_ramps[i], file=csv_daily_output_file)
    csv_daily_output_file.close()
    print("")
    print("CSV daily summary written to file=" + csv_daily_output_filename)

    #########################################
    # # close the hourly summary CSV file # #
    #########################################

    csv_hourly_output_file.close()
    print("")
    print("CSV hourly summary written to file=" + csv_hourly_output_filename)


###############################################################################
#########################        END PRESCIENT      ###########################
###############################################################################


###############################################################################
#########################     PRESCINT MAIN    ###############################
###############################################################################

def main_prescient(options):
    
    ans = None

    if pstats_available and options.profile > 0:
        #
        # Call the main ef writer with profiling.
        #
        tfile = pyutilib.services.TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx('simulate(options)', globals(), locals(), tfile)
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        p = p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('cumulative', 'calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        pyutilib.services.TempfileManager.clear_tempfiles()
        ans = [tmp, None]

    else:
        if options.traceback is True:
            ans = simulate(options)
        else:
            errmsg = None
            try:
                ans = simulate(options)
            except ValueError:
                err = sys.exc_info()[1]
                errmsg = 'VALUE ERROR: %s' % err
            except KeyError:
                err = sys.exc_info()[1]
                errmsg = 'KEY ERROR: %s' % err
            except TypeError:
                err = sys.exc_info()[1]
                errmsg = 'TYPE ERROR: %s' % err
            except NameError:
                err = sys.exc_info()[1]
                errmsg = 'NAME ERROR: %s' % err
            except IOError:
                err = sys.exc_info()[1]
                errmsg = 'I/O ERROR: %s' % err
            except ConverterError:
                err = sys.exc_info()[1]
                errmsg = 'CONVERSION ERROR: %s' % err                
            except RuntimeError:
                err = sys.exc_info()[1]
                errmsg = 'RUN-TIME ERROR: %s' % err
            except pyutilib.common.ApplicationError:
                err = sys.exc_info()[1]
                errmsg = 'APPLICATION ERROR: %s' % err
            except Exception:
                err = sys.exc_info()[1]
                errmsg = 'UNKNOWN ERROR: %s' % err
                traceback.print_exc()

            if errmsg is not None:
                sys.stderr.write(errmsg+'\n')

    return ans


def main(args=None):

    if args is None:
        args = sys.argv

    #
    # Parse command-line options.
    #
    try:
        options_parser, guiOverride = MasterOptions.construct_options_parser()
        (options, args) = options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return
        
    main_prescient(options)


# MAIN ROUTINE STARTS NOW #
if __name__ == '__main__':
    result = main(sys.argv)
