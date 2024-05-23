# carrionarroyov3
########################################################################################################
# a basic (thermal) unit commitment model, drawn from:                                                 #
# A Computationally Efficient Mixed-Integer Linear Formulation for the Thermal Unit Commitment Problem #
# Miguel Carrion and Jose M. Arroyo                                                                    #
# IEEE Transactions on Power Systems, Volume 21, Number 3, August 2006.                                #
########################################################################################################

# JPW's model modified by DLW to have transmission constraints
# comments for these items have an (S) for security
# constraints now have penalties such as load loss, fast ramping, etc.

# WIND:
# wind is basically a generator, set the upper and lower bounds to be the same to make it non-dispatchable
# not doing it as negative load (but this has the same effect if it is non-dispatch....)
# as of 9 Feb 2012, wind does not appear in the objective function on the assumption that it is a sunk cost

# STORAGE:
# Storage has properties of load and generator. It functions as generator (discharge) when its power OUTPUT
# is > 0, and it acts as a load (charging) when its power INPUT is > 0.
# As of 21 Feb 2012, storage does not appear in the objective function on the assumption that it is a sunk
# cost and that it is operated by the ISO/RTO, so the energy employed to charge the storage is paid for by
# the Load Serving Entities (LSEs)
# As of 21 Feb 2012, energy storage is not included as a source of reserves
# As of 21 Feb 2012, energy storage assummes hourly time periods - energy conservation constraint
# As of 21 Feb 2012, energy storage assummes 20 time periods (NoNondispatchableStorage.dat) - end-point constraint
# Minimum input/output power must be set 0.0, otherwise there could be errors
# As of 22 Feb 2012, energy storage has linear losses (round trip efficiency) - energy conservation

# CASM - Alstom validation
# Changes made to fit Alstom5Bus_UC test case: CASM 12/17/2012
# Ignore initial conditions for power generated
# Ignore start-up and shut-down ramp rates
# For UC: Load is given by zone

# Other optional changes:
# Make Pmin and Pmax for each generator vary through time (optional)
# Include hot and cold start-up costs and times (optional)
# Ignore ramp rates between hours for the UC (optional)

# TODO:
# Finish including TMSR, TMNSR and TMO
# Include penalty mismatch factors for not meeting reserve requirements (also in ED)
# Review Regulation cost calculation ($/On vs. $/MWh)

# CASM comments ends here

from __future__ import division
from coopr.pyomo import *
import math
import six

# in the Alstom UC model, ramp rates are not enforced for time period T=1. we 
# don't want to do this by default, but we need a switch to enable this feature
# to allow for regression against the Alstom baseline results.
enforce_t1_ramp_rates = True

# If we want to run the multistage version of our model we have to set multistage = True
multistage = True

# flag to enable regulation ancillary service components of the model.
regulation_services = False

# flag to enable reserve ancillary service components of the model.
reserve_services = False

# flag to enable storage components of the model.
storage_services = False

#########
# Model #
#########

model = AbstractModel()

model.name = "carrionarroyov3"

#
# Parameters
#

##############################################
# string indentifiers for the set of busses. #
##############################################

model.Buses = Set()

###################
#   Load Zones    #
###################
#Aggregated loads are distributed in the system based on load coefficient values

model.Zones = Set(initialize=['SingleZone'])

def buildBusZone(m):
    an_element = six.next(m.Zones.__iter__())
    if len(m.Zones) == 1 and an_element == 'SingleZone':
        for b in m.Buses:
            m.BusZone[b] = an_element
    else:
        print("Multiple buses is not supported by buildBusZone in ReferenceModel.py -- someone should fix that!")
        exit(1)

model.BusZone = Param(model.Buses, mutable=True)
model.BuildBusZone = BuildAction(rule=buildBusZone)

model.LoadCoefficient = Param(model.Buses, default=0.0)

def total_load_coefficient_per_zone(m, z):
    return sum(m.LoadCoefficient[b] for b in m.Buses if str(value(m.BusZone[b]))==str(z))
model.TotalLoadCoefficientPerZone = Param(model.Zones, initialize=total_load_coefficient_per_zone)

def load_factors_per_bus(m,b):
    if (m.TotalLoadCoefficientPerZone[value(m.BusZone[b])] != 0.0):
        return m.LoadCoefficient[b]/m.TotalLoadCoefficientPerZone[value(m.BusZone[b])]
    else:
        return 0.0
model.LoadFactor = Param(model.Buses, initialize=load_factors_per_bus, within=NonNegativeReals)

################################

model.StageSet = Set(ordered=True) 

# IMPORTANT: The stage set must be non-empty - otherwise, zero costs result.
def check_stage_set(m):
   return (len(m.StageSet) != 0)
model.CheckStageSet = BuildCheck(rule=check_stage_set)

model.TimePeriodLength = Param(default=1.0)
model.NumTimePeriods = Param(within=PositiveIntegers, mutable=True)

model.InitialTime = Param(within=PositiveIntegers, default=1)
model.TimePeriods = RangeSet(model.InitialTime, model.NumTimePeriods)

# the following sets must must come from the data files or from an initialization function that uses 
# a parameter that tells you when the stages end (and that thing needs to come from the data files)
model.CommitmentTimeInStage = Set(model.StageSet, within=model.TimePeriods) 
model.GenerationTimeInStage = Set(model.StageSet, within=model.TimePeriods)

model.CommitmentStageCost = Var(model.StageSet, within=NonNegativeReals)
model.GenerationStageCost = Var(model.StageSet, within=NonNegativeReals)

model.StageCost = Var(model.StageSet, within=NonNegativeReals)

##############################################
# Network definition (S)
##############################################

model.NumTransmissionLines = Param(default=0)
model.TransmissionLines = RangeSet(model.NumTransmissionLines)

model.BusFrom = Param(model.TransmissionLines)
model.BusTo   = Param(model.TransmissionLines)

def derive_connections_to(m, b):
   return (l for l in m.TransmissionLines if m.BusTo[l]==b)
model.LinesTo = Set(model.Buses, initialize=derive_connections_to)  # derived from TransmissionLines

def derive_connections_from(m, b):
   return (l for l in m.TransmissionLines if m.BusFrom[l]==b)
model.LinesFrom = Set(model.Buses, initialize=derive_connections_from)  # derived from TransmissionLines


model.Reactance = Param(model.TransmissionLines)
def get_b_from_Reactance(m, l):
    return 1/float(m.Reactance[l])
model.B = Param(model.TransmissionLines, initialize=get_b_from_Reactance) # Susceptance (1/Reactance; usually 1/x)
model.ThermalLimit = Param(model.TransmissionLines) # max flow across the line

##########################################################
# string indentifiers for the set of thermal generators. #
# and their locations. (S)                               #
##########################################################

model.ThermalGenerators = Set()
model.ThermalGeneratorsAtBus = Set(model.Buses)

# thermal generator types must be specified as 'N', 'C', 'G', and 'H',
# with the obvious interpretation.
# TBD - eventually add a validator.

model.ThermalGeneratorType = Param(model.ThermalGenerators, within=Any, default='C')

# could be made more efficient through a build rule, if it would matter.
def thermal_generator_bus_rule(m, g):
   for b in m.Buses:
      if g in m.ThermalGeneratorsAtBus[b]:
         return b
   print("SERIOUS PROBLEM ENCOUNTERED WHEN INSTANTIATING UC MODEL - NO BUS ASSIGNED TO THERMAL GENERATOR="+str(g))
   return None

model.ThermalGeneratorBus = Param(model.ThermalGenerators, within=model.Buses, initialize=thermal_generator_bus_rule)

model.QuickStartGenerators = Set(within=model.ThermalGenerators)

def nd_gen_init(m,b):
    return []
model.NondispatchableGeneratorsAtBus = Set(model.Buses, initialize=nd_gen_init) 

def NonNoBus_init(m):
    retval = set()
    for b in m.Buses:
        retval = retval.union([gen for gen in m.NondispatchableGeneratorsAtBus[b]])
    return retval

model.AllNondispatchableGenerators = Set(initialize=NonNoBus_init)

######################
#   Reserve Zones    #
######################

# Generators are grouped in zones to provide zonal reserve requirements. #
# All generators can contribute to global reserve requirements           #

model.ReserveZones = Set()
model.ZonalReserveRequirement = Param(model.ReserveZones, model.TimePeriods, default=0.0, mutable=True, within=NonNegativeReals)
model.ReserveZoneLocation = Param(model.ThermalGenerators)

def form_thermal_generator_reserve_zones(m,rz):
    return (g for g in m.ThermalGenerators if m.ReserveZoneLocation[g]==rz)
model.ThermalGeneratorsInReserveZone = Set(model.ReserveZones, initialize=form_thermal_generator_reserve_zones)

#################################################################
# the global system demand, for each time period. units are MW. #
# demand as at busses (S) so total demand is derived            #
#################################################################

# at the moment, we allow for negative demand. this is probably
# not a good idea, as "stuff" representing negative demand - including
# renewables, interchange schedules, etc. - should probably be modeled
# explicitly.

# Demand can also be given by Zones

model.DemandPerZone = Param(model.Zones, model.TimePeriods, default=0.0, mutable=True)

# Convert demand by zone to demand by bus
def demand_per_bus_from_demand_per_zone(m,b,t):
    return m.DemandPerZone[value(m.BusZone[b]), t] * m.LoadFactor[b]
model.Demand = Param(model.Buses, model.TimePeriods, initialize=demand_per_bus_from_demand_per_zone, mutable=True)

def calculate_total_demand(m, t):
    return sum(value(m.Demand[b,t]) for b in m.Buses)
model.TotalDemand = Param(model.TimePeriods, within=NonNegativeReals, initialize=calculate_total_demand)

# at this point, a user probably wants to see if they have negative demand.
def warn_about_negative_demand_rule(m, b, t):
   this_demand = value(m.Demand[b,t])
   if this_demand < 0.0:
      print("***WARNING: The demand at bus="+str(b)+" for time period="+str(t)+" is negative - value="+str(this_demand)+"; model="+str(m.name)+".")

model.WarnAboutNegativeDemand = BuildAction(model.Buses, model.TimePeriods, rule=warn_about_negative_demand_rule)

##################################################################
# the global system reserve, for each time period. units are MW. #
# NOTE: We don't have per-bus / zonal reserve requirements. they #
#       would be easy to add. (dlw oct 2013: this comment is incorrect, I think)                                   #
##################################################################

# we provide two mechanisms to specify reserve requirements. the
# first is a scaling factor relative to demand, on a per time 
# period basis. the second is an explicit parameter that specifies
# the reserver requirement on a per-time-period basis. if the 
# reserve requirement factor is > 0, then it is used to populate
# the reserve requirements. otherwise, the user-supplied reserve
# requirements are used.

model.ReserveFactor = Param(within=Reals, default=-1.0, mutable=True)

model.ReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True)

def populate_reserve_requirements_rule(m):
   reserve_factor = value(m.ReserveFactor)
   if reserve_factor > 0.0:
      for t in m.TimePeriods:
         demand = sum(value(m.Demand[b,t]) for b in m.Buses)
         m.ReserveRequirement[t] = reserve_factor * demand

model.PopulateReserveRequirements = BuildAction(rule=populate_reserve_requirements_rule)

##############################################################
# failure probability for each generator, in any given hour. #
# not used within the model itself at present, but rather    #
# used by scripts that read / manipulate the model.          #
##############################################################

def probability_failure_validator(m, v, g):
   return v >= 0.0 and v <= 1.0

model.FailureProbability = Param(model.ThermalGenerators, validate=probability_failure_validator, default=0.0)

#####################################################################################
# a binary indicator as to whether or not each generator is on-line during a given  #
# time period. intended to represent a sampled realization of the generator failure #
# probability distributions. strictly speaking, we interpret this parameter value   #
# as indicating whether or not the generator is contributing (injecting) power to   #
# the PowerBalance constraint. this parameter is not intended to be used in the     #
# context of ramping or time up/down constraints.                                   # 
#####################################################################################

model.GeneratorForcedOutage = Param(model.ThermalGenerators * model.TimePeriods, within=Binary, default=False)

####################################################################################
# minimum and maximum generation levels, for each thermal generator. units are MW. #
# could easily be specified on a per-time period basis, but are not currently.     #
####################################################################################

# you can enter generator limits either once for the generator or for each period (or just take 0)

model.MinimumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)

def maximum_power_output_validator(m, v, g):
   return v >= value(m.MinimumPowerOutput[g])

model.MaximumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, validate=maximum_power_output_validator, default=0.0)

# wind is similar, but max and min will be equal for non-dispatchable wind

model.MinNondispatchablePower = Param(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True)

def maximum_nd_output_validator(m, v, g, t):
   return v >= value(m.MinNondispatchablePower[g,t])

model.MaxNondispatchablePower = Param(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True, validate=maximum_nd_output_validator)

#################################################
# generator ramp up/down rates. units are MW/h. #
# IMPORTANT: Generator ramp limits can exceed   #
# the maximum power output, because it is the   #
# ramp limit over an hour. If the unit can      #
# fully ramp in less than an hour, then this    #
# will occur.                                   #
#################################################

# limits for normal time periods
model.NominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, mutable=True)
model.NominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, mutable=True)

# limits for time periods in which generators are brought on or off-line.
# must be no less than the generator minimum output.
# We're ignoring this validator for right now and enforcing meaning when scaling
def ramp_limit_validator(m, v, g):
   return True
   #return v >= m.MinimumPowerOutput[g] and v <= m.MaximumPowerOutput[g]

model.StartupRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, mutable=True)
model.ShutdownRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, mutable=True)

def scale_ramp_up(m, g):
    temp = value(m.NominalRampUpLimit[g]) * m.TimePeriodLength
    if temp > m.MaximumPowerOutput[g]:
        return m.MaximumPowerOutput[g]
    else:
        return temp
model.ScaledNominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=scale_ramp_up, mutable=True)

def scale_ramp_down(m, g):
    temp = value(m.NominalRampDownLimit[g]) * m.TimePeriodLength
    if temp > m.MaximumPowerOutput[g]:
        return m.MaximumPowerOutput[g]
    else:
        return temp
model.ScaledNominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=scale_ramp_down, mutable=True)

def scale_startup_limit(m, g):
    temp = value(m.StartupRampLimit[g]) #* m.TimePeriodLength
    if temp > m.MaximumPowerOutput[g]:
        return m.MaximumPowerOutput[g]
    else:
        return temp
model.ScaledStartupRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, initialize=scale_startup_limit, mutable=True)

def scale_shutdown_limit(m, g):
    temp = value(m.ShutdownRampLimit[g]) #* m.TimePeriodLength
    if temp > m.MaximumPowerOutput[g]:
        return m.MaximumPowerOutput[g]
    else:
        return temp
model.ScaledShutdownRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, initialize=scale_shutdown_limit, mutable=True)


##########################################################################################################
# the minimum number of time periods that a generator must be on-line (off-line) once brought up (down). #
##########################################################################################################

model.MinimumUpTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, default=0)
model.MinimumDownTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, default=0)

def scale_min_uptime(m, g):
    return int(round(m.MinimumUpTime[g] / m.TimePeriodLength))
model.ScaledMinimumUpTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=scale_min_uptime)

def scale_min_downtime(m, g):
    return int(round(m.MinimumDownTime[g] / m.TimePeriodLength))
model.ScaledMinimumDownTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=scale_min_downtime)

#############################################
# unit on state at t=0 (initial condition). #
#############################################

# if positive, the number of hours prior to (and including) t=0 that the unit has been on.
# if negative, the number of hours prior to (and including) t=0 that the unit has been off.
# the value cannot be 0, by definition.

def t0_state_nonzero_validator(m, v, g):
    return v != 0

model.UnitOnT0State = Param(model.ThermalGenerators, within=Reals, validate=t0_state_nonzero_validator, mutable=True)

def t0_unit_on_rule(m, g):
    return int(value(m.UnitOnT0State[g]) >= 1)

model.UnitOnT0 = Param(model.ThermalGenerators, within=Binary, initialize=t0_unit_on_rule, mutable=True)

#######################################################################################
# the number of time periods that a generator must initally on-line (off-line) due to #
# its minimum up time (down time) constraint.                                         #
#######################################################################################

def initial_time_periods_online_rule(m, g):
   if not value(m.UnitOnT0[g]):
      return 0
   else:
      return int(min(value(m.NumTimePeriods),
             round(max(0, value(m.MinimumUpTime[g]) - value(m.UnitOnT0State[g])) / value(m.TimePeriodLength))))

model.InitialTimePeriodsOnLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_online_rule, mutable=True)

def initial_time_periods_offline_rule(m, g):
   if value(m.UnitOnT0[g]):
      return 0
   else:
      return int(min(value(m.NumTimePeriods),
             round(max(0, value(m.MinimumDownTime[g]) + value(m.UnitOnT0State[g])) / value(m.TimePeriodLength)))) # m.UnitOnT0State is negative if unit is off

model.InitialTimePeriodsOffLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_offline_rule, mutable=True)

####################################################################
# generator power output at t=0 (initial condition). units are MW. #
####################################################################

def between_limits_validator(m, v, g):
   status = (v <= (value(m.MaximumPowerOutput[g]) * value(m.UnitOnT0[g]))  and v >= (value(m.MinimumPowerOutput[g]) * value(m.UnitOnT0[g])))
   if status == False:
      print("Failed to validate PowerGeneratedT0 value for g="+g+"; new value="+str(v)+", UnitOnT0="+str(value(m.UnitOnT0[g])))
   return v <= (value(m.MaximumPowerOutput[g]) * value(m.UnitOnT0[g]))  and v >= (value(m.MinimumPowerOutput[g]) * value(m.UnitOnT0[g]))
model.PowerGeneratedT0 = Param(model.ThermalGenerators, within=NonNegativeReals, validate=between_limits_validator, mutable=True)

##################################################################################################################
# production cost coefficients (for the quadratic) a0=constant, a1=linear coefficient, a2=quadratic coefficient. #
##################################################################################################################

model.ProductionCostA0 = Param(model.ThermalGenerators, default=0.0) # units are $/hr (or whatever the time unit is).
model.ProductionCostA1 = Param(model.ThermalGenerators, default=0.0) # units are $/MWhr.
model.ProductionCostA2 = Param(model.ThermalGenerators, default=0.0) # units are $/(MWhr^2).

# the parameters below are populated if cost curves are specified as linearized heat rate increment segments.
#
# CostPiecewisePoints represents the power output levels defining the segment boundaries.
# these *must* include the minimum and maximum power output points - a validation check
# if performed below.
# 
# CostPiecewiseValues are the absolute heat rates / costs associated with the corresponding 
# power output levels. the precise interpretation of whether a value is a heat rate or a cost
# depends on the value of the FuelCost parameter, specified below.

# there are many ways to interpret the cost piecewise point/value data, when translating into
# an actual piecewise construct for the model. this interpretation is controlled by the following
# string parameter, whose legal values are: NoPiecewise (no data provided), Absolute, Incremental, 
# and SubSegmentation. NoPiecewise means that we're using quadraic cost curves, and will 
# construct the piecewise data ourselves directly from that cost curve. 

def piecewise_type_validator(m, v):
   return (v == "NoPiecewise") or (v == "Absolute") or (v == "Incremental") or (v != "SubSegementation")

def piecewise_type_init(m):
    boo = False
    for g in m.ThermalGenerators:
        if not (m.ProductionCostA0[g] == 0.0 and m.ProductionCostA1[g] == 0.0 and m.ProductionCostA2[g] == 0.0):
            boo = True
    if boo:
        return "NoPiecewise"
    else:
        return "Absolute"

model.PiecewiseType = Param(validate=piecewise_type_validator,initialize=piecewise_type_init, mutable=True)  #irios: default="Absolute" initialize=piecewise_type_init

def piecewise_init(m, g):
    return []

model.CostPiecewisePoints = Set(model.ThermalGenerators, initialize=piecewise_init, ordered=True, within=NonNegativeReals)
model.CostPiecewiseValues = Set(model.ThermalGenerators, initialize=piecewise_init, ordered=True, within=NonNegativeReals)

# a check to ensure that the cost piecewise point parameter was correctly populated.
# these are global checks, which cannot be performed as a set validation (which 
# operates on a single element at a time).

# irios: When the check fails, I add the missing PiecewisePoints and Values in order to make it work.
# I did this in an arbitrary way, but you can change it. In particular, I erased those values which are not 
# between the minimum power output and the maximum power output. Also, I added those values if they are not in
# the input data. Finally, I added values (0 and this_generator_piecewise_values[-1] + 1) to end with the same 
# number of points and values.

def validate_cost_piecewise_points_and_values_rule(m, g):

    # IGNACIO: LOOK HERE - CAN YOU MERGE IN THE FOLLOWING INTO THIS RULE?
#    if m.CostPiecewisePoints[g][1] > m.CorrectCostPiecewisePoints[g][1]:
#        this_generator_piecewise_values.insert(0,0)
#    if m.CostPiecewisePoints[g][len(m.CostPiecewisePoints[g])] < m.CorrectCostPiecewisePoints[g][len(m.CorrectCostPiecewisePoints[g])]:
#        this_generator_piecewise_values.append(this_generator_piecewise_values[-1] + 1)
    
    if value(m.PiecewiseType) == "NoPiecewise":
        # if there isn't any piecewise data specified, we shouldn't find any.
        if len(m.CostPiecewisePoints[g]) > 0:
            print("DATA ERROR: The PiecewiseType parameter was set to NoPiecewise, but piecewise point data was specified!")
            return False
        # if there isn't anything to validate and we didn't expect piecewise 
        # points, we can safely skip the remaining validation steps.
        return True
    else:
        # if the user said there was going to be piecewise data and none was 
        # supplied, they should be notified as to such.
        if len(m.CostPiecewisePoints[g]) == 0:
            print("DATA ERROR: The PiecewiseType parameter was set to something other than NoPiecewise, but no piecewise point data was specified!")
            return False

   # per the requirement below, there have to be at least two piecewise points if there are any.

    min_output = value(m.MinimumPowerOutput[g])
    max_output = value(m.MaximumPowerOutput[g])   

    new_points = sorted(list(m.CostPiecewisePoints[g]))
    new_values = sorted(list(m.CostPiecewiseValues[g]))

    if min_output not in new_points:
        print("DATA WARNING: Cost piecewise points for generator g="+str(g)+" must contain the minimum output level="+str(min_output)+" - so we added it.")
        new_points.insert(0, min_output)

    if max_output not in new_points:
        print("DATA WARNING: Cost piecewise points for generator g="+str(g)+" must contain the maximum output level="+str(max_output)+" - so we added it.")
        new_points.append(max_output)

    # We delete those values which are not in the interval [min_output, max_output]
    new_points = [new_points[i] for i in range(len(new_points)) if (min_output <= new_points[i] and new_points[i] <= max_output)]

    # We have to make sure that we have the same number of Points and Values
    if len(new_points) < len(new_values): # if the number of points is less than the number of values, we take the first len(new_points) elements of new_values
        new_values = [new_values[i] for i in range(len(new_points))]
    if len(new_points) > len(new_values): # if the number of values is lower, then we add values at the end of new_values increasing by 1 each time.
        i = 1
        while len(new_points) != len(new_values):
            new_values.append(new_values[-1] + i)
            i += 1 

    if list(m.CostPiecewisePoints[g]) != new_points:
        m.CostPiecewisePoints[g].clear()
        #m.CostPiecewisePoints[g].add(*new_points) # dlw and Julia July 2014 - changed to below.
        for pcwpoint in new_points:
            m.CostPiecewisePoints[g].add(pcwpoint) 

    if list(m.CostPiecewiseValues[g]) != new_values:
        m.CostPiecewiseValues[g].clear()
        # m.CostPiecewiseValues[g].add(*new_values) # dlw and Julia July 2014 - changed to below.
        for pcwvalue in new_values:
            m.CostPiecewiseValues[g].add(pcwvalue)
        
    return True

model.ValidateCostPiecewisePointsAndValues = BuildCheck(model.ThermalGenerators, rule=validate_cost_piecewise_points_and_values_rule)

# Sets the cost of fuel to the generator.  Defaults to 1 so that we could just input cost as heat rates.
model.FuelCost = Param(model.ThermalGenerators, default=1.0) 

# Minimum production cost (needed because Piecewise constraint on ProductionCost 
# has to have lower bound of 0, so the unit can cost 0 when off -- this is added
# back in to the objective if a unit is on
def minimum_production_cost(m, g):
    if len(m.CostPiecewisePoints[g]) > 1:
        return m.CostPiecewiseValues[g].first() * m.FuelCost[g]
    else:
        return  m.FuelCost[g] * \
               (m.ProductionCostA0[g] + \
                m.ProductionCostA1[g] * m.MinimumPowerOutput[g] + \
                m.ProductionCostA2[g] * (m.MinimumPowerOutput[g]**2))
model.MinimumProductionCost = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=minimum_production_cost, mutable=True)

##############################################################################################
# number of pieces in the linearization of each generator's quadratic cost production curve. #
##############################################################################################

model.NumGeneratorCostCurvePieces = Param(within=PositiveIntegers, default=2, mutable=True)

#######################################################################
# points for piecewise linearization of power generation cost curves. #
#######################################################################

# maps a (generator, time-index) pair to a list of points defining the piecewise cost linearization breakpoints.
# the time index is redundant, but required - in the current implementation of the Piecewise construct, the 
# breakpoints must be indexed the same as the Piecewise construct itself.

# the points are expected to be on the interval [0, maxpower], and must contain both endpoints. 
# power generated can always be 0, and piecewise expects the entire variable domain to be represented.
model.PowerGenerationPiecewisePoints = {}

# NOTE: the values are relative to the minimum production cost, i.e., the values represent
# incremental costs relative to the minimum production cost.

# IMPORTANT: These values are *not* scaled by the FuelCost of the generator. This scaling is
#            performed subsequently, in the "production_cost_function" code, which is used
#            by Piecewise to compute the production cost of the generator. all values must
#            be non-negative.
model.PowerGenerationPiecewiseValues = {}

def power_generation_piecewise_points_rule(m, g, t):

    # factor out the fuel cost here, as the piecewise approximation is scaled by fuel cost
    # elsewhere in the model (i.e., in the Piecewise construct below).
    minimum_production_cost = value(m.MinimumProductionCost[g]) / value(m.FuelCost[g])
    
    piecewise_type = value(m.PiecewiseType)

    if piecewise_type == "Absolute":

       m.PowerGenerationPiecewisePoints[g,t] = list(m.CostPiecewisePoints[g])
       piecewise_values = list(m.CostPiecewiseValues[g])
       m.PowerGenerationPiecewiseValues[g,t] = {}
       for i in xrange(len(m.CostPiecewisePoints[g])):
          this_point = m.PowerGenerationPiecewisePoints[g,t][i]
          m.PowerGenerationPiecewiseValues[g,t][this_point] = piecewise_values[i] - minimum_production_cost

       # ensure the (0.0, 0.0) point is present.
       if m.PowerGenerationPiecewisePoints[g,t][0] != 0.0:
          m.PowerGenerationPiecewisePoints[g,t].insert(0, 0.0)
       m.PowerGenerationPiecewiseValues[g,t][0.0] = 0.0

    elif piecewise_type == "Incremental":
       # NOTE: THIS DOESN'T WORK!!!
       if len(m.CostPiecewisePoints[g]) > 0:
          PowerBreakpoints = list(m.CostPiecewisePoints[g])
          PowerBreakpoints.insert(0, 0)
          IncrementalCost = list(m.CostPiecewiseValues[g])
          CostBreakpoints = {}
          CostBreakpoints[0] = 0
          for i in xrange(1, len(IncrementalCost) + 1):
             CostBreakpoints[PowerBreakpoints[i]] = CostBreakpoints[PowerBreakpoints[i-1]] + \
                 (PowerBreakpoints[i] - PowerBreakpoints[i-1])* IncrementalCost[i-1]
          m.PowerGenerationPiecewisePoints[g,t] = list(PowerBreakpoints)
          m.PowerGenerationPiecewiseValues[g,t] = dict(CostBreakpoints)
          #        print g, t, m.PowerGenerationPiecewisePoints[g, t]
          #        print g, t, m.PowerGenerationPiecewiseValues[g, t]

       else:
          print("***BADGenerators must have at least 1 point in their incremental cost curve")
          assert(False)

    elif piecewise_type == "SubSegmentation":

       print("***BAD - we don't have logic for generating piecewise points of type SubSegmentation!!!")
       assert(False)

    else: # piecewise_type == "NoPiecewise"

       if value(m.ProductionCostA2[g]) == 0:
          # If cost is linear, we only need two points -- (MinOutput,0) and (MaxOutput, MaxCost-MinCost))
          min_power = value(m.MinimumPowerOutput[g])
          max_power = value(m.MaximumPowerOutput[g])
          if min_power == max_power:
             m.PowerGenerationPiecewisePoints[g, t] = [0.0, min_power]
          else:
             m.PowerGenerationPiecewisePoints[g, t] = [0.0, min_power, max_power]

          m.PowerGenerationPiecewiseValues[g,t] = {}

          m.PowerGenerationPiecewiseValues[g,t][0.0] = 0.0
          m.PowerGenerationPiecewiseValues[g,t][min_power] = 0.0

          if min_power != max_power:
             m.PowerGenerationPiecewiseValues[g,t][max_power] = \
                 value(m.ProductionCostA0[g]) + \
                 value(m.ProductionCostA1[g]) * m.PowerGenerationPiecewisePoints[g, t][-1] \
                 - minimum_production_cost

       else:
           min_power = value(m.MinimumPowerOutput[g])
           max_power = value(m.MaximumPowerOutput[g])
           n = value(m.NumGeneratorCostCurvePieces)
           width = (max_power - min_power) / float(n)
           if width == 0:
               m.PowerGenerationPiecewisePoints[g, t] = [0, min_power]
           else:
               m.PowerGenerationPiecewisePoints[g, t] = []
               m.PowerGenerationPiecewisePoints[g, t].extend([min_power + i*width for i in xrange(0,n+1)])
               # NOTE: due to numerical precision limitations, the last point in the x-domain
               #       of the generation piecewise cost curve may not be precisely equal to the 
               #       maximum power output level of the generator. this can cause Piecewise to
               #       sqawk, as it would like the upper bound of the variable to be represented
               #       in the domain. so, we will make it so.
               m.PowerGenerationPiecewisePoints[g, t][-1] = max_power
           m.PowerGenerationPiecewiseValues[g,t] = {}
           for i in xrange(len(m.PowerGenerationPiecewisePoints[g, t])):
               m.PowerGenerationPiecewiseValues[g,t][m.PowerGenerationPiecewisePoints[g,t][i]] = \
                          value(m.ProductionCostA0[g]) + \
                          value(m.ProductionCostA1[g]) * m.PowerGenerationPiecewisePoints[g, t][i] + \
                          value(m.ProductionCostA2[g]) * m.PowerGenerationPiecewisePoints[g, t][i]**2 \
                          - minimum_production_cost
           if m.PowerGenerationPiecewisePoints[g, t][0] != 0:
               m.PowerGenerationPiecewisePoints[g, t].insert(0,0)
               m.PowerGenerationPiecewiseValues[g, t][0] = 0
    
    # validate the computed points, independent of the method used to generate them.
    # nothing should be negative, and the costs should be monotonically non-decreasing.
    for i in xrange(0, len(m.PowerGenerationPiecewisePoints[g, t])):
       this_level = m.PowerGenerationPiecewisePoints[g, t][i]
       assert this_level >= 0.0

model.CreatePowerGenerationPiecewisePoints = BuildAction(model.ThermalGenerators * model.TimePeriods, rule=power_generation_piecewise_points_rule)

###############################################
# startup cost parameters for each generator. #
###############################################

# startup costs are conceptually expressed as pairs (x, y), where x represents the number of hours that a unit has been off and y represents
# the cost associated with starting up the unit after being off for x hours. these are broken into two distinct ordered sets, as follows.

def startup_lags_init_rule(m, g):
   return [value(m.MinimumDownTime[g])] 
model.StartupLags = Set(model.ThermalGenerators, within=NonNegativeIntegers, ordered=True, initialize=startup_lags_init_rule) # units are hours / time periods.

def startup_costs_init_rule(m, g):
   return [0.0] 

model.StartupCosts = Set(model.ThermalGenerators, within=NonNegativeReals, ordered=True, initialize=startup_costs_init_rule) # units are $.

# startup lags must be monotonically increasing...
def validate_startup_lags_rule(m, g):
   startup_lags = list(m.StartupLags[g])

   if len(startup_lags) == 0:
      print("DATA ERROR: The number of startup lags for thermal generator="+str(g)+" must be >= 1.")
      assert(False)

   if startup_lags[0] != value(m.MinimumDownTime[g]):
      print("DATA ERROR: The first startup lag for thermal generator="+str(g)+" must be equal the minimum down time="+str(value(m.MinimumDownTime[g]))+".")
      assert(False)      

   for i in xrange(0, len(startup_lags)-1):
      if startup_lags[i] >= startup_lags[i+1]:
         print("DATA ERROR: Startup lags for thermal generator="+str(g)+" must be monotonically increasing.")
         assert(False)

model.ValidateStartupLags = BuildAction(model.ThermalGenerators, rule=validate_startup_lags_rule)

# while startup costs must be monotonically non-decreasing!
def validate_startup_costs_rule(m, g):
   startup_costs = list(m.StartupCosts[g])
   for i in xrange(0, len(startup_costs)-2):
      if startup_costs[i] > startup_costs[i+1]:
         print("DATA ERROR: Startup costs for thermal generator="+str(g)+" must be monotonically non-decreasing.")
         assert(False)

model.ValidateStartupCosts = BuildAction(model.ThermalGenerators, rule=validate_startup_costs_rule)

def validate_startup_lag_cost_cardinalities(m, g):
   if len(m.StartupLags[g]) != len(m.StartupCosts[g]):
      print("DATA ERROR: The number of startup lag entries ("+str(len(m.StartupLags[g]))+") for thermal generator="+str(g)+" must equal the number of startup cost entries ("+str(len(m.StartupCosts[g]))+")")
      assert(False)

model.ValidateStartupLagCostCardinalities = BuildAction(model.ThermalGenerators, rule=validate_startup_lag_cost_cardinalities)

# for purposes of defining constraints, it is useful to have a set to index the various startup costs parameters.
# entries are 1-based indices, because they are used as indicies into Pyomo sets - which use 1-based indexing.

def startup_cost_indices_init_rule(m, g):
   return xrange(1, len(m.StartupLags[g])+1)

model.StartupCostIndices = Set(model.ThermalGenerators, within=NonNegativeIntegers, initialize=startup_cost_indices_init_rule)

##################################################################################
# shutdown cost for each generator. in the literature, these are often set to 0. #
##################################################################################

model.ShutdownFixedCost = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0) # units are $.

#################################
# Regulation ancillary services #
#################################

if regulation_services:

    # Regulation
    model.RegulationProvider = Param(model.ThermalGenerators, within=Binary, default=0) #indicates if a unit is offering regulation

    # When units are selected for regulation, their limits are bounded by the RegulationHighLimit and RegulationLowLimit
    # I'll refer to it as the "regulation band"
    def regulation_high_limit_validator(m, v, g):
        return v <= value(m.MaximumPowerOutput[g])
    def regulation_high_limit_init(m, g):
        return value(m.MaximumPowerOutput[g])   
    model.RegulationHighLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=regulation_high_limit_validator, initialize=regulation_high_limit_init)

    def calculate_max_power_minus_reg_high_limit_rule(m, g):
        return m.MaximumPowerOutput[g] - m.RegulationHighLimit[g]
    model.MaxPowerOutputMinusRegHighLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=calculate_max_power_minus_reg_high_limit_rule)

    def regulation_low_limit_validator(m, v, g):
        return (v <= value(m.RegulationHighLimit[g]) and v >= value(m.MinimumPowerOutput[g]))
    def regulation_low_limit_init(m, g):
        return value(m.MinimumPowerOutput[g])
    model.RegulationLowLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=regulation_low_limit_validator, initialize=regulation_low_limit_init)

    # Regulation capacity is calculated as the max of "regulation band" and 5*AutomaticResponseRate
    model.AutomaticResponseRate = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)

    def calculate_regulation_capability_rule(m, g):
        temp1 = 5 * m.AutomaticResponseRate[g]
        temp2 = (m.RegulationHighLimit[g] - m.RegulationLowLimit[g])/2
        if temp1 > temp2:
            return temp2
        else:
            return temp1

    model.RegulationCapability = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=calculate_regulation_capability_rule, default=0.0)
    model.ZonalRegulationRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
    model.GlobalRegulationRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)

    # regulation cost
    model.RegulationOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)

if reserve_services:
    # At ISO-NE the ancillary services are not "co-optimize" in the day-ahead.
    # However, this formulation handles ancillary service offers which are common in other markets.
    #
    # spinning reserve
    model.SpinningReserveTime = Param(within=NonNegativeReals, default=0.16666667) # in hours, varies among ISOs
    model.ZonalSpinningReserveRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
    model.SystemSpinningReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)
    model.SpinningReserveOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)

    # non-spinning reserve
    model.NonSpinningReserveAvailable = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)  #ISO-NE's Claim10 parameter
    model.NonSpinningReserveOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    model.ZonalTenMinuteReserveRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
    model.SystemTenMinuteReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)

    # Thirty-minute operating reserve
    model.OperatingReserveAvailable = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)  #ISO-NE's Claim30 parameter
    model.OperatingReserveOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    model.ZonalOperatingReserveRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
    model.SystemOperatingReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)

#
# STORAGE parameters
#

if storage_services:

    model.Storage = Set()
    model.StorageAtBus = Set(model.Buses, initialize=Set())

    ####################################################################################
    # minimum and maximum power ratings, for each storage unit. units are MW.          #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################

    # Storage power output >0 when discharging

    model.MinimumPowerOutputStorage = Param(model.Storage, within=NonNegativeReals, default=0.0)

    def maximum_power_output_validator_storage(m, v, s):
       return v >= value(m.MinimumPowerOutputStorage[s])

    model.MaximumPowerOutputStorage = Param(model.Storage, within=NonNegativeReals, validate=maximum_power_output_validator_storage, default=0.0)

    #Storage power input >0 when charging

    model.MinimumPowerInputStorage = Param(model.Storage, within=NonNegativeReals, default=0.0)

    def maximum_power_input_validator_storage(m, v, s):
       return v >= value(m.MinimumPowerInputStorage[s])

    model.MaximumPowerInputStorage = Param(model.Storage, within=NonNegativeReals, validate=maximum_power_input_validator_storage, default=0.0)

    ###############################################
    # storage ramp up/down rates. units are MW/h. #
    ###############################################

    # ramp rate limits when discharging
    model.NominalRampUpLimitStorageOutput    = Param(model.Storage, within=NonNegativeReals)
    model.NominalRampDownLimitStorageOutput  = Param(model.Storage, within=NonNegativeReals)

    # ramp rate limits when charging
    model.NominalRampUpLimitStorageInput     = Param(model.Storage, within=NonNegativeReals)
    model.NominalRampDownLimitStorageInput   = Param(model.Storage, within=NonNegativeReals)

    def scale_storage_ramp_up_out(m, s):
        return m.NominalRampUpLimitStorageOutput[s] * m.TimePeriodLength
    model.ScaledNominalRampUpLimitStorageOutput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_up_out)

    def scale_storage_ramp_down_out(m, s):
        return m.NominalRampDownLimitStorageOutput[s] * m.TimePeriodLength
    model.ScaledNominalRampDownLimitStorageOutput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_down_out)

    def scale_storage_ramp_up_in(m, s):
        return m.NominalRampUpLimitStorageInput[s] * m.TimePeriodLength
    model.ScaledNominalRampUpLimitStorageInput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_up_in)

    def scale_storage_ramp_down_in(m, s):
        return m.NominalRampDownLimitStorageInput[s] * m.TimePeriodLength
    model.ScaledNominalRampDownLimitStorageInput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_down_in)

    ####################################################################################
    # minimum state of charge (SOC) and maximum energy ratings, for each storage unit. #
    # units are MWh for energy rating and p.u. (i.e. [0,1]) for SOC     #
    ####################################################################################

    # you enter storage energy ratings once for each storage unit

    model.MaximumEnergyStorage = Param(model.Storage, within=NonNegativeReals, default=0.0)
    model.MinimumSocStorage = Param(model.Storage, within=PercentFraction, default=0.0)

    ################################################################################
    # round trip efficiency for each storage unit given as a fraction (i.e. [0,1]) #
    ################################################################################

    model.EfficiencyEnergyStorage = Param(model.Storage, within=PercentFraction, default=1.0)

    ########################################################################
    # end-point SOC for each storage unit. units are in p.u. (i.e. [0,1])  #
    ########################################################################

    # end-point values are the SOC targets at the final time period. With no end-point constraints
    # storage units will always be empty at the final time period.

    model.EndPointSocStorage = Param(model.Storage, within=PercentFraction, default=0.5)

    ############################################################
    # storage initial conditions: SOC, power output and input  #
    ############################################################

    def t0_storage_power_input_validator(m, v, s):
        return (v >= value(m.MinimumPowerInputStorage[s])) and (v <= value(m.MaximumPowerInputStorage[s]))

    def t0_storage_power_output_validator(m, v, s):
        return (v >= value(m.MinimumPowerInputStorage[s])) and (v <= value(m.MaximumPowerInputStorage[s]))

    model.StoragePowerOutputOnT0 = Param(model.Storage, within=NonNegativeIntegers, validate=t0_storage_power_output_validator, default=0)
    model.StoragePowerInputOnT0  = Param(model.Storage, within=NonNegativeIntegers, validate=t0_storage_power_input_validator, default=0)
    model.StorageSocOnT0         = Param(model.Storage, within=PercentFraction, default=0.5)

#########################################
# penalty costs for constraint violation #
#########################################

BigPenalty = 1e6
ModeratelyBigPenalty = 1e5

model.LoadMismatchPenalty = Param(within=NonNegativeReals, default=BigPenalty)
model.ReserveMismatchPenalty = Param(within=NonNegativeReals, default=ModeratelyBigPenalty)

#
# Variables
#

######################
# decision variables #
######################

# indicator variables for each generator, at each time period.
model.UnitOn = Var(model.ThermalGenerators, model.TimePeriods, within=Binary) 

# amount of power produced by each generator, at each time period.
def power_bounds_rule(m, g, t):
    return (0, m.MaximumPowerOutput[g])
model.PowerGenerated = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, bounds=power_bounds_rule) 

# amount of power flowing along each line, at each time period
def line_power_bounds_rule(m, l, t):
   return (-m.ThermalLimit[l], m.ThermalLimit[l])
model.LinePower = Var(model.TransmissionLines, model.TimePeriods, bounds=line_power_bounds_rule)

# assume wind can be curtailed, then wind power is a decision variable
def nd_bounds_rule(m,n,t):
    return (m.MinNondispatchablePower[n,t], m.MaxNondispatchablePower[n,t])
model.NondispatchablePowerUsed = Var(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, bounds=nd_bounds_rule)

# maximum power output for each generator, at each time period.
model.MaximumPowerAvailable = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)

# voltage angles at the buses (S) (lock the first bus at 0) in radians
model.Angle = Var(model.Buses, model.TimePeriods, within=Reals, bounds=(-3.14159265,3.14159265))

def fix_first_angle_rule(m,t):
    first_bus = six.next(m.Buses.__iter__())
    return m.Angle[first_bus,t] == 0.0
model.FixFirstAngle = Constraint(model.TimePeriods, rule=fix_first_angle_rule)

###################################################
# Regulation ancillary service decision variables #
###################################################

if regulation_services:

    model.RegulationOn = Var(model.ThermalGenerators, model.TimePeriods, within=Binary)
    model.RegulationDispatched = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.TotalRegulationCapabilityAvailable = Var(model.TimePeriods, within=NonNegativeReals)
    model.RegulationCost = Var(model.TimePeriods, within=NonNegativeReals)
    model.TotalRegulationCost = Var(within=NonNegativeReals)

if reserve_services:

    # spinning reserve
    model.SpinningReserveDispatched = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.SpinningReserveCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.TotalSpinningReserveCost = Var(within=NonNegativeReals)

    # non-spinning reserve
    model.NonSpinningReserveDispatched = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.NonSpinningReserveCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.TotalNonSpinningReserveCost = Var(within=NonNegativeReals)

    # thirty-minute operating reserve
    model.OperatingReserveDispatched = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.OperatingReserveCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
    model.TotalOperatingReserveCost = Var(within=NonNegativeReals)

if storage_services:

    ##############################
    # Storage decision variables #
    ##############################

    # binary variables for storage (input/output are semicontinuous)
    model.OutputStorage = Var(model.Storage, model.TimePeriods, within=Binary)
    model.InputStorage = Var(model.Storage, model.TimePeriods, within=Binary)

    # amount of output power of each storage unit, at each time period
    def power_output_storage_bounds_rule(m, s, t):
        return (0, m.MaximumPowerOutputStorage[s])
    model.PowerOutputStorage = Var(model.Storage, model.TimePeriods, within=NonNegativeReals, bounds=power_output_storage_bounds_rule)

    # amount of input power of each storage unit, at each time period
    def power_input_storage_bounds_rule(m, s, t):
        return (0, m.MaximumPowerInputStorage[s])
    model.PowerInputStorage = Var(model.Storage, model.TimePeriods, within=NonNegativeReals, bounds=power_input_storage_bounds_rule)

    # state of charge of each storage unit, at each time period
    model.SocStorage = Var(model.Storage, model.TimePeriods, within=PercentFraction)

###################
# cost components #
###################

# production cost associated with each generator, for each time period.
model.ProductionCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)

# startup and shutdown costs for each generator, each time period.
model.StartupCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)
model.ShutdownCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)

# cost over all generators, for all time periods.
model.TotalProductionCost = Var(model.TimePeriods, within=NonNegativeReals)

# all other overhead / fixed costs, e.g., associated with startup and shutdown.
model.TotalNoLoadCost = Var(model.TimePeriods, within=NonNegativeReals)

#####################################################
# load "shedding" can be both positive and negative #
#####################################################
model.LoadGenerateMismatch = Var(model.Buses, model.TimePeriods, within=Reals)
model.posLoadGenerateMismatch = Var(model.Buses, model.TimePeriods, within=NonNegativeReals) # load shedding
model.negLoadGenerateMismatch = Var(model.Buses, model.TimePeriods, within=NonNegativeReals) # over generation

# JPW 7/28: I don't think we technically need to have a negative mismatch here, as we
#           are talking about headroom only - which is not subjected to physical ramping
#           constraints. If you don't need it, you don't need it!
model.ReserveMismatch = Var(model.TimePeriods, within=Reals)
model.posReserveMismatch = Var(model.TimePeriods, within=NonNegativeReals) 
model.negReserveMismatch = Var(model.TimePeriods, within=NonNegativeReals)

# the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
# the appearance of load generation mismatch component values in the range of *negative* e-5.
# what these small negative values do is to cause the optimal objective to be a very large negative,
# due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
# at this point, but it does seem to work broadly. we tried a single global constraint, across all
# buses, but that failed to correct the problem, and caused the solve times to explode.

def pos_load_generate_mismatch_tolerance_rule(m, b):
   return sum((m.posLoadGenerateMismatch[b,t] for t in m.TimePeriods)) >= 0.0
model.PosLoadGenerateMismatchTolerance = Constraint(model.Buses, rule=pos_load_generate_mismatch_tolerance_rule)

def neg_load_generate_mismatch_tolerance_rule(m, b):
   return sum((m.negLoadGenerateMismatch[b,t] for t in m.TimePeriods)) >= 0.0
model.NegLoadGenerateMismatchTolerance = Constraint(model.Buses, rule=neg_load_generate_mismatch_tolerance_rule)

#
# Constraints
#

def line_power_rule(m, l, t):
   return m.LinePower[l,t] == m.B[l] * (m.Angle[m.BusFrom[l], t] - m.Angle[m.BusTo[l], t])
model.CalculateLinePower = Constraint(model.TransmissionLines, model.TimePeriods, rule=line_power_rule)

# Power balance at each node (S)
def power_balance(m, b, t):
    # bus b, time t (S)
    if storage_services:
        return sum((1 - m.GeneratorForcedOutage[g,t]) * m.PowerGenerated[g, t] for g in m.ThermalGeneratorsAtBus[b]) \
               + sum(m.PowerOutputStorage[s, t] for s in m.StorageAtBus[b])\
               - sum(m.PowerInputStorage[s, t] for s in m.StorageAtBus[b])\
               + sum(m.NondispatchablePowerUsed[g, t] for g in m.NondispatchableGeneratorsAtBus[b]) \
               + sum(m.LinePower[l,t] for l in m.LinesTo[b]) \
               - sum(m.LinePower[l,t] for l in m.LinesFrom[b]) \
               + m.LoadGenerateMismatch[b,t] \
               == m.Demand[b, t] 
    else:
        return sum((1 - m.GeneratorForcedOutage[g,t]) * m.PowerGenerated[g, t] for g in m.ThermalGeneratorsAtBus[b]) \
               + sum(m.NondispatchablePowerUsed[g, t] for g in m.NondispatchableGeneratorsAtBus[b]) \
               + sum(m.LinePower[l,t] for l in m.LinesTo[b]) \
               - sum(m.LinePower[l,t] for l in m.LinesFrom[b]) \
               + m.LoadGenerateMismatch[b,t] \
               == m.Demand[b, t] 
        
model.PowerBalance = Constraint(model.Buses, model.TimePeriods, rule=power_balance)

# give meaning to the positive and negative parts of the mismatch
def posneg_rule(m, b, t):
    return m.posLoadGenerateMismatch[b, t] - m.negLoadGenerateMismatch[b, t] == m.LoadGenerateMismatch[b, t]
model.Defineposneg_Mismatch = Constraint(model.Buses, model.TimePeriods, rule = posneg_rule)

def global_posneg_rule(m, t):
    return m.posReserveMismatch[t] - m.negReserveMismatch[t] == m.ReserveMismatch[t]
model.Global_Defineposneg_Mismatch = Constraint(model.TimePeriods, rule = global_posneg_rule)

# ensure there is sufficient maximal power output available to meet both the
# demand and the spinning reserve requirements in each time period.
# encodes Constraint 3 in Carrion and Arroyo.

def enforce_reserve_requirements_rule(m, t):
    if storage_services:
        return sum(m.MaximumPowerAvailable[g, t] for g in m.ThermalGenerators) \
               + sum(m.NondispatchablePowerUsed[n,t] for n in m.AllNondispatchableGenerators) \
               + sum(m.PowerOutputStorage[s,t] for s in m.Storage) \
                  == \
               m.TotalDemand[t] + m.ReserveRequirement[t] + m.ReserveMismatch[t]
    else:
        return sum(m.MaximumPowerAvailable[g, t] for g in m.ThermalGenerators) \
               + sum(m.NondispatchablePowerUsed[n,t] for n in m.AllNondispatchableGenerators) \
                  == \
               m.TotalDemand[t] + m.ReserveRequirement[t] + m.ReserveMismatch[t]        

model.EnforceReserveRequirements = Constraint(model.TimePeriods, rule=enforce_reserve_requirements_rule)

if regulation_services:

    # CASM: zonal reserve requirement - ensure there is enough "regulation" reserve
    # in each reserve zone and each time period - This is not an accurate representation or reg up reserves.
    # It will be refined after verification with Alstom. It's just to see if the zonal reserve requirement
    # works.

    model.RegulatingReserveUpAvailable = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals)

    def calculate_regulating_reserve_up_available_per_generator(m, g, t):
        return m.RegulatingReserveUpAvailable[g, t] == m.MaximumPowerAvailable[g,t] - m.PowerGenerated[g,t]
    model.CalculateRegulatingReserveUpPerGenerator = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_regulating_reserve_up_available_per_generator)

    def enforce_zonal_reserve_requirement_rule(m, rz, t):
        return sum(m.RegulatingReserveUpAvailable[g,t] for g in m.ThermalGeneratorsInReserveZone[rz]) >= m.ZonalReserveRequirement[rz, t]

    model.EnforceZonalReserveRequirements = Constraint(model.ReserveZones, model.TimePeriods, rule=enforce_zonal_reserve_requirement_rule)

############################################
# generation limit and ramping constraints #
############################################

# enforce the generator power output limits on a per-period basis.
# the maximum power available at any given time period is dynamic,
# bounded from above by the maximum generator output.

# the following three constraints encode Constraints 16 and 17 defined in Carrion and Arroyo.

# NOTE: The expression below is what we really want - however, due to a pyomo design feature, we have to split it into two constraints:
# m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t]

def enforce_generator_output_limits_rule_part_a(m, g, t):
   return m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t]

model.EnforceGeneratorOutputLimitsPartA = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_a)

def enforce_generator_output_limits_rule_part_b(m, g, t):
   return m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]

model.EnforceGeneratorOutputLimitsPartB = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_b)

# CASM - (ancillary service) regulation high limit is also enforce here
def enforce_generator_output_limits_rule_part_c(m, g, t):
    if regulation_services:
        return m.MaximumPowerAvailable[g,t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t] - m.MaxPowerOutputMinusRegHighLimit[g] * m.RegulationOn[g, t]
    else:
        return m.MaximumPowerAvailable[g,t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t]
model.EnforceGeneratorOutputLimitsPartC = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_c)

# note: as of 9 Feb 2012 wind is done using Var bounds

# impose upper bounds on the maximum power available for each generator in each time period,
# based on standard and start-up ramp limits.

# the following constraint encodes Constraint 18 defined in Carrion and Arroyo.

def enforce_max_available_ramp_up_rates_rule(m, g, t):
   # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from m.PowerGenerated[g, t-1])
   # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound due to unit being off)
   # (0, 1) - unit switching on:  RHS = startup ramp limit
   # (1, 0) - unit switching off: RHS = standard ramp limit minus startup ramp limit plus maximum power output (degenerate upper bound due to unit off)
   # (1, 1) - unit staying on:    RHS = standard ramp limit plus power generated in previous time period
   if t == m.InitialTime:
      return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] + \
                                              m.ScaledNominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                              m.ScaledStartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]) + \
                                              m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
   else:
      return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g, t-1] + \
                                              m.ScaledNominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                              m.ScaledStartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1]) + \
                                              m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])

model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)

# the following constraint encodes Constraint 19 defined in Carrion and Arroyo.

def enforce_max_available_ramp_down_rates_rule(m, g, t):
    # 4 cases, split by (t, t+1) unit status
    # (0, 0) - unit staying off:   RHS = 0 (degenerate upper bound)
    # (0, 1) - unit switching on:  RHS = maximum generator output minus shutdown ramp limit (degenerate upper bound) - this is the strangest case.
    # (1, 0) - unit switching off: RHS = shutdown ramp limit
    # (1, 1) - unit staying on:    RHS = maximum generator output (degenerate upper bound)
    #NOTE: As expressed in Carrion-Arroyo and subsequently here, this constraint does NOT consider ramp down from initial conditions to t=1!
    #if t == value(m.NumTimePeriods):
    #   return Constraint.Skip
    #else:
    #   return m.MaximumPowerAvailable[g, t] <= \
    #          m.MaximumPowerOutput[g] * m.UnitOn[g, t+1] + \
    #          m.ScaledShutdownRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t+1])

    #This version fixes the problem with ignoring initial conditions mentioned in the above note
    if t == m.InitialTime:
        # Not 100% sure of this one since there is no MaximumPowerAvailableT0
        if not enforce_t1_ramp_rates: #CASM: Alstom model does not enforce ramp rates on first time step
            return Constraint.Skip
        else:
            return m.PowerGeneratedT0[g] <= \
                 m.MaximumPowerOutput[g] * m.UnitOn[g,t] + \
                 m.ScaledShutdownRampLimit[g] * (m.UnitOnT0[g] - m.UnitOn[g,t])
    else:
        return m.MaximumPowerAvailable[g, t-1] <= \
                 m.MaximumPowerOutput[g] * m.UnitOn[g, t] + \
                 m.ScaledShutdownRampLimit[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])

model.EnforceMaxAvailableRampDownRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_down_rates_rule)

# the following constraint encodes Constraint 20 defined in Carrion and Arroyo.

def enforce_ramp_down_limits_rule(m, g, t):
    # 4 cases, split by (t-1, t) unit status:
    # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound)
    # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus shutdown ramp limit plus maximum generator output - this is the strangest case.
    #NOTE: This may never be physically true, but if a generator has ScaledShutdownRampLimit >> MaximumPowerOutput, this constraint causes problems
    # (1, 0) - unit switching off: RHS = shutdown ramp limit
    # (1, 1) - unit staying on:    RHS = standard ramp-down limit
    if t == m.InitialTime:
        if not enforce_t1_ramp_rates:
            return Constraint.Skip
        else:
            return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= \
                 m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] + \
                 m.ScaledShutdownRampLimit[g]  * (m.UnitOnT0[g] - m.UnitOn[g, t]) + \
                 m.MaximumPowerOutput[g] * (1 - m.UnitOnT0[g])
    else:
        return m.PowerGenerated[g, t-1] - m.PowerGenerated[g, t] <= \
             m.ScaledNominalRampDownLimit[g]  * m.UnitOn[g, t] + \
             m.ScaledShutdownRampLimit[g]  * (m.UnitOn[g, t-1] - m.UnitOn[g, t]) + \
             m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t-1])

model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)

if storage_services:

    #######################################
    # energy storage bounding constraints #
    #######################################
    # NOTE: The expressions below are what we really want - however, due to a pyomo design feature, we have to split it into two constraints:
    # m.MinimumPowerInputStorage[g] * m.InputStorage[g, t] <= m.StoragePowerInput[g,t] <= m.MaximumPowerInputStorage[g] * m.InputStorage[g, t]
    # m.MinimumPowerOutputStorage[g] * m.OutputStorage[g, t] <= m.StoragePowerOutput[g,t] <= m.MaximumPowerOutputStorage[g] * m.OutputStorage[g, t]

    def enforce_storage_input_limits_rule_part_a(m, s, t):
       return m.MinimumPowerInputStorage[s] * m.InputStorage[s, t] <= m.PowerInputStorage[s,t]

    model.EnforceStorageInputLimitsPartA = Constraint(model.Storage, model.TimePeriods, rule=enforce_storage_input_limits_rule_part_a)

    def enforce_storage_input_limits_rule_part_b(m, s, t):
       return m.PowerInputStorage[s,t] <= m.MaximumPowerInputStorage[s] * m.InputStorage[s, t]

    model.EnforceStorageInputLimitsPartB = Constraint(model.Storage, model.TimePeriods, rule=enforce_storage_input_limits_rule_part_b)

    def enforce_storage_output_limits_rule_part_a(m, s, t):
       return m.MinimumPowerOutputStorage[s] * m.OutputStorage[s, t] <= m.PowerOutputStorage[s,t]

    model.EnforceStorageOutputLimitsPartA = Constraint(model.Storage, model.TimePeriods, rule=enforce_storage_output_limits_rule_part_a)

    def enforce_storage_output_limits_rule_part_b(m, s, t):
       return m.PowerOutputStorage[s,t] <= m.MaximumPowerOutputStorage[s] * m.OutputStorage[s, t]

    model.EnforceStorageOutputLimitsPartB = Constraint(model.Storage, model.TimePeriods, rule=enforce_storage_output_limits_rule_part_b)

    def enforce_input_output_exclusivity_rule(m, s, t):
        return m.PowerOutputStorage[s,t] + m.PowerInputStorage[s,t] <= 1

    #model.EnforceInputOutputExclusivity = Constraint(model.Storage, model.TimePeriods, rule=enforce_input_output_exclusivity_rule)

    #####################################
    # energy storage ramping contraints #
    #####################################

    def enforce_ramp_up_rates_power_output_storage_rule(m, s, t):
       if t == m.InitialTime:
          return m.PowerOutputStorage[s, t] <= m.StoragePowerOutputOnT0[s] + m.ScaledNominalRampUpLimitStorageOutput[s]
       else:
          return m.PowerOutputStorage[s, t] <= m.PowerOutputStorage[s, t-1] + m.ScaledNominalRampUpLimitStorageOutput[s]

    model.EnforceStorageOutputRampUpRates = Constraint(model.Storage, model.TimePeriods, rule=enforce_ramp_up_rates_power_output_storage_rule)

    def enforce_ramp_down_rates_power_output_storage_rule(m, s, t):
       if t == m.InitialTime:
          return m.PowerOutputStorage[s, t] >= m.StoragePowerOutputOnT0[s] - m.ScaledNominalRampDownLimitStorageOutput[s]
       else:
          return m.PowerOutputStorage[s, t] >= m.PowerOutputStorage[s, t-1] - m.ScaledNominalRampDownLimitStorageOutput[s]

    model.EnforceStorageOutputRampDownRates = Constraint(model.Storage, model.TimePeriods, rule=enforce_ramp_down_rates_power_output_storage_rule)

    def enforce_ramp_up_rates_power_input_storage_rule(m, s, t):
       if t == m.InitialTime:
          return m.PowerInputStorage[s, t] <= m.StoragePowerInputOnT0[s] + m.ScaledNominalRampUpLimitStorageInput[s]
       else:
          return m.PowerInputStorage[s, t] <= m.PowerInputStorage[s, t-1] + m.ScaledNominalRampUpLimitStorageInput[s]

    model.EnforceStorageInputRampUpRates = Constraint(model.Storage, model.TimePeriods, rule=enforce_ramp_up_rates_power_input_storage_rule)

    def enforce_ramp_down_rates_power_input_storage_rule(m, s, t):
       if t == m.InitialTime:
          return m.PowerInputStorage[s, t] >= m.StoragePowerInputOnT0[s] - m.ScaledNominalRampDownLimitStorageInput[s]
       else:
          return m.PowerInputStorage[s, t] >= m.PowerInputStorage[s, t-1] - m.ScaledNominalRampDownLimitStorageInput[s]

    model.EnforceStorageInputRampDownRates = Constraint(model.Storage, model.TimePeriods, rule=enforce_ramp_down_rates_power_input_storage_rule)

    ##########################################
    # storage energy conservation constraint #
    ##########################################

    def energy_conservation_rule(m, s, t):
        # storage s, time t
        if t == m.InitialTime:
            return m.SocStorage[s, t] == m.StorageSocOnT0[s]  + \
                   (- m.PowerOutputStorage[s, t] + m.PowerInputStorage[s,t]*m.EfficiencyEnergyStorage[s])/m.MaximumEnergyStorage[s]
        else:
            return m.SocStorage[s, t] == m.SocStorage[s, t-1]  + \
                   (- m.PowerOutputStorage[s, t] + m.PowerInputStorage[s,t]*m.EfficiencyEnergyStorage[s])/m.MaximumEnergyStorage[s]
    model.EnergyConservation = Constraint(model.Storage, model.TimePeriods, rule=energy_conservation_rule)

    ##################################
    # storage end-point constraints  #
    ##################################

    def storage_end_point_soc_rule(m, s):
        # storage s, last time period
        return m.SocStorage[s, value(m.NumTimePeriods)] == m.EndPointSocStorage[s]
    #model.EnforceEndPointSocStorage = Constraint(model.Storage, rule=storage_end_point_soc_rule)

#############################################
# constraints for computing cost components #
#############################################

# a function for use in piecewise linearization of the cost function.
def production_cost_function(m, g, t, x):
    return m.TimePeriodLength * m.PowerGenerationPiecewiseValues[g,t][x] * m.FuelCost[g]

# compute the per-generator, per-time period production costs. this is a "simple" piecewise linear construct.
# the first argument to piecewise is the index set. the second and third arguments are respectively the input and output variables.
model.ComputeProductionCosts = Piecewise(model.ThermalGenerators * model.TimePeriods, model.ProductionCost, model.PowerGenerated, pw_pts=model.PowerGenerationPiecewisePoints, f_rule=production_cost_function, pw_constr_type='LB')

# compute the total production costs, across all generators and time periods.
def compute_total_production_cost_rule(m, t):
    return m.TotalProductionCost[t] == sum(m.ProductionCost[g, t] for g in m.ThermalGenerators)

model.ComputeTotalProductionCost = Constraint(model.TimePeriods, rule=compute_total_production_cost_rule)

def compute_total_no_load_cost_rule(m,t):
    return m.TotalNoLoadCost[t] == sum(m.MinimumProductionCost[g] * m.UnitOn[g,t] for g in m.ThermalGenerators)

model.ComputeTotalNoLoadCost = Constraint(model.TimePeriods, rule=compute_total_no_load_cost_rule)

############################################################
# compute the per-generator, per-time period startup costs #
############################################################

def startup_costs_index_set_generator(m):
   return ((g,t,i) for t in m.TimePeriods for g in m.ThermalGenerators for i in m.StartupCostIndices[g])

model.StartupCostsIndexSet = Set(initialize=startup_costs_index_set_generator, dimen=3)

def compute_startup_costs_rule(m, g, t, i):
    # irios, Nov 18th: I had to change this because the last version didn't work with the last update in coopr.
    this_lag = list(m.StartupLags[g])[i-1]
    this_cost = list(m.StartupCosts[g])[i-1]
    
    generator_t0_state = value(m.UnitOnT0State[g])

    # if the lag involves time periods preceding t=1, then we need to determine if the T0 
    # state is consistent with the lag - if not, we can skip generation of the constraint.
    if this_lag >= t:
        if generator_t0_state >= 0:
            # the unit has been on - we can't meet the target lag.
            return Constraint.Skip

        time_diff = this_lag - t + 1
        if (-generator_t0_state) < time_diff:
            # the generator has not been off for a sufficient number of time periods.
            return Constraint.Skip

    # can only "look back" in terms of UnitOn variable state (t-1) or this_lag time periods - whichever is smallest.
    # the rest of the time period is captured in the unit T0 state, and is handled in the logic above.
    
    #---------------------------- irios ------------------------------------------
    # When we solve starting from a node in a later stage we have to consider what happended in the previous hours.
    # Then, if generator_T0_state is positive (i.e the unit was on in the previous hours) the constraint is not active, so we can skip it.
    if int(t - min(t, this_lag+1) + 1) not in m.TimePeriods: #e.g: t = 14 (stage 2), tmax = 4, then t-tmax = 10 is not in TimePeriods (because we are solving for a subtree)
        if generator_t0_state >= 0: #e.g: UOT0S = 4
            return Constraint.Skip
    #---------------------------- irios ------------------------------------------

    return m.StartupCost[g, t] >= this_cost * (m.UnitOn[g, t] - sum(m.UnitOn[g, t - k] for k in xrange(1, min(t-value(m.InitialTime)+1, this_lag+1))))

model.ComputeStartupCosts = Constraint(model.StartupCostsIndexSet, rule=compute_startup_costs_rule)

#############################################################
# compute the per-generator, per-time period shutdown costs #
#############################################################

def compute_shutdown_costs_rule(m, g, t):
   if t == m.InitialTime:
      return m.ShutdownCost[g, t] >= m.ShutdownFixedCost[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
   else:
      return m.ShutdownCost[g, t] >= m.ShutdownFixedCost[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])

model.ComputeShutdownCosts = Constraint(model.ThermalGenerators, model.TimePeriods, rule=compute_shutdown_costs_rule)

#######################
# up-time constraints #
#######################

# constraint due to initial conditions.
def enforce_up_time_constraints_initial(m, g):
    if m.InitialTime > value(m.InitialTimePeriodsOnLine[g]):
        return Constraint.Skip
    else:
        return sum((1 - m.UnitOn[g, t]) for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOnLine[g])) == 0.0

model.EnforceUpTimeConstraintsInitial = Constraint(model.ThermalGenerators, rule=enforce_up_time_constraints_initial)

# constraint for each time period after that not involving the initial condition.
@simple_constraint_rule
def enforce_up_time_constraints_subsequent(m, g, t):
   if t <= value(m.InitialTimePeriodsOnLine[g]):
      # handled by the EnforceUpTimeConstraintInitial constraint.
      return Constraint.Skip
   elif t <= (value(m.NumTimePeriods - m.ScaledMinimumUpTime[g]) + 1):
      # the right-hand side terms below are only positive if the unit was off in the previous time period but on in this one =>
      # the value is the minimum number of subsequent consecutive time periods that the unit is required to be on.
      if t == m.InitialTime:
         return sum(m.UnitOn[g, n] for n in m.TimePeriods if n >= t and n <= (t + value(m.ScaledMinimumUpTime[g]) - 1)) >= \
                m.ScaledMinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOnT0[g])
      else:
         return sum(m.UnitOn[g, n] for n in m.TimePeriods if n >= t and n <= (t + value(m.ScaledMinimumUpTime[g]) - 1)) >= \
                m.ScaledMinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1])
   else:
      # handle the final (ScaledMinimumUpTime[g] - 1) time periods - if a unit is started up in
      # this interval, it must remain on-line until the end of the time span.
      if t == m.InitialTime: # can happen when small time horizons are specified
         return sum((m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOnT0[g])) for n in m.TimePeriods if n >= t) >= 0.0
      else:
         return sum((m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOn[g, t-1])) for n in m.TimePeriods if n >= t) >= 0.0

model.EnforceUpTimeConstraintsSubsequent = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_up_time_constraints_subsequent)

#########################
# down-time constraints #
#########################

# constraint due to initial conditions.
def enforce_down_time_constraints_initial(m, g):
    if m.InitialTime > value(m.InitialTimePeriodsOffLine[g]):
        return Constraint.Skip
    else:
        return sum(m.UnitOn[g, t] for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOffLine[g])) == 0.0

model.EnforceDownTimeConstraintsInitial = Constraint(model.ThermalGenerators, rule=enforce_down_time_constraints_initial)

# constraint for each time period after that not involving the initial condition.
@simple_constraint_rule
def enforce_down_time_constraints_subsequent(m, g, t):
   if t <= value(m.InitialTimePeriodsOffLine[g]):
      # handled by the EnforceDownTimeConstraintInitial constraint.
      return Constraint.Skip
   elif t <= (value(m.NumTimePeriods - m.ScaledMinimumDownTime[g]) + 1):
      # the right-hand side terms below are only positive if the unit was off in the previous time period but on in this one =>
      # the value is the minimum number of subsequent consecutive time periods that the unit is required to be on.
      if t == m.InitialTime:
         return sum((1 - m.UnitOn[g, n]) for n in m.TimePeriods if n >= t and n <= (t + value(m.ScaledMinimumDownTime[g]) - 1)) >= \
                m.ScaledMinimumDownTime[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
      else:
         return sum((1 - m.UnitOn[g, n]) for n in m.TimePeriods if n >= t and n <= (t + value(m.ScaledMinimumDownTime[g]) - 1)) >= \
                m.ScaledMinimumDownTime[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])
   else:
      # handle the final (ScaledMinimumDownTime[g] - 1) time periods - if a unit is shut down in
      # this interval, it must remain off-line until the end of the time span.
      if t == m.InitialTime: # can happen when small time horizons are specified
         return sum(((1 - m.UnitOn[g, n]) - (m.UnitOnT0[g] - m.UnitOn[g, t])) for n in m.TimePeriods if n >= t) >= 0.0
      else:
         return sum(((1 - m.UnitOn[g, n]) - (m.UnitOn[g, t-1] - m.UnitOn[g, t])) for n in m.TimePeriods if n >= t) >= 0.0

model.EnforceDownTimeConstraintsSubsequent = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_down_time_constraints_subsequent)

#################################
# Regulation ancillary services #
#################################

if regulation_services:

    # constraint whether a unit can provide regulation
    def identify_regulation_providers_rule(m, g, t):
        if m.RegulationProvider[g] == 0:
            return m.RegulationOn[g, t] == 0
        else:
            return Constraint.Skip

    model.IdentifyRegulationProviders = Constraint(model.ThermalGenerators, model.TimePeriods, rule=identify_regulation_providers_rule)

    # constrain the min power output of a unit to regulation low limit
    def enforce_generator_ouptut_low_limit_regulation_rule(m, g, t):
        return m.RegulationLowLimit[g] * m.RegulationOn[g, t] <= m.PowerGenerated[g, t]

    model.EnforceGeneratorOutputLowLimitRegulation = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_ouptut_low_limit_regulation_rule)

    # Regulation high limit is enforce with the Maximum Power Available constraint.

    # a generator can provide regulation only when it's on
    def provide_regulation_when_unit_on_rule(m, g, t):
        return m.RegulationOn[g, t] <= m.UnitOn[g, t]

    model.EnforceRegulationOnWhenUnitOn = Constraint(model.ThermalGenerators, model.TimePeriods, rule=provide_regulation_when_unit_on_rule)

    # zonal regulation calculation
    def enforce_zonal_regulation_requirement_rule(m, rz, t):
        expr = (m.ZonalRegulationRequirement[rz, t] == 0.0)
        if expr == True:
            return Constraint.Feasible
        else:
            return m.ZonalRegulationRequirement[rz, t] <= sum(m.RegulationOn[g, t] * m.RegulationCapability[g] for g in m.ThermalGeneratorsInReserveZone[rz])

    model.EnforceZonalRegulationRequirement = Constraint(model.ReserveZones, model.TimePeriods, rule=enforce_zonal_regulation_requirement_rule)

    # global regulation calculation
    def calculate_total_regulation_capability_available(m, t):
        return m.TotalRegulationCapabilityAvailable[t] == sum(m.RegulationOn[g, t] * m.RegulationCapability[g] for g in m.ThermalGenerators)

    model.CalculateTotalRegulationCapabilityAvailable = Constraint(model.TimePeriods, rule=calculate_total_regulation_capability_available)

    def enforce_global_regulation_requirement_rule(m, t):
        expr = (m.GlobalRegulationRequirement[t] == 0.0)
        if expr == True:
            return Constraint.Feasible
        else:
            return m.GlobalRegulationRequirement[t] <= sum(m.RegulationOn[g, t] * m.RegulationCapability[g] for g in m.ThermalGenerators)

    model.EnforceGlobalRegulationRequirement = Constraint(model.TimePeriods, rule=enforce_global_regulation_requirement_rule)

    def compute_regulation_cost_rule(m, t):
        return m.RegulationCost[t] >= sum(m.RegulationOffer[g] * m.RegulationOn[g, t] for g in m.ThermalGenerators)

    #model.EnforceRegulationCost = Constraint(model.TimePeriods, rule=compute_regulation_cost_rule)

    def compute_total_regulation_cost_rule(m):
        return m.TotalRegulationCost == sum(m.RegulationCost[t] for t in m.TimePeriods)

if reserve_services:
    # spinning reserve

    def calculate_spinning_reserve_available_rule_part_a(m, g, t):
        return m.SpinningReserveDispatched[g, t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t] - m.PowerGenerated[g, t]

    model.CalculateSpinningReserveAvailableRulePartA = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_spinning_reserve_available_rule_part_a)

    def calculate_spinning_reserve_available_rule_part_b(m, g, t):
        return m.SpinningReserveDispatched[g, t] <= m.NominalRampUpLimit[g] * m.SpinningReserveTime

    model.CalculateSpinningReserveAvailableRulePartB = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_spinning_reserve_available_rule_part_b)

    def enforce_zonal_spinning_reserve_requirement(m, rz, t):
        return m.ZonalSpinningReserveRequirement[rz, t] <= sum(m.SpinningReserveDispatched[g, t] for g in m.ThermalGeneratorsInReserveZone[rz])

    model.EnforceZonalSpinningReserveRequirement = Constraint(model.ReserveZones, model.TimePeriods, rule=enforce_zonal_spinning_reserve_requirement)

    def enforce_global_spinning_reserve_requirement(m, t):
        return m.SystemSpinningReserveRequirement[t] <= sum(m.SpinningReserveDispatched[g, t] for g in m.ThermalGenerators)

    model.EnforceSystemSpinningReserveRequirement = Constraint(model.TimePeriods, rule=enforce_global_spinning_reserve_requirement)

    def compute_spinning_reserve_cost(m, g, t):
        return m.SpinningReserveCost[g, t] >= m.SpinningReserveDispatched[g, t] * m.SpinningReserveOffer[g] * m.TimePeriodLength

    model.ComputeSpinningReserveCost = Constraint(model.ThermalGenerators, model.TimePeriods, rule=compute_spinning_reserve_cost)

    def compute_total_spinning_reserve_cost(m):
        return m.TotalSpinningReserveCost >= sum(m.SpinningReserveCost[g, t]  for g in m.ThermalGenerators for t in m.TimePeriods)

    model.ComputeTotalSpinningReserveCost = Constraint(rule=compute_total_spinning_reserve_cost)

    # non-spinning reserve

    def calculate_non_spinning_reserve_limit_rule(m, g, t):
        return m.NonSpinningReserveDispatched[g, t] <= m.NonSpinningReserveAvailable[g] * (1 - m.UnitOn[g, t])

    model.CalculateNonSpinningReserveLimit = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_non_spinning_reserve_limit_rule)

    def calculate_non_spinning_reserve_cost(m, g, t):
        return m.NonSpinningReserveCost[g, t] >= m.NonSpinningReserveDispatched[g, t] * m.NonSpinningReserveOffer[g] * m.TimePeriodLength

    model.CalculateNonSpinningReserveCost = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_non_spinning_reserve_cost)

    def enforce_zonal_non_spinning_reserve_rule(m, rz, t):
        return m.ZonalTenMinuteReserveRequirement[rz, t] <= sum(m.SpinningReserveDispatched[g, t] + m.NonSpinningReserveDispatched[g, t] \
                                                                for g in m.ThermalGeneratorsInReserveZone[rz])

    model.EnforceTenMinuteZonalReserveRequirement = Constraint(model.ReserveZones, model.TimePeriods, rule=enforce_zonal_non_spinning_reserve_rule)

    def enforce_system_ten_minute_reserve_requirement(m, t):
        return m.SystemTenMinuteReserveRequirement[t] <= sum(m.SpinningReserveDispatched[g, t] + m.NonSpinningReserveDispatched[g, t] for g in m.ThermalGenerators)

    model.EnforceSystemTenMinuteReserveRequirement = Constraint(model.TimePeriods, rule=enforce_system_ten_minute_reserve_requirement)

    def compute_non_spinning_reserve_total_cost(m):
        return m.TotalNonSpinningReserveCost >= sum(m.NonSpinningReserveCost[g, t] for g in m.ThermalGenerators for t in m.TimePeriods)

    model.ComputeNonSpinningReserveCost = Constraint(rule=compute_non_spinning_reserve_total_cost)

    # thirty-minute operating reserve

    def calculate_operating_reserve_limit_rule(m, g, t):
        return m.OperatingReserveDispatched[g, t] + m.NonSpinningReserveDispatched[g, t] <= m.OperatingReserveAvailable[g] * (1 - m.UnitOn[g, t])

    model.CalculateOperatingReserveLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_operating_reserve_limit_rule)

    def enforce_zonal_operating_reserve_requirement_rule(m, rz, t):
        return m.ZonalOperatingReserveRequirement[rz, t] <= sum(m.SpinningReserveDispatched[g, t] + m.NonSpinningReserveDispatched[g, t] \
                                                                + m.OperatingReserveDispatched[g, t] for g in m.ThermalGeneratorsInReserveZone[rz])

    model.EnforceZonalOperatingReserveRequirement = Constraint(model.ReserveZones, model.TimePeriods, rule=enforce_zonal_operating_reserve_requirement_rule)

    def enforce_system_operating_reserve_requirement(m, t):
        return m.SystemOperatingReserveRequirement[t] <= sum(m.SpinningReserveDispatched[g, t] + m.NonSpinningReserveDispatched[g, t] \
                                                             + m.OperatingReserveDispatched[g, t] for g in m.ThermalGenerators)

    model.EnforceSystemOperatingReserveRequirement = Constraint(model.TimePeriods, rule=enforce_system_operating_reserve_requirement)

    def calculate_operating_reserve_cost_rule(m, g, t):
        return m.OperatingReserveCost[g, t] >= m.OperatingReserveDispatched[g, t] * m.OperatingReserveOffer[g] * m.TimePeriodLength

    model.CalculateOperatingReserveCost = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_operating_reserve_cost_rule)

    def calculate_operating_reserve_total_cost(m):
        return m.TotalOperatingReserveCost >= sum(m.OperatingReserveCost[g, t] for g in m.ThermalGenerators for t in m.TimePeriods)

    model.CalculateOperatingReserveTotalCost = Constraint(rule=calculate_operating_reserve_total_cost)

if multistage:
    # we create a set of tuples Stage - CTIS
    def SCTIS_init(m):
        return ((s, t) for s in m.StageSet for t in m.CommitmentTimeInStage[s])
    model.SCTIS = Set(dimen=2, initialize=SCTIS_init)
    
    # indicator variables that tell us if a given generator was commited at a given stage for a given time
    model.UnitCommited = Var(model.ThermalGenerators, model.SCTIS, within=Binary, initialize=1)

    def FindStageIndex(m,s):
        idx = 1
        for st in m.StageSet:
            if st == s:
                break
            else:
                idx += 1
        return idx
    model.StageIndex = Param(model.StageSet, initialize=FindStageIndex)

    # Value of commitments at time T0
    def initial_commitments_validator(m, v, g, t):
       return v >= 0.0 and v <= 1.0
    model.UnitCommitedT0 = Param(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, validate=initial_commitments_validator, mutable=True, default=0.0)

    # Fixed Cost of commiting units
    def compute_commitment_fixed_cost_rule(m, g, s):
        return m.MinimumProductionCost[g]
    model.CommitmentFixedCost = Param(model.ThermalGenerators, model.StageSet, within=NonNegativeReals, initialize=compute_commitment_fixed_cost_rule)
    model.CommitmentCost = Var(model.ThermalGenerators, model.SCTIS, within=NonNegativeReals, initialize=0.0)

    # Cost of commiting units
    def compute_commitment_cost_rule(m, g, s, t):
        i = m.StageIndex[s]
        if s == m.StageSet[1]:
            return m.CommitmentCost[g,s,t] >= m.CommitmentFixedCost[g,s] * (m.UnitCommited[g, s, t] - m.UnitCommitedT0[g, t])
        else:
            return m.CommitmentCost[g,s,t] >= m.CommitmentFixedCost[g,s] * (m.UnitCommited[g, s, t] - m.UnitCommited[g, m.StageSet[i-1], t])
    model.ComputeCommitmentCost = Constraint(model.ThermalGenerators, model.SCTIS, rule=compute_commitment_cost_rule)

    # Cost of de-commiting units
    # Fixed Cost of commiting units
    def compute_decommitment_fixed_cost_rule(m, g):
        return m.MinimumProductionCost[g]
    model.deCommitmentFixedCost = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=compute_decommitment_fixed_cost_rule) # units are $.
    model.deCommitmentCost = Var(model.ThermalGenerators, model.SCTIS, within=NonNegativeReals, initialize=0.0)

    def compute_decommitment_costs_rule(m, g, s, t):
        i = m.StageIndex[s]
        if s == m.StageSet[1]:
            return m.deCommitmentCost[g, s, t] >= m.deCommitmentFixedCost[g] * (m.UnitCommitedT0[g, t] - m.UnitCommited[g, s, t])
        else:
            return m.deCommitmentCost[g, s, t] >= m.deCommitmentFixedCost[g] * (m.UnitCommited[g, m.StageSet[i-1], t] - m.UnitCommited[g, s, t])
    model.ComputedeCommitmentCosts = Constraint(model.ThermalGenerators, model.SCTIS, rule=compute_decommitment_costs_rule)

    # Relation between planning (UnitCommited) and physical (UnitOn) variables: a Unit that is on at a given time t must have been commited at least in the 
    # stage inmediatly prior to that stage.
    def relate_planning_and_physical_variables(m, g, t):
        for s in m.StageSet:
            if t in m.GenerationTimeInStage[s]:
                break
        st = m.StageSet[FindStageIndex(m,s)-1]
        return m.UnitCommited[g, st, t] == m.UnitOn[g, t]

    model.RelatePlanningandPhysicalVariables = Constraint(model.ThermalGenerators, model.TimePeriods, rule=relate_planning_and_physical_variables)

# 
# Cost computations
#

def commitment_in_stage_st_cost_rule(m, st):
    if not multistage: # if we are not solving the multistage version the startup and shutdown costs are first stage costs.
        cc = (sum(m.StartupCost[g,t] + m.ShutdownCost[g,t] for g in m.ThermalGenerators for t in m.CommitmentTimeInStage[st]) + sum(sum(m.UnitOn[g,t] for t in m.CommitmentTimeInStage[st]) * m.MinimumProductionCost[g] * m.TimePeriodLength for g in m.ThermalGenerators))
    else:
        cc = sum(m.CommitmentCost[g, st, t] + m.deCommitmentCost[g, st, t] for g in m.ThermalGenerators for t in m.CommitmentTimeInStage[st])
    if regulation_services:
        cc += sum(m.RegulationCost[t] for t in m.CommitmentTimeInStage[st])
    return m.CommitmentStageCost[st] == cc
model.Compute_commitment_in_stage_st_cost = Constraint(model.StageSet, rule = commitment_in_stage_st_cost_rule)

def generation_in_stage_st_cost_rule(m, st):
    gc = sum(m.ProductionCost[g, t] for g in m.ThermalGenerators for t in m.GenerationTimeInStage[st]) +  \
        (m.LoadMismatchPenalty * sum(m.posLoadGenerateMismatch[b, t] + m.negLoadGenerateMismatch[b, t] for b in m.Buses for t in m.GenerationTimeInStage[st])) + \
        (m.ReserveMismatchPenalty * sum(m.posReserveMismatch[t] + m.negReserveMismatch[t] for t in m.GenerationTimeInStage[st]))
    if multistage: # if we are solving multistage Stratup and Shutdown are physical changes, so they are considered as part of the generation costs
        gc += (sum(m.StartupCost[g,t] + m.ShutdownCost[g,t] for g in m.ThermalGenerators for t in m.GenerationTimeInStage[st]) + sum(sum(m.UnitOn[g,t] for t in m.GenerationTimeInStage[st]) * m.MinimumProductionCost[g] * m.TimePeriodLength for g in m.ThermalGenerators))
    return m.GenerationStageCost[st] == gc
model.Compute_generation_in_stage_st_cost = Constraint(model.StageSet, rule = generation_in_stage_st_cost_rule)

def StageCost_rule(m, st):
    return m.StageCost[st] == m.GenerationStageCost[st] + m.CommitmentStageCost[st]
model.Compute_Stage_Cost = Constraint(model.StageSet, rule = StageCost_rule)

#
# Objectives
#

def total_cost_objective_rule(m):
   return sum(m.StageCost[st] for st in m.StageSet)	

model.TotalCostObjective = Objective(rule=total_cost_objective_rule, sense=minimize)