import matplotlib
matplotlib.use('Agg')
import datetime
import sys
import os
import shutil
import argparse
import traceback
import multiprocessing

from collections import OrderedDict
from random import choice

import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import prescient.gosm.sources as sources
from prescient.distributions.distributions import UnivariateEmpiricalDistribution
from prescient.gosm.structures.skeleton_scenario import PowerScenario, SkeletonScenarioSet
from prescient.gosm.markov_chains import markov_chains, states, descriptions


parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
required.add_argument('--power-source',
                    help='The name of the file containing the source',
                    action='store',
                    type=str,
                    dest='power_source',
                    required=True)

parser.add_argument('--error-bin-size',
                    help='The bin size for each different error state',
                    action='store',
                    type=float,
                    dest='error_bin_size',
                    default=100)

parser.add_argument('--use-error-quantiles',
                    help='Set this option if you wish to use quantiles instead'
                         ' of raw MegaWatt windows when computing the state. '
                         'This option should be used in conjunction with the '
                         'option --error-quantile-size.',
                    action='store_true',
                    dest='use_error_quantiles')

parser.add_argument('--error-quantile-size',
                    help='The size of each quantile interval in considered in '
                         'the state. This should be less than 1 and evenly '
                         'divide 1. If it does not, the last interval will be '
                         'larger than the rest. For an example of how this '
                         'affects the program, if it is set to 0.25, the error'
                         ' states will be (0,0.25),(0.25,0.5),(0.5,0.75), and '
                         ' (0.75,1)',
                    dest='error_quantile_size',
                    type=float,
                    default=0.1)

parser.add_argument('--explicit-error-quantiles',
                    help='Set this option to a string specifying the exact '
                         'quantiles you wish to use as the error state. This '
                         'will be a comma-delimited string starting with 0 and'
                         'ending with 1. For example, this can be 0,0.1,0.6,1',
                    dest='explicit_error_quantiles',
                    type=str)

parser.add_argument('--consider-forecasts',
                    help='Include this option if we are to consider forecasts '
                         'as part of the state.',
                    action='store_true',
                    dest='consider_forecasts')

parser.add_argument('--forecast-bin-size',
                    help='The bin size for each different forecast state',
                    action='store',
                    type=float,
                    dest='forecast_bin_size',
                    default=100)

parser.add_argument('--use-forecast-quantiles',
                    help='Set this option if you wish to use quantiles instead'
                         ' of raw MegaWatt windows when computing the state. '
                         'This option should be used in conjunction with the '
                         'option --forecast-quantile-size.',
                    action='store_true',
                    dest='use_forecast_quantiles')

parser.add_argument('--forecast-quantile-size',
                    help='The size of each quantile interval in considered in '
                         'the state. This should be less than 1 and evenly '
                         'divide 1. If it does not, the last interval will be '
                         'larger than the rest. For an example of how this '
                         'affects the program, if it is set to 0.25, the '
                         'forecast states will be (0,0.25),(0.25,0.5), '
                         '(0.5,0.75), and (0.75,1).',
                    dest='forecast_quantile_size',
                    type=float,
                    default=0.1)

parser.add_argument('--explicit-forecast-quantiles',
                    help='Set this option to a string specifying the exact '
                         'quantiles you wish to use as the forecast state. This '
                         'will be a comma-delimited string starting with 0 and'
                         'ending with 1. For example, this can be 0,0.1,0.6,1',
                    dest='explicit_forecast_quantiles',
                    type=str)

parser.add_argument('--consider-derivatives',
                    help='Include this option if we are to consider '
                         'forecast derivatives as part of the state.',
                    action='store_true',
                    dest='consider_derivatives')

parser.add_argument('--derivative-bin-size',
                    help='The bin size for each different derivative state',
                    type=float,
                    dest='derivative_bin_size',
                    default=100)

parser.add_argument('--use-derivative-quantiles',
                    help='Set this option if you wish to use quantiles instead'
                         ' of raw MegaWatt windows when computing the state. '
                         'This option should be used in conjunction with the '
                         'option --derivative-quantile-size.',
                    action='store_true',
                    dest='use_derivative_quantiles')

parser.add_argument('--derivative-quantile-size',
                    help='The size of each quantile interval in considered in '
                         'the state. This should be less than 1 and evenly '
                         'divide 1. If it does not, the last interval will be '
                         'larger than the rest. For an example of how this '
                         'affects the program, if it is set to 0.25, the'
                         ' derivative states will be (0,0.25),(0.25,0.5),'
                         ' (0.5,0.75), and (0.75,1)',
                    dest='derivative_quantile_size',
                    type=float,
                    default=0.1)

parser.add_argument('--explicit-derivative-quantiles',
                    help='Set this option to a string specifying the exact '
                         'quantiles you wish to use as the derivative state. '
                         'This will be a comma-delimited string starting with '
                         '0 and ending with 1. For example, this can be '
                         '0,0.1,0.6,1',
                    dest='explicit_derivative_quantiles',
                    type=str)

parser.add_argument('--state-memory',
                    help='The amount of states to "remember" when making the'
                         'next transition in the Markov Chain',
                    type=int,
                    dest='state_memory',
                    default=1)

parser.add_argument('--use-equal-probability',
                    help='Set this option if, you want each transition to be'
                         'of equal probability',
                    action='store_true',
                    dest='use_equal_probability')

required.add_argument('--output-directory',
                    help='Where to save any output files produced from the'
                         'output of this script',
                    type=str,
                    dest='output_directory',
                    required=True)

required.add_argument('--start-date',
                    help="The first date for which you wish to start "
                         "computing scenarios. This should be in the "
                         "YYYY-MM-DD format.",
                    type=str,
                    dest='start_date',
                    required=True)

required.add_argument('--end-date',
                    help="The first date for which you wish to start "
                         "computing scenarios. This should be in the "
                         "YYYY-MM-DD format.",
                    type=str,
                    dest='end_date',
                    required=True)

parser.add_argument('--number-of-scenarios',
                    help='The number of scenarios to generate',
                    type=int,
                    dest='number_of_scenarios',
                    default=10)

parser.add_argument('--seed',
                    help='Seed for randomization',
                    type=int,
                    dest='seed')

parser.add_argument('--capacity',
                    help='This option specifies the capacity of the power '
                         'source. Scenarios will be truncated if they exceed '
                         'this value.',
                    type=float,
                    dest='capacity',
                    default=np.inf)

parser.add_argument('--source-name',
                    help='This option can be set to specify the name of the '
                         'power source being used to generate scenarios.'
                         'This only affects the names of plots and such.',
                    type=str,
                    dest='source_name',
                    default='Source')

parser.add_argument('--allow-multiprocessing',
                    help='Set this option if you wish to allow parallelization'
                         ' over the different days of scenario generation.',
                    action='store_true',
                    dest='allow_multiprocessing')

parser.add_argument('--alpha',
                    help='This option will determine the error values will be '
                         'determined to be outliers. This is a decimal value '
                         'between 0 and 1, where any errors outside of the '
                         'quantile range (alpha/2, 1-alpha/2) are thrown out.',
                    dest='alpha',
                    type=float)

def error_callback(exception):
    print("Process died with exception '{}'".format(exception),
          file=sys.stderr)


def prepare_data(args, source):
    """
    This function will compute errors and add an errors column as well
    as compute derivatives of forecasts if the arguments call for it.

    Args:
        args: The user-specified command-line arguments
        source (Source): The source of data
    """
    forecasts = source.get_column('forecasts')
    actuals = source.get_column('actuals')
    source.add_column('errors', actuals-forecasts)

    if args.consider_derivatives:
        print("1.5 Computing Derivatives")
        source.compute_derivatives('forecasts')

    # Here we remove any outliers in the original data.
    # We remove any that are outside of quantiles [alpha/2, 1-alpha/2]
    if args.alpha:
        lower, upper = source.get_quantiles('errors',
                                            [args.alpha/2, 1-args.alpha/2])
        source.data = source.window('errors', lower, upper).data

def parse_quantiles(quantile_string):
    return list(map(float, quantile_string.split(',')))


def compute_description(args, source, error_distribution):
    """
    This function will
    """
    # First we compute the error description
    if args.use_error_quantiles:
        if args.explicit_error_quantiles:
            quantiles = parse_quantiles(args.explicit_error_quantiles)
        else:
            quantile_width = args.error_quantile_size
            # We add to upper bound because arange's upper bound is exclusive
            quantiles = np.arange(0, 1 + quantile_width, quantile_width)
        description = descriptions.sample_quantile_description(
            quantiles, 'errors', source, error_distribution)
    else:
        description = descriptions.sample_bin_description(
            args.error_bin_size, 'errors', error_distribution)

    # For forecasts and derivatives, we do not use sample descriptions as, we
    # do not care to evaluate them. The state-value for forecasts and
    # derivatives is simply the interval, whereas for errors, we want to
    # sample.
    if args.consider_forecasts:
        if args.use_forecast_quantiles:
            if args.explicit_forecast_quantiles:
                quantiles = parse_quantiles(args.explicit_forecast_quantiles)
            else:
                quantile_width = args.forecast_quantile_size
                # We add to bound because arange's upper bound is exclusive
                quantiles = np.arange(0, 1 + quantile_width, quantile_width)
            description += descriptions.quantile_description(
                quantiles, 'forecasts', source)
        else:
            description += descriptions.bin_description(
                args.forecast_bin_size, 'forecasts')

    if args.consider_derivatives:
        if args.use_derivative_quantiles:
            if args.explicit_derivative_quantiles:
                quantiles = parse_quantiles(args.explicit_derivative_quantiles)
            else:
                quantile_width = args.derivative_quantile_size
                # We add to bound because arange's upper bound is exclusive
                quantiles = np.arange(0, 1 + quantile_width, quantile_width)
            description += descriptions.quantile_description(
                quantiles, 'forecasts_derivatives', source)
        else:
            description += descriptions.bin_description(
                args.derivative_bin_size, 'forecasts_derivatives')

    return description

def start_state(args, source, dt, distribution, description):
    """
    This function will produce a plausible start state from the source data
    and the fitted error distribution. It will sample the error from the
    distribution on the passed in start date and if the args specify, it will
    include the source's forecast and forecast derivative in the state.

    It will then construct a State object using the StateDescription object
    passed in.

    Args:
        args: The user-specified command-line arguments
        source (Source): The source of data in question
        dt (datetime-like): The datetime of the state
        distribution (BaseDistribution): The fitted distribution to the errors
        description (StateDescription): The description of what a state is
    Returns:
        State: A plausible start state
    """
    start_values = {}
    error = distribution.sample_one()
    start_values['errors'] = error
    if args.consider_forecasts:
        start_values['forecasts'] = source.all_data['forecasts'][dt]
    if args.consider_derivatives:
        start_values['forecasts_derivatives'] = \
            source.all_data['forecasts_derivatives'][dt]

    return description.to_state(**start_values)

def start_state_function(args, start_state, historic_states):
    """
    This function will return a function to compute the start state of the
    Markov Chain. It will sample from all the historic states which match
    the start state. Matching means they agree on the forecast state and the
    forecast derivative state if the args specify it. If forecasts and
    forecast derivatives are not to be considered, it just samples from all
    historic states.

    Args:
        args: The user-specified command-line arguments
        start_state (State): A plausible start state
        historic_states (list[State]): The list of historically observed
            states
    Returns:
        function: A function of no arguments which produces a state
    """
    criteria = []
    if args.consider_forecasts:
        criteria.append('forecasts')
    if args.consider_derivatives:
        criteria.append('forecasts_derivatives')

    matching_states = [state for state in historic_states
                       if start_state.match(state, criteria)]

    if len(matching_states) == 0:
        raise RuntimeError("No states match the first hour, cannot choose "
                           "a start state")

    def start_state():
        """
        This function will select a random choice from all the historic states
        which match the start state.

        Returns:
            State: A plausible start state for the Markov Chain
        """
        return choice(matching_states)

    return start_state


def evaluate_walk(walk):
    """
    This will evaluate every state in the walk to get the corresponding error
    value. This should return a list of 24 elements corresponding to every
    hour of the day.

    Args:
        walk (list[State]): A list of states with an 'errors' value
    Returns:
        list[float]: A list of 24 error values
    """
    errors = [state.evaluate()['errors'] for state in walk]
    return errors


def generate_scenarios(args, dt, source):
    """
    This function will run the Markov Chain scenario generation process.
    It proceeds in a few steps:
        - Computing a State Walk from historic data based on a description
          of a state specified by user arguments
        - Computing a transition matrix based with probabilities weighted by
          the frequency with which one state follows another historically
        - Computing a start state function which chooses a start state by
          sampling historic states which "match" the data for the date of
          scenario generation
        - Generating random walks from a Markov chain with the start state
          and transition matrix
        - Recovering Errors from these walks and then applying them to the
          forecast of the day and producing scenarios which are then truncated

    Args:
        args: The user-specified command-line arguments
        dt (datetime-like): A datetime for the date of scenario generation
        source (Source): The source of data
    """
    # We create a frame of all values up to and including dt
    rolling_window = source.rolling_window(dt)

    error_values = rolling_window.get_column('errors').values.tolist()
    error_distribution = UnivariateEmpiricalDistribution(error_values)

    state_description = compute_description(args, rolling_window,
                                            error_distribution)

    print("{}: 2. Computing Transition Matrix".format(dt))
    state_walk = rolling_window.get_state_walk(state_description)
    all_states = list(state_walk.values())
    states, matrix = markov_chains.matrix_from_walk(state_walk,
                                                    args.state_memory)

    if args.use_equal_probability:
        matrix = markov_chains.generate_equal_probability_matrix(len(states))

    # We generate a plausible start state based on the forecast for the day.
    plausible_start = start_state(args, rolling_window, dt,
                                  error_distribution, state_description)

    # Then our start state for each walk will be randomly chosen from historic
    # states which "match" the start state.
    start_function = start_state_function(args, plausible_start, all_states)

    chain = markov_chains.MarkovChain(states, matrix, start_function)

    scenarios = []

    actuals = list(source.get_day_of_data('actuals', dt).values)
    forecasts = list(source.get_day_of_data('forecasts', dt).values)

    failures = 0
    count = 0

    print("{}: 3. Generating Scenarios".format(dt))
    while count < args.number_of_scenarios:
        try:
            walk = list(chain.random_walk(24))
        except markov_chains.MarkovError:
            failures += 1
            if failures > 100:
                raise RuntimeError("The Markov Chain has failed to generate "
                                   "scenarios over 100 times because of states"
                                   " from which there is no transition. Your "
                                   "state space is probably to large and you "
                                   "do not have enough historic data.")
            continue
        name = "Scenario {}".format(count)
        errors = evaluate_walk(walk)
        power_values = [error + forecast
                        for error, forecast in zip(errors, forecasts)]
        scenarios.append(PowerScenario(name,
                                       {args.source_name: power_values},
                                       1/args.number_of_scenarios))
        count += 1

    for scenario in scenarios:
        scenario.truncate(args.source_name, 0, args.capacity)


    forecast_scenario = PowerScenario('forecasts',
                                      {args.source_name: forecasts}, 1)
    actual_scenario = PowerScenario('actuals', {args.source_name: actuals}, 1)

    scenario_set = SkeletonScenarioSet(scenarios, actual_scenario,
                                       forecast_scenario)

    title = dt.strftime('%Y-%m-%d')
    daily_dir = args.output_directory + os.sep + title
    os.mkdir(daily_dir)
    scenario_set.plot_scenarios(daily_dir, title + ' ')
    scenario_set.write_raw_scenarios(daily_dir, dt)


def main():
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)

    print("1. Reading Data")
    source = sources.source_from_csv(args.power_source, args.source_name,
                                     'wind')
    prepare_data(args, source)

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)
    os.mkdir(args.output_directory)

    date_range = pd.date_range(args.start_date, args.end_date, freq='D')

    # This parallelizes scenario generation over the days
    if args.allow_multiprocessing:
        pool = multiprocessing.Pool()
        for dt in date_range:
            pool.apply_async(generate_scenarios, args=(args, dt, source),
                             error_callback=error_callback)
        pool.close()
        pool.join()
    else:
        # If generating scenarios in serial
        for dt in date_range:
            try:
                generate_scenarios(args, dt, source)
            except:
                print("Failed to produce scenarios for day {}".format(dt))
                exc_type, exc, tb = sys.exc_info()
                print("{}: {}".format(exc_type, exc))
                traceback.print_traceback(tb)



if __name__ == '__main__':
    main()
