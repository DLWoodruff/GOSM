import matplotlib
matplotlib.use('Agg')
import datetime
import sys
import os
import shutil
import argparse
import traceback
import multiprocessing
import time as t

from collections import OrderedDict
from random import choice

import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy.interpolate import interp1d

import prescient_gosm.sources.segmenter as segmenter
import prescient_gosm.sources as sources
import statdist.copula as cop
import statdist.base_distribution
import prescient_gosm.gosm_options as gosm_options
from statdist.distributions import UnivariateEmpiricalDistribution, MultiNormalDistribution
from prescient_gosm.structures.skeleton_scenario import PowerScenario, SkeletonScenarioSet
from prescient_gosm.markov_chains import markov_chains, states, descriptions

import cProfile


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

parser.add_argument('--number-scenarios',
                    help='The number of scenarios to generate',
                    type=int,
                    dest='number_scenarios',
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

parser.add_argument('--consider-hour',
                        help='Set this option if you want to include the '
                             'states hour in the state (e.g. the state '
                             'of the date 2013-01-01:04:00 includes the '
                             'hour 4.)',
                        dest='consider_hour')

def error_callback(exception):
    print("Process died with exception '{}'".format(exception),
          file=sys.stderr)


def common_date_range(srcs):
    """
    This function is utilised for multiple solar sources. Since each solar
        has different sunrise and sunset times we find a common sunrise
        and sunset times for every solar source by just getting the latest
        sunrise and earliest sunset so that every solar source has data for
        it. This only works for multiple solar sources only, this will
        have to be readjusted for mixed wind and solar sources.
    Args:
        srcs (list[RollingWindow]): The sources list.

    Returns: (datetime-index): the date range from the latest sunrise
        to earliest sunset.

    """

    latest_starting_date = srcs[0].daylight_dayahead.index[0]
    earliest_ending_date = srcs[0].daylight_dayahead.index[-1]

    for i in range(1, len(srcs)):
        if srcs[i].daylight_dayahead.index[0] > latest_starting_date:
            latest_starting_date = srcs[i].daylight_dayahead.index[0]
        if srcs[i].daylight_dayahead.index[-1] < earliest_ending_date:
            earliest_ending_date = srcs[i].daylight_dayahead.index[-1]

    return pd.date_range(latest_starting_date, earliest_ending_date, freq=srcs[0].time_step)


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

    # For including the hours in the state, we have to add them to the
    # state description. Because we are just interested in the hour itself,
    # we are using the function identity_description from descriptions.py.
    if args.consider_hour:
        description += descriptions.identity_description('hours')

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
    if args.consider_hour:
        start_values['hours'] = dt.hour

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

    # TODO: If hours are included in state, what does that mean for the start_state?
    # There are different options: Using only states with hour = 0 for finding
    # a start state or using states with a random hour or using all states.
    # For now, if no forecast is included in the states, every state is used
    # to find a start state.

    if args.consider_hour:
        criteria.append('hours')

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

"""
In this section we provide functions for scenario generation using 
2 dimensional copulas to get into the next state.
"""

def fit_marginal(source, day, window_size = 0.4):
    """
    This function fits a distribution to the historic errors at the given hour.
    Therefore it segments the data by window based on the days forecast.

    Args:
        source (RollingWindow): The source of data.
        day (datetime-like): Up to this day the error distribution is fitted.
        hour (int): The hour of the day for which you want to get the error
                    distribution. Because the hour represents a daytime, it has
                    to be inside the interval [0,23].
    Returns:
        BaseDistribution: The fitted error distribution.
    """
    #print(source.get_dayahead_value('forecasts', day))
    criterion = segmenter.Criterion('forecasts', 'forecasts', 'window',
                                    window_size = window_size)
    segmented_source = source.segment_by_window(day, criterion)
    errors = segmented_source.get_column('errors')
    distribution = UnivariateEmpiricalDistribution(errors)
    return distribution

def fit_copula(time_1, time_2, copula_class, source):
    """
    This function fits a copula of the given copula class to the errors of the
    given two hours. That means, that you get a 2 dimensional multivariate
    distribution where the errors of hour1 and the errors of hour2 are the 2
    dimensions. After getting the marginals for both dimensions, the errors of
    both dimensions are converted using the marginals into the copula space.
    At the end the copula is fitted to the converted errors.

    Args:
        hour1 (int): The hour which is used as the first dimension. It has to
                     be in the interval [0,23].
        hour2 (int): The hour which is used as the second dimension. It has to
                     be in the interval [0, 23].
        copula_class (CopulaBase): The class of the copula which is fitted to
                                   the errors.
        source(RollingWindow): The source of data.
        day (datetime-like): Up to this day the copula is fitted.
    Returns:
        copula (CopulaBase): The fitted copula.

    """


    rolling_window1 = source.mask_data(time_1)
    rolling_window2 = source.mask_data(time_2)

    marginal1 = fit_marginal(rolling_window1, time_1)
    marginal2 = fit_marginal(rolling_window2, time_2)

    errors1 = rolling_window1.get_column('errors')
    errors2 = rolling_window2.get_column('errors')

    converted_errors1 = [marginal1.cdf(error1) for error1 in errors1]
    converted_errors2 = [marginal2.cdf(error2) for error2 in errors2]
    converted_errors = [converted_errors1, converted_errors2]

    copula = copula_class.fit(converted_errors, dimkeys = [time_1, time_2])

    return copula, marginal1, marginal2


def find_start_state(source, day):
    """
    This function computes a start state, which is in this case just an error
    value for the "copula random walk". Therefore it fits an error distribution
    to the errors of hour 0 and samples a random value from it.

    Args:
        source (Source): The source of data.
        day (datetime-like): The day you want to get the start error for.
    Returns:
        start_error (float): The computed start state / start error.
    """
    distribution = fit_marginal(source, day)
    start_error = distribution.sample_one()
    return start_error

def step(time_1, time_2, copula_class, source, current_state, capacity):
    """
    This function computes the next step out of the current state in a
    "copula random walk". Therefore it fits a copula to the hour of the
    current state and the following hour. After that it uses the conditional
    cdf inverse conditioned on the current state and a uniformly distributed
    random number to create a sample from the conditional cdf conditioned on
    the current state. (State in the case of the "copula random walk" refers
    to a explicit error value.

    Args:
        hour1 (int): The hour which is used as the first dimension. It has to
                     be in the interval [0,23].
        hour2 (int): The hour which is used as the second dimension. It has to
                     be in the interval [0, 23].
        copula_class (CopulaBase): The class of the copula which is fitted to
                                   the errors.
        source(RollingWindow): The source of data.
        day (datetime-like): Up to this day the copula is fitted.
        current_state (float): The current state from which you want to compute
                               the next state. State means in the case of the
                               "copula random walk" a explicit error value.
    Returns:
        next_state (float): The computed next state. In the case of a
                            "copula random walk" it is just a explicit error
                            value.
    """
    # First the data has to be cleared. If there is only data available for
    # one hour of both hours of a specific day, this whole day is ignored for
    # this computation/step.
    dayahead_data = source.dayahead_data
    data = source.historic_data

    hours = [time_1.hour, time_2.hour]
    ignored_days = []
    for date in data.index:
        for hour in hours:
            if date.hour == hour:
                x = hours.index(hour)
                l = list(hours)
                l.pop(x)
                for h in l:
                    if date.replace(hour = h) not in data.index:
                        ignored_days.append(date)
                        data = data.drop(date)
    new_source = sources.RollingWindow('cleared source', data, 'wind',
                                       dayahead_data)

    copula, marginal1, marginal2 = fit_copula(time_1, time_2, copula_class,
                                              new_source)
    random_number = np.random.random()
    cond_xs = {time_1: marginal1.cdf(current_state)}
    next_state = copula.conditional_cdf_inverse(cond_xs, random_number,
                                                time_2, marginal2,
                                                capacity = capacity)
    return next_state

def get_power_values(source, errors):
    """
    This function creates a list of power values for a whole day for the
    purpose of creating scenarios out of it.
    It takes the dayahed forecasts for every hour and add the errors computed
    by a "copula random walk" (or any other random walk) to them.

    Args:
         source(RollingWindow): The source of data.
         errors (list[floats]): The computed errors from any random walk.
    Returns:
        power_values (list[floats]): The resulting power values which can be
                                     used for creating scenarios.
    """
    forecasts = source.dayahead_data['forecasts']
    power_values = [forecast + error for forecast, error in
                    zip(forecasts, errors)]
    return power_values

def copula_random_walk(start_state, copula_class, source, day, capacity):
    """
    This function computes a copula random walk. It takes a start state and
    uses a copula fitted to the error distributions of the current states hour
    and the next hour to compute the next state. At the end, all states are
    stored in a list.

    Args:
        start_state (float): The initial state for the random walk.
        ccopula_class (CopulaBase): The class of the copula which is fitted to
                                   the errors.
        source (RollingWindow): The source of data.
        day (datetime-like): Up to this day the copula is fitted.
    Returns:
        errors (list[floats]): The computed states, which are just error values
                               in this case.
    """
    current_state = start_state
    start = day
    end = day + pd.Timedelta(gosm_options.planning_period_length)
    freq = source.time_step
    time_points = pd.date_range(start, end, freq=freq)
    errors = [current_state]

    if source.source_type == "solar":
        dayahead_dates = source.daylight_dayahead.index
    else:
        dayahead_dates = source.dayahead_data.index

    for i in range(len(time_points) - 1):
        time_1 = time_points[i]
        time_2 = time_points[i + 1]
        if (time_1 in dayahead_dates) and (time_2 in dayahead_dates):
            next_state = step(time_1, time_2, copula_class, source, current_state,
                              capacity)
            errors.append(next_state)
            current_state = next_state
        else:
            errors.append(0)
            current_state = 0

    return errors

"""
In this section we provide functions for scenario generation using 
mutli-dimensional copulas to get into the next state. That means, these 
function handle the multiple source case.
"""


def make_grid(capacity, grid_size, pdf, cond_xs, marginal_cdf, error_tolerance):
    """
    This function creates a grid of the n-dimensional unit interval. Therefore
    it divides each dimension into a number of subintervals which is determined
    by the passed in grid_size. After that each cell of this grid is
    stored in a list. Each cell itself is a dictionary containing its bounds
    for each dimension and the integral of the probability density over this
    cell.

    Args:
        capacity (dict): A dictionary mapping the dimensions to the lower
            and upper bound of the whole grid in this dimension.
        grid_size (int): Number of subintervals per dimension.
        pdf (distribution-like): The multivariate density, NOT the
            conditional density!
        cond_xs (dict): A dictionary mapping the conditioned dimensions to
            their values.
        marginal_cdf (float): The cdf value of the multivariate density defined
            by the conditioned dimensions.
        error_tolerance (int): A value to signify how much room for error we
            would allow.

    Returns:
        cells (list[dict]): The computed cells. Each cell contains its bounds
            for every dimension and the integral of the conditional pdf over
            this cell.
        intervals (dict): A dictionary mapping each dimension to its
            subintervals.
    """

    dimensions = list(capacity.keys())
    #dimensions = [dim for dim in capacity]
    intervals = {}
    cell_borders = []
    for dim in dimensions:
            dim_points = np.linspace(capacity[dim][0], capacity[dim][1],
                                     grid_size + 1)
            dim_borders = []
            for i in range(len(dim_points)-1):
                dim_borders.append([dim_points[i], dim_points[i+1]])
            intervals[dim] = dim_borders
            cell_borders.append(dim_borders)
    product = cartesian_product(cell_borders)
    cells_border = []
    for borders in product:
        cell = {'bounds': {}}
        for i in range(len(dimensions)):
            cell['bounds'][dimensions[i]] = borders[i]
        cells_border.append(cell)
    cells = []
    for cell_border in cells_border:
        cell = cell_integral(cell_border, pdf, cond_xs, marginal_cdf, dimensions, error_tolerance)
        cells.append(cell)
    return cells, intervals


def cartesian_product(*args, repeat = 1):
    """
    This function computes the cartesian product of two sets represented by
    lists.
    Note: This sourcecode was copied from the documentation of "product"
    function of the python package "itertools". We did small changes in this
    code to serve our purposes.
    We use this function to get the bounds of each cell of the grid, which are
    represented by the cartesian product of all intervals for each dimension.

    Args:
        args (list[list]): The intervals of each dimension as a list.
        repeat (int): How often you want to repeat the cartesian product.
            We only use 1.

    Returns:
        l (list[list[float]): The list of the bounds for each cell.
    """

    pools = [tuple(item) for item in args[0]] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def cell_integral(cell, pdf, cond_xs, marginal_cdf, dimensions, error_tolerance):
    """
    This function takes the pdf and the conditioned dimensions to create a
    conditional pdf and integrates this over the given cell.

    Args:
        cell (dict): The cell as a dictionary mapping bounds to the bounds
            for each dimension.
        pdf (distribution-like): The multivariate density, NOT the
            conditional density!
        cond_xs (dict): A dictionary mapping the conditioned dimensions to
            their values.
        marginal_cdf (float): The cdf value of the multivariate density defined
            by the conditioned dimensions.
        error_tolerance (int): Value to increase the error tolerance
                of the integration process by powers of 10

    Returns:
        cell (dictionary): The same dictionary like before, but it includes
            now the integral of the conditional pdf over the cell.
    """

    bounds = []
    dimensions = []
    for dim in cell['bounds']:
        bounds.append((cell['bounds'][dim]))
        dimensions.append(dim)

    def func(*xs):
        args = {}
        for x, dim in zip(xs, dimensions):
            args[dim] = x
        return pdf.conditional_pdf(args, cond_xs, marginal_cdf)

    tol = error_tolerance

    cell['integral'] = integrate.nquad(func, bounds,
                        opts={'epsabs': (1.49e-08 * (10**tol)), 'epsrel': (1.49e-08 * (10**tol))})[0]
    return cell

def conditional_sample(intervals, cells):
    """
    This function samples a random point from a arbitrary (multidimensional)
    distribution. Therefor it uses the "grid"-method, which is described more
    detailed in the documentation for the "Copula Random Walk for multiple
    sources".

    Args:
        intervals (dict): A dictionary mapping each dimension to the
            subintervals in this dimension.
        cells (list[dict]): The cells of the grid containing the bounds of
            itself and the integral of the conditional pdf over itself.

    Returns:
        sample (dict): A dictionary mapping each dimension to its sampled
            value.
    """
    sample = {}
    for dim in intervals:
        sum = 0
        for cell in cells:
            sum += cell['integral']
        raw_values = [0]
        points = [0]
        for interval in intervals[dim]:
            cum_integral = 0
            for cell in cells:
                if cell['bounds'][dim] == interval:
                    cum_integral += cell['integral']
            raw_values.append(cum_integral)
            points.append(interval[1])
        cum_raw_values = np.cumsum(raw_values)
        values = list(cum_raw_values / cum_raw_values[-1])
        rn = np.random.uniform(0, 1, 1)
        for j in range(len(values) - 1):
            if values[j] <= rn <= values[j+1]:
                lin = interp1d([values[j], values[j+1]],
                               [points[j], points[j+1]])
                inverse_rn = float(lin(rn))
        for interval in intervals[dim]:
            if interval[0] <= inverse_rn <= interval[1]:
                cells = [cell for cell in cells
                         if cell['bounds'][dim] == interval]
                break
        sample[dim] = inverse_rn

    return sample

def fit_copula_ms(times, copula_class, srcs, use_copula=True):
    """
    This function fits a copula to multiple sources. It includes also an
    option to use a multivariate Gaussian distribution instead of a copula
    to speed up the computation process for testing purposes.

    Args:
        times (list[datetime-like]): The times for the next step. They are used
            to define the dimensions. Each source defines one dimension at
            each time. That means for 2 sources you get 4 dimensions, for
            3 sources you get 6 dimensions and so on.
        copula_class (CopulaBase-like): The class of copula you want to use.
            For now, it was only tested with the Gaussian Copula.
        srcs (list[RollingWindow]): The sources list.
        use_copula (boolean): If true, the copula is used for the computation.
            If false, the multivariate Gaussian distribution is used. (This
            is only for testing purposes, because its a lot faster)

    Returns:
        copula (distribution-like): The fitted copula.
        marginals (dictionary): A dictionary mapping the dimensions of the
            copula to their respective marginal distribution.

    """
    marginals = {}
    all_converted_errors = {}
    for time in times:
        for source in srcs:

            rolling_window = source.mask_data(time)
            marginal = fit_marginal(rolling_window, time)

            errors = rolling_window.get_column('errors')
            converted_errors = [marginal.cdf(error) for error in errors]
            dimension = source.name + ' ' + str(time)
            all_converted_errors[dimension] = converted_errors
            marginals[dimension] = marginal
    if use_copula:
        copula = copula_class.fit(all_converted_errors,
                                  list(all_converted_errors.keys()))
    else:
        copula = MultiNormalDistribution.fit(all_converted_errors,
                                             list(all_converted_errors.keys()))
    return copula, marginals

def step_ms(times, copula_class, srcs, current_state, error_tolerance):
    """
    This function makes a step in the copula random walk for multiple
    sources. Therefor it fits a copula to each source at each time, creates
    a grid with the integral of the copula density conditioned on the
    current state over each cell, and samples from this.

    Args:
        times (list[datetime-like]): The times for the next step. They are used
            to define the dimensions. Each source defines one dimension at
            each time. That means for 2 sources you get 4 dimensions, for
            3 sources you get 6 dimensions and so on.
        copula_class (CopulaBase-like): The class of copula you want to use.
            For now, it was only tested with the Gaussian Copula.
        srcs (list[RollingWindow]): The sources list.
        current_state (dict): A dictionary mapping the dimensions of the start
            of the step (the "current" dimension) to their values.
        error_tolerance (int): A value to signify how much room for error we
            would allow.

    Returns:
        next_state (dict): A dictionary mapping the dimensions of the end
            of the step (the "next" dimension) to their values.
    """
    new_srcs = []
    for source in srcs:
        dayahead_data = source.dayahead_data
        historic_data = source.historic_data

        ignored_days = []
        for date in historic_data.index:
            for time in times:
                if date.time() == time.time():
                    x = times.index(time)
                    l = list(times)
                    l.pop(x)
                    for d in l:
                        replaced = date.replace(hour=d.hour,
                                                minute=d.minute,
                                                second=d.second)
                        if replaced not in historic_data.index:
                            ignored_days.append(date)
                            historic_data = historic_data.drop(date)

        new_window = sources.RollingWindow(source.name, historic_data,
                                           source.source_type,
                                           dayahead_data)
        new_source = sources.ExtendedWindow(new_window, source.criteria, source.capacity,
                                            avg_sunrise_sunset=source.avg_sunrise_sunset)

        # Since this is a new window must recalculate daylight data for solar sources
        if source.source_type == "solar":
            new_source.daylight_data()

        new_srcs.append(new_source)

    copula, marginals = fit_copula_ms(times, copula_class, new_srcs,
                                      use_copula = True)

    conv_current_state = {}
    for dim in current_state:
        error = current_state[dim]
        conv_error = marginals[dim].cdf(error)
        conv_current_state[dim] = conv_error

    dimensions = [source.name + ' ' + str(time) for time in times for source in srcs]

    cap = {dim: [0,1] for dim in dimensions if not dim in conv_current_state}

    #Times are for debugging purposes when testing three or more multiple sources
    #t1 = t.time()

    bounds = [(0,1)] * len(current_state)
    conditional_marginal = copula.marginal(conv_current_state, bounds, error_tolerance=error_tolerance)

    #t2 = t.time()
    #print('TIME:', (t2 - t1)/60)
    #t5 = t.time()

    cells, intervals = make_grid(cap, 2, copula, conv_current_state,
                                 conditional_marginal, error_tolerance)

    #t6 = t.time()
    #print('Time', (t6 - t5)/60)

    sample = conditional_sample(intervals, cells)

    next_state = {}
    for dim in sample:
        value = marginals[dim].cdf_inverse(sample[dim])
        next_state[dim] = value
    return next_state

def copula_random_walk_ms(start_state, copula_class, srcs, start_time, error_tolerance):
    """
    This function does a random walk for multiple sources using copuls to
    get from one state into the other. In the end it will return a list of
    all states for this random walk. Its length depends on the planning
    period length and the time step the user chooses in the option file.

    Args:
        start_state (dict): A dictionary mapping each dimension of the start
            to its error values.
        copula_class (CopulaBase-like): The class of copula you want to use.
            For now, it was only tested with the Gaussian Copula.
        srcs (list[RollingWindow]): The sources list.
        start_time (datetime-like): The start time of the random walk.
        error_tolerance (int): A value to signify how much room for error we
            would allow.

    Returns:
        errors (list[dict]): The list of the states. Each state is a dictionary
            mapping its dimensions to the respective error values.
    """
    current_state = start_state
    start = start_time
    end = start_time + pd.Timedelta(gosm_options.planning_period_length)
    freq = list(srcs)[0].time_step
    time_points = pd.date_range(start, end, freq=freq)
    errors = [start_state]

    # Have the names for the sources in order to create keys for the dictionaries in errors
    names = [src.name for src in srcs]


    # Multiple sources for solar
    if srcs[0].source_type == "solar":
        daytime_dates = common_date_range(srcs)
    else:
        daytime_dates = srcs[0].dayahead_data.index

    for i in range(len(time_points) - 1):
        times = [time_points[i], time_points[i+1]]
        # Check if the time is in the dates which we are considering for scenarios. See solar sources
        if (times[0] in daytime_dates) and (times[1] in daytime_dates):
            next_state = step_ms(times, copula_class, srcs, current_state, error_tolerance)

        # For solar sources we must create a new start state for the first hour of sunlight
        elif (times[0] not in daytime_dates) and (times[1] in daytime_dates):
            next_state = solar_start_state_ms(time_points[i + 1], srcs)

        else:
            # ex: k is 'Solar_NP15' and times[1] is '2014-10-03 01:00:00'
            # would result in {'Wind_NP15 2014-10-03 01:00:00': 0}
            date = times[1].strftime("%Y-%m-%d %H:%M:%S")
            next_state = {(k + ' ' + date):0 for k in names}

        errors.append(next_state)
        current_state = next_state

    return errors

def start_state_ms(time, srcs):
    """
    This function generates a start state for the random walk for multiple
    sources. Therefore it fits a univariate empirical distribution to each
    source at the regarding time and samples from these.

    Args:
        time (datetime-like): At this time the error distribution for each
            source is computed.
        src (dict[RollingWindow]): Dictionary of rolling widows of sources.

    Returns:
        starting_state (dict): A dictionary mapping the dimensions to their error
            values.
    """
    starting_state = {}
    for source in srcs.values():
        distribution = fit_marginal(source, time)
        start_error = distribution.sample_one()
        starting_state[source.name + ' ' + str(time)] = start_error

    return starting_state


def solar_start_state_ms(time, srcs):
    """
    This function generates a start state for the random walk for multiple
    solar sources because we have to find a start state for the first
    daylight time and not during the nighttime. It is a condensed version
    of the other start_state_ms and fit_marginal functions. This function
    still needs to be worked on and fixed since it only provides a
    temporary fix for creating a solar start state.

    Args:
        time (datetime-like): At this time the error distribution for each
            source is computed.
        src (dict[RollingWindow]): Dictionary of rolling widows of solar sources.

    Returns:
        starting_state (dict): A dictionary mapping the dimensions to their error
            values.
    """

    starting_state = {}
    for source in srcs:

        rolling_window = source.mask_data(time)
        distribution = fit_marginal(rolling_window, time)
        start_error = distribution.sample_one()

        starting_state[source.name + ' ' + str(time)] = start_error
    return starting_state

def get_power_values_ms(walk, srcs):
    """
    This function adds the forecast to the predicted errors of the random
    walk to get power values, which are processed in the PowerScenarios.

    Args:
        walk (list[dict]): The computed states of a random walk. Each state
            is a dictionary mapping its dimensions to their error values.
        srcs (list[RollingWindow]): The list of sources.

    Returns:
        power_values (dict): A dictionary mapping each source to the list of
            their predicted power values.
    """
    power_values = {}
    for source in srcs:
        forecasts = source.dayahead_data['forecasts']
        errors = []
        for step in walk:
            for dim in step:
                if dim.split(' ')[0] == source.name:
                    errors.append(step[dim])
        power_values[source.name] = [forecast + error for forecast, error
                                     in zip(forecasts, errors)]
    return power_values




def multiple_sources(args, dt, sources_list):
    """
    This function will run the Markov Chain scenario generation process for
    multiple sources. It proceeds in a few steps:
        - Computing a State Walk from historic data based on a description
          of a state specified by user arguments
        - Computing a start state
        - Generating random walks from a Markov chain with the start state
          and copulas conditioned on the state before.
        - Recovering Errors from these walks and then applying them to the
          forecast of the day and producing scenarios which are then truncated

    Args:
        args: The user-specified command-line arguments
        dt (datetime-like): A datetime for the date of scenario generation
        sources_list (list[sources]): The source of data
    """

    rolling_windows = {}
    for source in sources_list:
        key = source.name

        if source.source_type == "solar":
            rolling_window = source.daylight_data(dt)
        else:
            rolling_window = source.rolling_window(dt)

        rolling_windows[key] = rolling_window

    # Adding some noise for testing purposes. (If theres real data available
    # for one source)
    """
    for i, rolling_window in enumerate(rolling_windows.values()):
        if i > 0:
            raw = rolling_window.get_column('errors')
            errors = [raw[j] + int(np.random.randint(-500, 500, 1)) for j in
                      range(len(raw))]
            rolling_window.data['errors'] = errors
    """



    if args.copula_random_walk:
        scenarios = []
        failures = 0
        count = 0
        while count < args.number_scenarios:
            try:
                start = start_state_ms(dt, rolling_windows)
                walk = copula_random_walk_ms(start, cop.GaussianCopula,
                                             list(rolling_windows.values()),
                                             dt, args.error_tolerance)

            except markov_chains.MarkovError:
                failures += 1
                if failures > 100:
                    raise RuntimeError(
                        "The Markov Chain has failed to generate "
                        "scenarios over 100 times because of states"
                        " from which there is no transition. Your "
                        "state space is probably to large and you "
                        "do not have enough historic data.")
                continue
            name = "Scenario {}".format(count)
            power_values = get_power_values_ms(walk,
                                               list(rolling_windows.values()))
            power_sc = PowerScenario(name,
                                     {source.name: power_values[source.name]
                                      for source in rolling_windows.values()},
                                     1 / args.number_scenarios, begin=dt,
                                     planning_period_length=gosm_options.planning_period_length)
            scenarios.append(power_sc)
            count += 1
    else:
        raise ValueError('You need to use the option "copula_random_walk" '
                         'for multiple sources.')
    #print(scenarios)

    for scenario in scenarios:
        for source in rolling_windows.values():
            scenario.truncate(source.name, 0, source.capacity)

    return scenarios


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

    rolling_window = source.rolling_window(dt, time_step=source.time_step,
                                           planning_period_length=gosm_options.planning_period_length)

    if source.source_type == "solar":
        rolling_window.daylight_data()

    capacity = rolling_window.capacity

    if args.copula_random_walk:
        scenarios = []
        failures = 0
        count = 0
        while count < args.number_scenarios:
            try:
                start= find_start_state(rolling_window, dt)
                walk = copula_random_walk(start, cop.GaussianCopula,
                                          rolling_window, dt, capacity)

            except markov_chains.MarkovError:
                failures += 1
                if failures > 100:
                    raise RuntimeError(
                        "The Markov Chain has failed to generate "
                        "scenarios over 100 times because of states"
                        " from which there is no transition. Your "
                        "state space is probably too large and you "
                        "do not have enough historic data.")
                continue
            name = "Scenario {}".format(count)
            power_values = get_power_values(rolling_window, walk)
            scenarios.append(PowerScenario(name,
                                           {source.name: power_values},
                                           1 / args.number_scenarios,
                                           begin=dt,
                                           planning_period_length=gosm_options.planning_period_length))
            count += 1

        for scenario in scenarios:
            scenario.truncate(source.name, 0, source.capacity(dt))

        return scenarios

    else:
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
        while count < args.number_scenarios:
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
                                           {source.name: power_values},
                                           1/args.number_scenarios))
            count += 1

        for scenario in scenarios:
            scenario.truncate(source.name, 0, source.capacity(dt))

        return scenarios

def generate_scenarios_profile(args, dt, source):
    cProfile.runctx('generate_scenarios(args, dt, source)', globals(), locals(), sort='cumtime')
    return generate_scenarios(args, dt, source)

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
