"""
compute_multi_interval2.py

This script will compute prediction intervals for sums over a set of
contiguous hours a certain lead time out from the current time using
a multivariate distribution.

Therefor this script fits univariate marginal distributions to the
errors+forecasts for each of the hours in the range. After that it fits
a Gaussian Copula to the joint distribution. Then it computes the CDF of
this distribution, which will give us in turn the prediction intervals.

The difference between this script and compute_multi_interval is, that
this script doesn`t include the bound to find a feasible mesh for
integration.
"""

import gosm.mesh as mesh
import argparse
import datetime
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

import prescient.distributions.copula as copula
import prescient.distributions.distributions as distributions
import gosm.segmenter as segmenter
import gosm.sources as sources
import time


parser = argparse.ArgumentParser()

parser.add_argument('--data-file',
                    help='The name of the file containing power generation '
                         'data.',
                    type=str,
                    dest='data_file')

parser.add_argument('--date-time-now',
                    help='The current datetime from which we will compute '
                         'prediction intervals --lead-time out. This must '
                         'be in the format YYYY-MM-DD:HH:00 using a 24-hour '
                         'clock.',
                    type=str,
                    dest='date_time_now')

parser.add_argument('--lead-time',
                    help='The number of hours in advance for which the '
                         'prediction interval should be computed.',
                    type=int,
                    dest='lead_time')

parser.add_argument('--hour-range-size',
                    help='The number of contiguous hours to sum up for '
                         'computing prediction intervals',
                    type=int,
                    dest='hour_range_size')

parser.add_argument('--alpha',
                    help='This specifies the size of the prediction interval. '
                         'If alpha = 0.05, then this will compute a 95% '
                         'prediction interval.',
                    type=float,
                    dest='alpha',
                    default=0.05)

parser.add_argument('--method',
                    help='This specifies which of the two methods for '
                         'computing prediction intervals for sums of '
                         'contiguous hours in the future. If set to 1, '
                         'this will use fit a vine copula with marginals for '
                         'each of the individual hours and then use sums.py '
                         'to compute the cdf. If set to 2, this will sum up '
                         'the data for contiguous hours and then use a '
                         'univariate distribution for the data.',
                    type=int,
                    dest='method',
                    default=1)

parser.add_argument('--number-of-days',
                    help='The number of days, for which you want to evaluate'
                         'the computed interval. Only required, if you run'
                         'evaluate_compute_intervals.py.',
                    type=int,
                    dest='amount_of_days')

parser.add_argument('--capacity',
                    help='The file containing the capacities of power'
                         'generation for different dates.',
                    type=str,
                    dest='cap')

parser.add_argument('--subdivisions',
                    help='The number of subdivisions on a single dimension.',
                    type=int,
                    dest='n')

parser.add_argument('--epsilon',
                    help='Criterion for breaking the mesh generation.',
                    type=float,
                    dest='eps')

parser.add_argument('--subintervals',
                    help='Number of subintervals for computing the cdf '
                         'inverse.',
                    type=int,
                    dest='intervals')

parser.add_argument('--separation',
                    help='If you want to separate the days by a threshold '
                         'for the forecast, type in "low" for taking just the '
                         'days with a forecast less than the threshold and '
                         '"high" for taking just the days with a forecast '
                         'higher than the threshold. You also have to type '
                         'in a specific threshold.',
                    type=str,
                    dest='sep',
                    default=None)


parser.add_argument('--threshold',
                    help='The threshold for each hour of interest '
                         'for separating the days.',
                    type=float,
                    dest='th',
                    default=None)

parser.add_argument('--window-size',
                    help='The size of the window which you want use for '
                         'segmenting the data.',
                    type = float,
                    dest='ws',
                    default=0.5)

def distr_fit(filename, dt, hour_range):
    """
    This function will take the data out of the provided file and fit a
    multivariate distribution within three steps:
    1. Fitting univariate normal distribution to each hour in hour_range.
    2. Fitting a copula to the data.
    3. Computing the multivariate distributin using the marginals and the
        copula.

    Args:
        filename(str): The name of the provided file containing the
            power generation data.
        dt(datetime-link): The date for which prediction intervals will be
            computed.
        hour_range(int): The number of hours for which the prediction
            interval is computed. This is equal to the dimension of
            the computed multivariate distribution.
    Returns:
        f(distribution): The fitted multivariate distribution.
    """

    # Reading in the source file
    source = sources.source_from_csv(filename, 'BPA Wind', 'wind')

    # Since the data doesn't originally have errors, we add a column in
    error_series = source.get_column('actuals') - source.get_column(
        'forecasts')
    source.add_column('errors', error_series)
    forecasts_series = source.get_column('forecasts')

    # This creates a Source object for each hour to consider
    first_hour = dt.hour
    hours_in_range = list(range(first_hour, first_hour + hour_range))
    hours_in_range = [hour % 24 for hour in hours_in_range]
    hourly_sources = source.split_source_at_hours(hours_in_range)

    day_ahead = dt - datetime.timedelta(hours=dt.hour)
    hourly_windows = {hour: source.rolling_window(day_ahead) for hour, source
                      in hourly_sources.items()}

    #This segments the data and fits a univariate distribution to the
    #segmented data.
    segmented_windows = {}
    marginals = {}
    forecasts_hour = {}
    for hour, window in hourly_windows.items():
        curr_dt = day_ahead + datetime.timedelta(hours=hour)
        segmented_windows[hour] = segment_data(window, curr_dt)
        forecasts_hour[hour] = forecasts_series[curr_dt]
        error_series = segmented_windows[hour].get_column('errors') \
                       + forecasts_hour[hour]
        distr = distributions.UnivariateNormalDistribution.fit(error_series)
        marginals[hour] = distr

    #To fit a copula to the data we need all data, not only the seperated one.
    #We have to transform it to [0,1]^n for the purposes of fitting a copula.
    hourly_df = source.get_column_at_hours('errors', hours_in_range)
    transformed_series = {}
    for hour in hours_in_range:
        hourly_df[hour] = hourly_df[hour] + forecasts_hour[hour]
        error_series = [marginals[hour].cdf(x) for x in hourly_df[hour]]
        transformed_series[hour] = error_series

    #First fitting a copula to the transformed data and then computing a
    #multivariate distribution using the copula and the marginals.
    fitted_copula = copula.GaussianCopula.fit(transformed_series,
                                              hours_in_range)
    f = copula.CopulaWithMarginals(fitted_copula, marginals, hours_in_range)

    return f

def segment_data(source, dt):
    """
    This function will segment the source of power by forecasts at the
    specified datetime.

    Args:
        source (RollingWindow): The source of power
        dt (datetime-like): The datetime with which to segment
    Returns:
        RollingWindow: The segmented data
    """

    args = parser.parse_args()

    window_size = args.ws
    criterion = segmenter.Criterion('forecasts', 'forecasts', 'window',
                                    window_size)
    segmented_source = source.segment_by_window(dt, criterion)
    return segmented_source

def gen_mesh(length, n, f, epsilon):
    """
    This function generates the mesh. The initiated mesh is upgraded as long
    as the integral of the added shell is bigger than epsilon.

    Args:
        length (float): The length of any side of the cube.
        n (int): The number of subdivisions on a single dimension, there
                will be n^d cells total where d is the dimension of the cube
        f (function): The function which is used to compute the integral over
                the added shell.
        epsilon (float): The number for the breaking criterion.
    """
    curr_mesh = mesh.CubeMesh([0, 0], length, n)
    while True:
        outer_shell = curr_mesh.outer_shell()
        outer_sum = outer_shell.integrate_with(f)
        if outer_sum < epsilon:
            break
        curr_mesh = curr_mesh.add_shell()
    return curr_mesh

def multi_cdf(m, bound, f):
    """
    This function computes the value of the cdf of the multivariate random
    variable (X1+X2+...Xn). For a given distribution it gives back the
    probability, if the sum of the random variables is lower or equal to a
    specific bound. Therefore it uses the feasible_mesh, so it includes
    the capacity for each dimension for computation.

    Args:
        m (mesh): The initial mesh.
        bound (float): The bound for the sum of the random variables.
        f (pdf): The distribution of the random variable (X1+...+Xn), which
           has to be a pdf.
    Returns:
        integral (float): The probability, if the sum of the random variable
            (X1+...+Xn) is lower or equal than the bound.
    """
    integral=0
    fsbmesh = mesh.Mesh([cell for cell in m.cells
                      if sum(cell.lower_left) <= bound])
    integral += fsbmesh.integrate_with(f)
    return integral

def cdf_inverse(m, alpha, capacity, f, subint):
    """
    This function computes the inverse value of a specific probability for
    a given distribution.

    Args:
        m (mesh): The initial mesh.
        alpha (float): The probability for which the inverse value is computed.
        capacity (float): The capacity of the power generation for each hour
            of interest.
        f (pdf): The distribution of the random variable (X1+...+Xn), which has
            to be a pdf.
            subint (int): The number of subintervalls, which are used to
                interpolate the cdf.
    Returns:
        inverse_bound (float): The computed inverse value of alpha.
    """
    x = np.linspace(0, 2*capacity, subint)
    y = []
    for i in x:
        yi = multi_cdf(m, i, f)
        j = int(np.argwhere(x==i))
        y.append(yi)
        if (j == 0) and (yi > alpha):
            inverse_alpha = 0
            break
        elif (j != 0):
            if y[j-1] <= alpha <= y[j]:
                lin = interp1d([y[j - 1], y[j]], [x[j - 1], x[j]])
                inverse_alpha = lin(alpha)

                break
    else:
        inverse_alpha = capacity

    return inverse_alpha

def compute_interval(m, alpha, capacity, f, subint):
    """
    This function computes finally the prediction intervals.

    Args:
        m (mesh): The initial mesh.
        alpha (float): The significance level.
        capacity (float): The capacity of the power generation for each hour of
            interes.
        f (pdf): The distribution of the random variable (X1+...+Xn), which has
            to be a pdf.
        subint (int): The number of subintervals, which are used to interpolate
           the cdf.
    Returns:
        lower (float): The lower bound of the prediction interval.
        upper (float): The upper bound of the prediction interval.
    """
    lower = cdf_inverse(m, alpha/2, capacity, f, subint)
    upper = cdf_inverse(m, 1-alpha/2, capacity, f,subint)
    return lower, upper

def f_time():
    """
    This function parses the time information from the .txt file. It computes
    the starting daytime for computing the intervals, called future_time.

    Args:

    Returns:
        future_time(str): The daytime computed of date_time_now and
            lead_time.
    """
    args = parser.parse_args()

    dt = datetime.datetime.strptime(args.date_time_now, '%Y-%m-%d:%H:%M')
    now = pd.Timestamp(dt)
    future_time = now + datetime.timedelta(hours=args.lead_time)

    return future_time

def read_cap(filename, future_time):
    """
    This function extracts from a file the capacity for the day the
    prediction interval is computed. For reading the file it uses the code
    from the script gosm.upper_bounds.py. You can get there more information
    about the format of the file.

    Args:
        filename (str): The name of the file containing the capacity data.
        t (datetime-link): The time for which you need the capacity.
    Returns:
        capacity (float): The specific capacity for the day the prediction
            interval is computed.
    """
    bounds = {}
    capacity = None
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not(line.strip()):
                continue

            if line.startswith('first_date'):
                continue

            start_date, last_date, capacity = line.split()
            start_date = datetime.datetime.strptime(start_date, '%m/%d/%y')
            last_date = datetime.datetime.strptime(last_date, '%m/%d/%y')\
                        + datetime.timedelta(hours=23)

            bounds[(start_date, last_date)] = float(capacity)

    for keys in bounds:
        if (keys[0] <= future_time) and (keys[1] >= future_time):
            capacity = bounds[keys]

    if capacity is None:
        raise ValueError('The capacity-file doesn`t include a capacity for'
                         'this date.')

    return capacity


def main(future_time):
    args = parser.parse_args()
    f = distr_fit(args.data_file, future_time, args.hour_range_size)
    capacity = read_cap(args.cap, future_time)
    m = gen_mesh(capacity, args.n, f.pdf, args.eps)
    lower, upper = compute_interval(m, args.alpha, capacity, f.pdf,
                                    args.intervals)
    return lower, upper, future_time

if __name__ == '__main__':
    t1=time.time()
    lower, upper, future_time = main(f_time())
    print("Predicted interval for {}: {}, {}".format(f_time(), lower, upper))
    t2=time.time()
    print('Zeit:',t2-t1)
