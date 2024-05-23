"""
compute_intervals.py

This script will compute prediction intervals for sums over a set of
contiguous hours a certain lead time out from the current time.

We will have two separate methods for performing this procedure, the first of
which will fit marginal distributions to forecast+error for each of the hours
in the range and then a vine copula to model the joint distribution. Then it
will use sums.py to compute a CDF for the sum of variables which will in turn
give us a prediction interval.

The second method will take the sum of the contiguous hour ranges that are the
lead time out from the current time and fit a univariate distribution to this
data, using the CDF of this distribution to find the appropriate prediction
interval.
"""

import argparse
import datetime

import pandas as pd

import gosm.distributions.vine as vine
import gosm.distributions.copula as copula
import gosm.distributions.distributions as distributions
import gosm.segmenter as segmenter
import gosm.sources as sources

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
                    dest='number_of_days')

def read_data(filename, dt, hour_range):
    """
    This function will read in data from a source file and perform some basic
    processing of the data. Specifically, it expects a csv file with the first
    column being datetimes, and then two more columns with the headers
    'actuals' and 'forecasts'. It will compute errors from this data, and then
    it will compute a new DataFrame with with rolling sums of the data.

    From this it will create a RollingWindow object with historical data
    being all those dates which match the passed in dt's hour occuring before
    dt and the dayahead data being the row of data corresponding to dt.

    Args:
        filename (str): The name of the csv file containing forecast and actual
            information
        dt (datetime-like): The date for which prediction intervals will be
            computed
        hour_range (int): The number of contiguous hours to sum
    Returns:
        RollingWindow: The separation of data into historic and dayahead data
    """
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df['errors'] = df['actuals'] - df['forecasts']
    contiguous_sums = df.rolling(hour_range).sum()
    hour_mask = contiguous_sums.index.hour == dt.hour
    historic_data = contiguous_sums[(contiguous_sums.index < dt) & hour_mask]
    dayahead_data = contiguous_sums[(contiguous_sums.index >= dt) & hour_mask]
    return sources.RollingWindow('BPA Wind', historic_data, 'wind',
                                 dayahead_data)


def method_1(filename, dt, hour_range):
    # Reading in the source file
    source = sources.source_from_csv(filename, 'BPA Wind', 'wind')

    # Since the data doesn't originally have errors, we add a column in
    error_series = source.get_column('actuals') - source.get_column(
        'forecasts')
    source.add_column('errors', error_series)

    # This creates a Source object for each hour to consider
    first_hour = dt.hour
    hours_in_range = list(range(first_hour, first_hour + hour_range))
    hourly_sources = source.split_source_at_hours(hours_in_range)

    day_ahead = dt - datetime.timedelta(hours=dt.hour)
    hourly_windows = {hour: source.rolling_window(day_ahead) for hour, source
                      in hourly_sources.items()}

    segmented_windows = {}
    marginals = {}

    # Here we segment the data and create a marginal distribution for each
    # hour of interest.
    for hour, window in hourly_windows.items():
        curr_dt = day_ahead + datetime.timedelta(hours=hour)
        segmented_windows[hour] = segment_data(window, curr_dt)
        error_series = segmented_windows[hour].get_column('errors')
        distr = distributions.UnivariateEmpiricalDistribution.fit(error_series)
        marginals[hour] = distr

    # We draw from source now as we want to have all the data and not the
    # segmented data.
    hourly_df = source.get_column_at_hours('errors', hours_in_range)

    # We transform the data to [0, 1]^n for the purposes of fitting a copula
    transformed_series = {}
    for hour in hours_in_range:
        error_series = [marginals[hour].cdf(x) for x in hourly_df[hour]]
        transformed_series[hour] = error_series

    dimension = len(hours_in_range)

    # TODO make pairwise copula an option
    pair_copula_strings = [['gaussian-copula'] * dimension] * dimension

    # TODO make vine copula an option
    vine_copula = vine.CVineCopula(dimkeys=hours_in_range,
                                   input_data=transformed_series,
                                   pair_copulae_strings=pair_copula_strings)

    vine_copula.pdf({hour: 0 for hour in hours_in_range})

    # This will be our multivariate distribution of each hour of errors
    multi_distr = copula.CopulaWithMarginals(vine_copula, marginals,
                                             hours_in_range)

    multi_distr.pdf({hour: 0 for hour in hours_in_range})



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
    window_size = 0.5
    criterion = segmenter.Criterion('forecasts', 'forecasts', 'window',
                                    window_size)
    segmented_source = source.segment_by_window(dt, criterion)
    return segmented_source


def compute_interval(source: object, dt: object, alpha: object) -> object:
    """
    This function will fit a Empirical distribution to the error data of
    the source and then compute an interval on the errors based on the
    passed in alpha. It will compute
        [cdf_inverse(alpha/2), cdf_inverse(1-alpha/2)]
    and then it will add the forecast for the specified future datetime to
    get a prediction interval for the sum of forecasts.

    TODO: Add ability to specify an arbitrary distribution

    Args:
        source (RollingWindow): The source of power with an errors column
        dt (datetime-like): The future datetime to compute a prediction
            interval for
        alpha (float): A parameter specifying the confidence level of the
            prediction interval
    Returns:
        tuple: An ordered pair with the first element being the lower limit
            and the second being the upper limit
    """
    errors = source.get_column('errors')
    distr = distributions.UnivariateEmpiricalDistribution.fit(errors)
    dayahead_value = source.dayahead_data['forecasts'][dt]
    lower = distr.cdf_inverse(alpha / 2) + dayahead_value
    upper = distr.cdf_inverse(1 - alpha / 2) + dayahead_value

    return lower, upper

def time():
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


def main(future_time):
    args = parser.parse_args()

    if args.method == 1:
        method_1(args.data_file, future_time, args.hour_range_size)
        raise RuntimeError("The remaining code is not implemented yet.")
    elif args.method == 2:
        source = read_data(args.data_file, future_time, args.hour_range_size)

        segmented_source = segment_data(source, future_time)
        lower, upper = compute_interval(segmented_source, future_time,
                                        args.alpha)

    return lower, upper


if __name__ == '__main__':
    lower, upper = main(time())
    print("Predicted interval for {}: {}, {}".format(time(), lower, upper))
