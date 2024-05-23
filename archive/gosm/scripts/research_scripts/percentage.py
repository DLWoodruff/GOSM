"""
percentage.py

This script modifies compute_uni_interval.py and uses relative errors
instead of absolute errors for fitting a distribution.
"""

import argparse
import datetime

import pandas as pd

import prescient.distributions.distributions as distributions
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

parser.add_argument('--capacity',
                    help='The file containing the capacities of power'
                         'generation for different dates.',
                    type=str,
                    dest='cap')

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
    contiguous_sums = contiguous_sums.shift(1-hour_range)
    hour_mask = contiguous_sums.index.hour == dt.hour
    historic_data = contiguous_sums[(contiguous_sums.index < dt) & hour_mask]
    dayahead_data = contiguous_sums[(contiguous_sums.index >= dt) & hour_mask]
    return sources.RollingWindow('BPA Wind', historic_data, 'wind',
                                 dayahead_data)


def segment_data(source, dt, window_size):
    """
    This function will segment the source of power by forecasts at the
    specified datetime.

    Args:
        source (RollingWindow): The source of power
        dt (datetime-like): The datetime with which to segment
    Returns:
        RollingWindow: The segmented data
    """

    """
    forecasts = source.dayahead_data['forecasts']

    if forecasts[dt] > 6000:
        window_size = window_size / 4
    elif (forecasts[dt] > 5000) and (forecasts[dt] <= 6000):
        window_size = window_size / 2
    elif (forecasts[dt] > 4000) and (forecasts[dt] <= 5000):
        window_size = window_size * 3/4
    elif (forecasts[dt] > 3000) and (forecasts[dt] <= 4000):
        window_size = window_size
    elif (forecasts[dt] > 2000) and (forecasts[dt] <= 3000):
        window_size = window_size * 3/4
    elif (forecasts[dt] > 1000) and (forecasts[dt] <= 2000):
        window_size = window_size / 2
    else:
        window_size = window_size / 4
    """


    criterion = segmenter.Criterion('forecasts', 'forecasts', 'window',
                                    window_size)
    segmented_source = source.segment_by_window(dt, criterion)
    return segmented_source

def compute_interval(source, dt, alpha, hour_range_size, capacity):
    """
    This function will fit a Empirical distribution to the error data of
    the source and then compute an interval on the errors based on the
    passed in alpha. It will compute
        [cdf_inverse(alpha/2), cdf_inverse(1-alpha/2)]
    and then it will add the forecast for the specified future datetime to
    get a prediction interval for the sum of forecasts. After that it will
    set the lower bound of the interval to 0, if it`s negative, and the
    upper bound to the capacity, if it`s hihger than the capacity.

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
    forecasts = source.get_column('forecasts')
    rel_err = errors / forecasts
    number_historic_hours = len(rel_err)
    distr = distributions.UnivariateEmpiricalDistribution.fit(rel_err)
    #distr.plot(plot_cdf=True, plot_pdf=False)
    dayahead_value = source.dayahead_data['forecasts'][dt]
    lower = distr.cdf_inverse(alpha / 2) * dayahead_value + dayahead_value
    upper = distr.cdf_inverse(1 - alpha / 2) * dayahead_value + dayahead_value

    if lower < 0:
        lower = 0
    if upper > hour_range_size * capacity:
        upper = hour_range_size * capacity

    return lower, upper, number_historic_hours

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



def main(future_time, window_size):
    args = parser.parse_args()
    source = read_data(args.data_file, future_time, args.hour_range_size)
    segmented_source = segment_data(source, future_time, window_size)
    capacity = float(read_cap(args.cap, future_time))
    lower, upper, number = compute_interval(segmented_source, future_time,
                                    args.alpha, args.hour_range_size,
                                    capacity)

    return lower, upper, future_time, number

if __name__ == '__main__':
    args = parser.parse_args()
    lower, upper, future_time = main(f_time(), args.ws)
    print("Predicted interval for {}: {}, {}".format(f_time(), lower, upper))
