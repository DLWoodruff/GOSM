"""
This program evaluates the prediction intervals of generated power within the
next couple of hours computed by compute_uni_interval.py. For that there are
used two different methods.

At first this script checks, if the predicted interval for a specific day
contains the actual value for that day. By repeating this step for
other days and counting the points in- and outside the intervals, you can
compare this proportions with the alpha value of the computed confidence
intervals.

The second method deals with the width of the predicted intervals. This script
computes the average width of the computed intervals.
"""

import compute_multi_interval2_4
import argparse
import pandas as pd
import gosm.sources as sources
import time
import multiprocessing as mp

parser = argparse.ArgumentParser()

parser.add_argument('--data-file',
                    help='The name of the file containing power generation '
                         'data.',
                    type=str,
                    dest='data_file')

parser.add_argument('--date-time-now',
                    help='The current date time from which we will compute '
                         'and evaluate the prediction intervals --lead-time '
                         'out. This must be in the format YYYY-MM-DD:HH00 '
                         'using a 24 hour clock.',
                    type=str,
                    dest='date_time_now')

parser.add_argument('--lead-time',
                    help='The number of hours in advance for which the '
                         'prediction interval should be computed.',
                    type=int,
                    dest='lead_time')

parser.add_argument('--hour-range-size',
                    help='The number of contiguous hours to sum up for '
                         'computing prediction intervals.',
                    type=int,
                    dest='hour_range_size')

parser.add_argument('--alpha',
                    help='The significance level of the confidence interval '
                         'you want to compute.',
                    type=float,
                    dest='alpha',
                    default=0.05)

parser.add_argument('--method',
                    help='This specifies which of the two methods for '
                         'computing prediction intervals you want to use.'
                         'If set to 1, this will use fit a vine copula with '
                         'marginals for each of the individual hours and then '
                         'use sums.py to compute the cdf. If set to 2, this '
                         'will sum up the data for contiguous hours and then '
                         'use a univariate distribution for the data.',
                    type=int,
                    dest='method',
                    default=2)

parser.add_argument('--number-of-days',
                    help='The number of days for which you want to evaluate '
                         'the computed intervals. Only required if you run '
                         '"evaluate_compute_intervals.py".',
                    type=int,
                    dest='number_of_days',
                    default=10)

parser.add_argument('--capacity',
                    help='The file that includes the capacities for different'
                         'dates.',
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

def set_list(file, future_time, hour_range_size, threshold, separation):
    """
    This function reads the given data file with the read_data function in
    compute_intervals.py and converts it into a list, which can be handled
    by the counting function below.

    Args:
        file(str): The filename of the csv file containing the actuals and
            forecasts informations.
        future_time(str): The future time, from which you want to compute the
            intervals.
        hour_range_size(int): The hour range size which is the number of
            contiguous hours to sum up for computing prediction intervals.

    Returns:
            daylist(list): A list containing the contiguous date times and
                belonging actuals starting at dt.
    """
    data = read_data(file, future_time, hour_range_size)
    first_day = data.historic_data.index[0]
    actual_series = data.dayahead_data['actuals']
    forecast_series = data.dayahead_data['forecasts']

    th = threshold * hour_range_size

    if separation == 'low':
        daylist = forecast_series[forecast_series < th].index.tolist()
    elif separation == 'high':
        daylist = forecast_series[forecast_series > th].index.tolist()
    elif separation == None:
        daylist = forecast_series.index.tolist()

    return first_day, daylist, actual_series

def count(number, daylist_in, series, num_processes = 1):
    """
    This function uses a list with different date times and computes for
    every date in that list a prediction interval. After that it checks,
    if the belonging actual value is in- or outside the interval and counts
    the number of these points.

    Args:
        amount(int): The amount of date times you want to use for computing
            and evaluating an interval.
        list(list): A list containing date times which can be used for
            computing and evaluating intervals.
        series(pd series): A (pandas) series containing the date times of the
            list above and the belonging actuals.
    Returns:
        pl(float): The proportion of points on the left side of the intervals
            in percent.
        pr(float): The proportion of points on the right side of the intervals
            in percent.
        pin(float): The proportion of points inside the intervals in percent.
    """

    a = 0
    b = 0
    inside = 0
    number = int(number)
    sum_width = 0

    # Adapting length of daylist and number of analysis days to each other.

    if len(daylist_in) > number:
        daylist = daylist_in[0:number]
    else:
        daylist = daylist_in

    length = len(daylist)

    # Computing the interval for each day in daylist via multiprocessing.

    with mp.Pool(num_processes) as pool:
        # dayruns will have a list of result objects; one for each call to main
        dayruns = [pool.apply_async(compute_multi_interval2_4.main,
                                    (future_time,)) for future_time in daylist]
        # results will get a list of tuples, each with returned values
        results = [result.get() for result in dayruns]

    # Counting the points in- and outside the computed intervals.

    for day in results:
        lo, up, dt = day
        act = series[dt]
        sum_width += (up-lo)
        if act < lo:
            a += 1
        elif act > up:
            b += 1
        else:
            inside += 1

    # Calculating the proportion of in- and outside points and the avg. width.

    average_width = sum_width/length
    pl = round((a / length) * 100, 2)
    pr = round((b / length) * 100, 2)
    pin = round((inside / length) * 100, 2)

    return a, b, inside, pl, pr, pin, average_width, length

def main():
    args=parser.parse_args()

    fd, daylist, series = set_list(args.data_file,
                                   compute_multi_interval2_4.f_time(),
                                   args.hour_range_size, args.th, args.sep)
    a, b, inside, pl, pr, pin, aw, length = count(args.number_of_days,
                                                  daylist, series)

    with open('../../research_results/evaluation_multi2_' + args.date_time_now
                      + '_' + str(args.hour_range_size) + '.txt', 'w') as output:

        print('Hour-range-size:', args.hour_range_size, file=output)
        print('Alpha:', args.alpha * 100, '%', file = output)
        print('First training date:', fd, file = output)
        print('First analysis date:', daylist[0], file = output)
        print('Number of analysis days:', length, file = output)
        print('Out left (number):', a, 'Out right (number):', b, file = output)
        print('Out left (proportion):', pl, '%. Out right (proportion):', pr,
              '%', file = output)
        print('Inside(number):', inside, 'Inside (proportion):', pin, '%',
              file = output)
        print('Average width:', aw, file = output)
        print('Epsilon:', args.eps, 'Subintervals:', args.intervals,
              'Subdivisions:', args.n, file = output)
        print('Threshold per hour:', args.th, file=output)

    print('Hour-range-size:', args.hour_range_size)
    print('Alpha:', args.alpha * 100, '%')
    print('First training date:', fd)
    print('First analysis date:', daylist[0])
    print('Number of analysis days:', length)
    print('Out left (number):', a, 'Out right (number):', b)
    print('Out left (proportion):', pl, '%. Out right (proportion):', pr,
          '%')
    print('Inside(number):', inside, 'Inside (proportion):', pin, '%')
    print('Average width:', aw)
    print('Epsilon:', args.eps, 'Subintervals:', args.intervals,
          'Subdivisions:', args.n)



if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('Time:', t2 - t1)