"""
This program evaluates the prediction intervals of produced power within the
next couple of hours computed by compute_intervals.py. For that there are used
two different methods.

At first this script checks, if the predicted interval for a specific day
contains the actual value for that day. By repeating this step for
other days and counting the points in- and outside the intervals, you can
compare this proportions with the alpha value of the computed confidence
intervals.

The second method deals with the width of the predicted intervals. This script
computes the average width of the computed intervals.
"""

import compute_intervals
import pandas as pd
import argparse
import datetime

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

def set_list(file, future_time, hour_range_size):
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
    data = compute_intervals.read_data(file, future_time, hour_range_size)
    first_day = data.historic_data.index[0]
    series = data.dayahead_data['actuals']
    daylist = series.index.tolist()

    return first_day, daylist, series

def count(number, list, series):
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

    if number > len(list):
        number = len(list)

    for i in range(0, number):
        future_time = list[i]
        lo, up = compute_intervals.main(future_time)
        act = series[future_time]
        sum_width += (up-lo)
        if act < lo:
            a += 1
        elif act > up:
            b += 1
        else:
            inside += 1

    average_width = sum_width/number
    pl = round((a / len(range(0, number))) * 100, 2)
    pr = round((b / len(range(0, number))) * 100, 2)
    pin = round((inside / len(range(0, number))) * 100, 2)

    return a, b, inside, pl, pr, pin, average_width, number

def main():
    args=parser.parse_args()

    fd, daylist, series = set_list(args.data_file, compute_intervals.time(),
                                   args.hour_range_size)
    a, b, inside, pl, pr, pin, aw, number = count(args.number_of_days,
                                                  daylist, series)

    print('Alpha:', args.alpha*100, '%')
    print('First training date:', fd)
    print('First analysis date:', daylist[0])
    print('Number of analysis days:', number)
    print('Out left:', pl, '%. Out right:', pr, '%')
    print('Inside:', pin, '%')
    print('Average width:', aw)

if __name__ == '__main__':
    main()