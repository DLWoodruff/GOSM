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

import compute_uni_interval
import argparse
import time

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
    data = compute_uni_interval.read_data(file, future_time, hour_range_size)
    first_day = data.historic_data.index[0]
    series = data.dayahead_data['actuals']
    daylist = series.index.tolist()

    return first_day, daylist, series

def count(number, list, series, alpha, dimension):
    """
    This function uses a list with different date times and computes for
    every date in that list a prediction interval. After that it checks,
    if the belonging actual value is in- or outside the interval and counts
    the number of these points.
    This function calculates also the average width of the intervals and the
    Ellipsoidal skill score.

    Args:
        amount(int): The amount of date times you want to use for computing
            and evaluating an interval.
        list(list): A list containing date times which can be used for
            computing and evaluating intervals.
        series(pd series): A (pandas) series containing the date times of the
            list above and the belonging actuals.
        alpha (int): The significance level of the evaluated intervals.
        dimension (int): The number of hours used for computing the evaluated
            intervals.
    Returns:
        pl(float): The proportion of points on the left side of the intervals
            in percent.
        pr(float): The proportion of points on the right side of the intervals
            in percent.
        pin(float): The proportion of points inside the intervals in percent.
        average_width (float): The average width of the evaluated intervals.
        number (int): The number of days, for which an interval was
            evaluated.
        score (float): The ellipsoidal skill score for the method of
            computing the intervals.
    """

    a = 0
    b = 0
    inside = 0
    number = int(number)
    sum_width = 0
    print(dimension)
    if number > len(list):
        number = len(list)

    s=[]
    for i in range(0, number):
        future_time = list[i]
        lo, up = compute_uni_interval.main(future_time)
        act = series[future_time]
        sum_width += (up-lo)
        if act < lo:
            a += 1
            s.append((-alpha) * ((up-lo))**(1/dimension))
        elif act > up:
            b += 1
            s.append((-alpha) * ((up - lo)) ** (1 / dimension))
        else:
            inside += 1
            s.append((1-alpha) * ((up - lo)) ** (1 / dimension))

    score = abs(sum(s)/10)
    average_width = sum_width/number
    pl = round((a / len(range(0, number))) * 100, 2)
    pr = round((b / len(range(0, number))) * 100, 2)
    pin = round((inside / len(range(0, number))) * 100, 2)

    return a, b, inside,  pl, pr, pin, average_width, number, score

def main():
    args=parser.parse_args()

    fd, daylist, series = set_list(args.data_file,
                                   compute_uni_interval.f_time(),
                                   args.hour_range_size)
    a, b, inside, pl, pr, pin, aw, number, score = count(args.number_of_days,
                                                         daylist, series,
                                                         args.alpha,
                                                         args.hour_range_size)

    print('Alpha:', args.alpha*100, '%')
    print('First training date:', fd)
    print('First analysis date:', daylist[0])
    print('Number of analysis days:', number)
    print('Out left (number):', a, 'Out right (number):', b)
    print('Out left (proportion):', pl, '%. Out right (proportion):', pr, '%')
    print('Inside(number):', inside, 'Inside (proportion):', pin, '%')
    print('Average width:', aw)
    print('Ellipsoidal skill score:', score)

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('Time:', t2 - t1)