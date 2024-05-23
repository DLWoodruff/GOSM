"""
data_preprocessor.py

This script preprocesses given data. It includes functions to change the time
steps in the data's index. It's easy, to make the time steps bigger: Just sum
up the regarding rows.
Making the steps smaller, is not that easy. We provide a function for that,
but this function only interpolates the given data. It has nothing to do with
real data. Therefore it is not meant for users. We only use it to get data
with small time steps for testing other scripts.
"""

import pandas as pd
import numpy as np
import time
import os
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--source-file',
                    help='The data containing csv file for which the time '
                         'steps should be changed.',
                    type=str,
                    dest='source_file')

parser.add_argument('--output-directory',
                    help="After changing the time steps the data is stored "
                         "in this directory. If this directory doesn't exist, "
                         "it will be created.",
                    type=str,
                    dest='output_directory')

parser.add_argument('--output-name',
                    help='If this option is set, the output file will be '
                         'named like in this option specified. It must end '
                         'with ".csv". If it is not set, the original name '
                         'complemented by the new time step will be used.',
                    type=str,
                    dest='output_name',
                    default=None)

parser.add_argument('--time-step',
                    help='The desired time step for the processed data. It '
                         'must be a pandas offset alias.',
                    type=str,
                    dest='time_step')

parser.add_argument('--change-type',
                    help='Here you can set how you want to change the time '
                         'steps. The two options are either "smaller" or '
                         '"bigger".',
                    type=str,
                    dest='change_type')


def preprocess(data):
    """
    This function preprocesses the data for making bigger time steps.
    Therefor it is necessary to ignore time periods in which are not enough
    time steps available (i.e. we ignore all time periods with missing data).
    We also ignore the last period, if its not complete (again because of
    missing data). With this procedure we want to make sure, that every
    new time step includes the same amount of data points.

    Args:
        data (pandas DataFrame): The data which is preprocessed.
    Return:
        data (pandas DataFrame): The preprocessed data.
    """
    # First we need the frequency of the given data.
    time_step = pd.infer_freq(data.index)
    if time_step == None:
        time_step = pd.infer_freq(data.index[:5])

    # We fill missing data with -inf. All rows with a value of -inf will be
    # dropped later.
    start = data.index[0]
    end = data.index[len(data.index)-1]
    dates = pd.date_range(start, end, freq=time_step)
    for date in dates:
        if date not in data.index:
            df = pd.DataFrame(data={name: -np.inf
                                    for name in list(data)},
                              index=[date])
            data = data.append(df)

    # We also add another time step to the end with the value -inf, because
    # this data is also missing. But first we need to convert the string of
    # the frequency, so that it can be handled by pandas Timedelta.
    if "T" in time_step:
        time_step = time_step.replace("T", "min")
    if not any(char.isdigit() for char in time_step):
        time_step = '1' + time_step
    additional = end + pd.Timedelta(time_step)
    df = pd.DataFrame(data={name: -np.inf for name in list(data)},
                      index=[additional])
    data = data.append(df)
    return data

def drop_rows(data):
    """
    This function drops a row in the passed in DataFrame, if one of it's values
    is not a number or -inf.

    Args:
        data (pandas DataFrame): The data.
    Returns:
        data (pandas DataFrame): The data after dropping some rows.
    """
    for date in data.index:
        for column in list(data):
            if data.loc[date][column] == np.nan:
                data = data.drop(date)
                break
            elif data.loc[date][column] == -np.inf:
                data = data.drop(date)
                break
    return data

def bigger_time_step(data, time_step):
    """
    This function creates a new DataFrame out of the passed in with bigger time
    steps. Therefore it just resamples the old data and sums up the values of
    the old time steps to form bigger time steps.

    Args:
         data (pandas DataFrame): The data which will be resampled.
         time_step (string): The new, bigger time step for the "new" data.
    Returns:
        cleared_data (pandas DataFrame): The created DataFrame with bigger time
            steps in the index.
    """
    processed_data = preprocess(data)
    raw_data = processed_data.resample(time_step).sum()
    cleared_data = drop_rows(raw_data)
    return cleared_data

###############################################################################
# We only use the function smaller_time_step to get 5 min data out of hourly  #
# data for testing some other code. This function is not meant for users.     #
#                                                                             #
# If you want to provide a similar function for users, you have to think      #
# about it harder. Maybe one opportunity is to shift the interpolation to     #
# the middle points of the bigger time periods.                               #
###############################################################################

def smaller_time_step(data, time_step):
    """
    This function creates a new DataFrame out of the passed in with smaller
    time steps. Therefore it interpolates the values within the old time steps.

    Args:
        data (pandas DataFrame): The data from which the new data will be
            interpolated.
        time_step (string): The new, smaller time step size.
    Returns:
        new_data (pandas DataFrame): The created DataFrame with smaller time
            stepd in the index.
    """
    new_df = data.resample(time_step).sum()
    new_data = new_df.interpolate()
    return new_data

def main():
    args = parser.parse_args()

    old_data = pd.read_csv(args.source_file, index_col=0, parse_dates=True,
                           infer_datetime_format=True)

    if args.change_type == "smaller":
        new_data = smaller_time_step(old_data, args.time_step)
    elif args.change_type == "bigger":
        new_data = bigger_time_step(old_data, args.time_step)
    else:
        raise ValueError("Your provided change type is not provided!")

    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    if args.output_name:
        name = args.output_name
    else:
        old_name = args.source_file.split('/')[-1].split('.')[0]
        name = old_name + '_ts_' + args.time_step + '.csv'

    output_path = args.output_directory + name

    new_data.to_csv(output_path)



if __name__ == '__main__':
    main()


