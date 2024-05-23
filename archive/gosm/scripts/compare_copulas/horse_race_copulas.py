"""
horse_race_copulas.py

This script provides two different methods to evaluate how accurate copulas
fit a given data set. One can use this script to do a horse race between
different copulas or different ways of computing one copula (e.g. regarding
the segmentation of data used to fit the copula).

Method 1 uses the Earth Mover Distance of the copula to the uniform
distribution. Method 2 uses a rank histogram computed of the copula. Because
both methods only work for one dimension, we consider only the diagonals.

For more theoretical details see the article about this horse race.
"""

import pandas as pd
import datetime
import prescient.gosm.sources as sources
import prescient.gosm.sources.segmenter as segmenter
import numpy as np
import prescient.distributions.distributions as dist
import prescient.distributions.copula as cop
import prescient.gosm.copula_experiments.copula_diagonal as di
import prescient.gosm.copula_experiments.copula_evaluate as eva
import scipy.stats as stats
import multiprocessing as mp
import time

def read_data(filename, startdates, n):
    """
    This function reads the data from a given file and prepares it for
    the further using in this script.

    Args:
        filename (string): The filename of the data file.
        startdates (list(strings)): A list of startdates. Each date must be a
                string with the format "YYYY-MM-DD:HH:00". Furthermore, each
                date in this list must have the same date, only the time have
                to differ.
        n (int): Number of test days.

    Returns:
        df (pandas dataframe): The created and prepared dataframe containing
                the data provided in the file.
        dates (list(list(strings)): Test days for each startdate.
        dimension (int): The dimension of the random variable. For each start-
                date (or starting hour) the dimension is increased by one.
    """

    # First a pandas dataframe from a file containing forecasts/actuals data.
    # Then a error column is added to the dataframe
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df['errors'] = df['actuals'] - df['forecasts']
    dates = []
    ignored_days = []
    start_hour = []
    # Creating a list of the hours which correspond to one dimension of the
    # random variable.
    for date in startdates:
        time = datetime.datetime.strptime(date, '%Y-%m-%d:%H:%M')
        dt = pd.Timestamp(time)
        start_hour.append(dt.hour)
    # If there is no data available for one of the considered hours at some
    # dates, these dates are deleted from the dataframe. These are
    for date in df.index:
        for hour in start_hour:
            if date.hour == hour:
                x = start_hour.index(hour)
                l = list(start_hour)
                l.pop(x)
                for h in l:
                    if date.replace(hour = h) not in df.index:
                        ignored_days.append(date)
                        df = df.drop(date)

    #print(len(df))
    df = df[~df.index.duplicated()]
    # The list which contains the test days is created and afterwards converted
    # to the form (dimension x number_of_test_days).
    for date in startdates:
        time = datetime.datetime.strptime(date, '%Y-%m-%d:%H:%M')
        dt = pd.Timestamp(time)
        hour_mask = df.index.hour == dt.hour
        dates.append(df[(df.index >= dt) & hour_mask].index.tolist()[:n])
    first_day = df.index.tolist()[0].date()
    return df, np.array(dates).T.tolist(), ignored_days, len(startdates), \
           first_day

def split_data(dataframe, date):
    """
    This function splits the data for a given date into a historic and actual
    part.

    Args:
         dataframe (pandas dataframe): The dataframe which will be splitted.
         date (Timestamp): The date on which the dataframe will be splitted.

    Returns:
        RollingWindow: The RollingWindow object containing the historic and
                actual data.
        actual_error (float): The error at the given date.
    """
    hour_mask = dataframe.index.hour == date.hour
    historic_data = (dataframe[(dataframe.index < date)])
    actual_data = (dataframe[(dataframe.index == date)])
    actual_error = float(actual_data['errors'])
    historic_data_cop = (dataframe[(dataframe.index < date) & hour_mask])
    actual_data_cop = (dataframe[(dataframe.index == date) & hour_mask])
    return sources.RollingWindow('BPA Wind', historic_data, 'wind',
                                 actual_data), sources.RollingWindow('BPA Wind', historic_data_cop, 'wind',
                                 actual_data_cop), actual_error

def fit_copula(dates, dataframe, copula_class):
    """
    This function fits marginals to each dimension of the random variable
    (corresponding to one specific hour). After that a copula is fitted to
    the given data.

    Args:
         dates (list(Timestamp)): A list containing two Timestamps. One for
                every dimension of the random variable.
         dataframe (pandas dataframe): The dataframe containing the data.
         copula_class: The copula class fitted to the data.

    Returns:
        marginals (list(distribution)): The univariate error distribution for
                every dimension of the random variable (i.e. for every
                considered hour).
        copula: The fitted copula (i.e. a multivariate distribution).
        actual_errors (list(float)): Errors regarding every date in dates.
    """

    # First a criterion for the marginal-segmentation is chosen.
    ws = 0.4
    criterion = segmenter.Criterion('forecasts', 'forecasts', 'window', ws)
    marginals = []
    transformed = []
    actual_errors = []
    # For each date in the dates list (corresponding to one dimension of the
    # random variable) an error distribution is fitted after segmenting the
    # data. For the purpose of fitting a copula all error data is considered.
    # Before fitting a copula to it the data has to be transformed into the
    # interval [0,1]^d.
    for date in dates:
        source_m, source_c, actual_error = split_data(dataframe, date)
        actual_errors.append(actual_error)
        segmented_source = source_m.segment_by_window(date, criterion)
        seg_errors = segmented_source.get_column('errors')
        distr = dist.UnivariateEmpiricalDistribution(seg_errors)
        marginals.append(distr)
        errors = source_c.get_column('errors')
        transformed.append([distr.cdf(x) for x in errors])
    # The copula from the given class is fitted to the transformed error data.
    copula = copula_class.fit(transformed)
    training_observations_copula = len(errors)
    training_observations_marginals = len(seg_errors)
    return marginals, copula, actual_errors, training_observations_marginals, \
           training_observations_copula, ws

def projection(copula, n, d, proj, i, actual_errors, marginals):
    """
    This function projects realizations of the copula and an array of observed
    data to a diagonal of the copula space. Therefore some existing code
    is used. For more theoretical background read the report regarding
    this script.

    Args:
         copula: The copula from which the realizations are projected to the
                diagonal.
         n (int): Number of realizations which are generated from the copula.
         d (int): Dimension of the copula space (and the random variable
                represented by the copula)
         proj (string): The method of projection. Provided are "scalar" and
                "vector". (WARNING: "vector" isn't fully implemented yet. If
                this method is chosen, the scalar method is used instead.)
         i (int): The diagonal on which the points are projected.
         actual_errors (list(floats)): The observed data (usually errors).
         marginals (list(distribution)): The univariate error distribution for
                every dimension of the random variable (i.e. for every
                considered hour).

    Returns:
        P: The projected realizations of the copula. The type depends on the
            used projection mehtod.
        R: The projected observed data. The type depends on the used
            projection method.
    """
    # Realizations of the copula are generated with the build in function.
    U = copula.generates_U(n)
    # A diagonal object is created by using existing code.
    diagonal = di.diag(d)
    # For the projection itself existing code is used, too.
    if proj == 'scalar':
        P = diagonal.proj_scalar(U, i)
    elif proj == 'vector':
        P = diagonal.proj_vector(U, i)
    else:
        raise ValueError('The projection method does not match any provided '
                         'method.')
    Q = []
    # The observed data has to be transformed into the copula space before the
    # projection (which is executed by existing code again).
    for j in range(d):
        Q.append(marginals[j].cdf(actual_errors[j]))
    if proj == 'scalar':
        R = diagonal.proj_scalar(Q, i)
    elif proj == 'vector':
        R = diagonal.proj_vector(Q, i)
    else:
        raise ValueError('The projection method does not match any provided '
                         'method.')

    return P, R

def compute_s(day, dataframe, copula_class, n, d, proj, i):
    # For every test day a separate copula is fitted to the regarding
    # (historic) data.
    marginals, copula, actual_errors, tr_marg, \
    tr_cop, ws = fit_copula(day, dataframe, copula_class)
    # Using this specific copula for each day, the projections of
    # realizations from this copula and the observed data on this day are
    # computed.
    P, R = projection(copula, n, d, proj, i , actual_errors, marginals)
    # The "cdf" value of the empirial distribution of the projected
    # realizations from the copula for the observed data is appended to S.
    s = (sum(1 for i in range(n) if P[i] <= R)) / n
    return s, tr_marg, tr_cop, ws


def cdf_sample(days, dataframe, copula_class, n, d, proj, i):
    """
    This function creates the sample S, which contains the "cdf" values of
    the observed data for a empirical distribution constructed with the
    projected realizations of the copula. For more theoretical background
    read the report regarding this script.

    Args:
        days (list(list(Timestamp))): The test days. At each day one "cdf"
                value is computed.
        dataframe: The provided dataframe with the data.
        copula_class: The copula class which is tested.
        n (int): Number of realizations which are generated from the copula.
        d (int): Dimension of the copula space (and the random variable
                represented by the copula)
        proj (string): The method of projection. Provided are "scalar" and
                "vector". (WARNING: "vector" isn't fully implemented yet. If
                this method is chosen, the scalar method is used instead.)
        i (int): The diagonal on which the points are projected.

    Returns:
        S (list(float)): The sample of "cdf" values.
    """
    S = []
    # The vector projection isn't fully implemented yet. If this method is
    # chosen, the scalar method is used instead and a info message is printed.
    if proj == 'vector':
        proj = 'scalar'
        print("The vector projection isn't fully implemented yet. It was used "
              "the scalar projection instead.")
    training_marg = 0
    training_cop = 0
    window_size = []
    for day in days:
        s, tr_marg, tr_cop,\
        ws = compute_s(day, dataframe, copula_class, n, d, proj, i)
        training_marg += tr_marg
        training_cop += tr_cop
        window_size.append(ws)
        S.append(s)

    average_training_marg = training_marg / len(days)
    average_training_cop = training_cop / len(days)
    avg_marg = round(average_training_marg)
    avg_cop = round(average_training_cop)

    return S, avg_marg, avg_cop, window_size[0]

def cdf_sample_mp(days, dataframe, copula_class, n, d, proj, i, num_processes = 30):

    S = []
    training_marg = 0
    training_cop = 0
    window_size = []

    with mp.Pool(num_processes) as pool:
        dayruns = [pool.apply_async(compute_s, (day, dataframe, copula_class, n, d, proj, i,)) for day in days]
        results = [result.get() for result in dayruns]

    for res in results:
        s, tr_marg, tr_cop, ws = res
        training_marg += tr_marg
        training_cop += tr_cop
        window_size.append(ws)
        S.append(s)

    average_training_marg = training_marg / len(days)
    average_training_cop = training_cop / len(days)
    avg_marg = round(average_training_marg)
    avg_cop = round(average_training_cop)

    return S, avg_marg, avg_cop, window_size[0]



def wasserstein(A, random):
    """
    This function computes the Earth Mover Distance between a given sample and
    a uniformly distributed sample. Therefore it uses the Wassterstein Distance
    from scipy.stats. Wasserstein Distance is another name for the Earth
    Mover Distance.

    Args:
        A (list): From this sample the distance to the uniform sample is
                computed (i.e. the distance between the empirical distribution
                of the sample A to the uniform distribution).
        random (string): Specifies which uniform sample is used: The random one
                (i.e. "random") or the fixed one (i.e. "fixed").
    Returns:
        wassterstein_distance (float): The computed Wasserstein distance
                (or EMD).
        n (int): The length of the Samples.
    """
    n = len(A)
    if random == 'random':
        # In this case a random uniform sample is drawn and used to compute
        # the Wasserstein distance
        X = np.random.uniform(0,1,n)
        wasserstein_distance = stats.wasserstein_distance(A, X)
        return wasserstein_distance, n
    elif random == 'fixed':
        # In this case a fixed sample, which is also uniformly distributed,
        # is used to compute the Wasserstein distance.
        X = [0]
        for i in range(n):
            X.append(X[i] + (1/n))
        wasserstein_distance = stats.wasserstein_distance(A, X)
        return wasserstein_distance, n

def emd_uniform(A, emd_type, p, random):
    """
    This function computes the Earth Mover Distance between a given array and
    an array sampled from a uniform distribution. So the distance between the
    empirical distribution of the given array to the uniform distribution is
    computed. For more theoretical background read the report regarding this
    script.

    Args:
         A (list): The list the EMD from the uniform sample is computed.
                (i.e. the distance from the empirical distribution of this list
                to the uniform distribution)
         emd_type (string): There two differnt ways to compute the EMD;
                        1. Using the explicit formula from the report.
                            (i.e. emd_type = "sort")
                        2. Solving the optimization problem from the report
                            with pyomo. (i.e. emd_type = "pyomo")
         p (int): The p-th EMD is computed.

    Returns:
        EMD: The computed EMD>
        n (int): The length of the sample (which is the same as the number
                of test days)
    """
    # First a random uniformly distributed sample from the same length is
    # computed. After that the EMD between these two samples is computed by
    # using existing code.
    n = len(A)
    if random == 'random':
        X = np.random.uniform(0,1,n)
    elif random == 'fixed':
        X = [0]
        for i in range(n):
            X.append(X[i] + (1 / n))
    if emd_type == 'sort':
        return eva.emd_sort(X, A, p = p), n
    elif emd_type == 'pyomo':
        return eva.emd_pyomo(X, A, p = p), n
    else:
        raise ValueError('The EMD type does not match any provided type.')

def histo_uniform(A, r, output_file, method, random):
    """
    This function creates a rank histogram of a uniform distribution and
    populates it with the given sample. After that the histogram is shown or
    saved.

    Args:
        A (list): Sample which populates the rank histogram.
        r (int): Number of ranks for the rank histogram.
        output_file (string): Name of the output file for saving the plot.
                If None is chosen, the plot is not saved, but it is shown
                instead.
    Returns:
        n (int): The length of the given sample (which is the same as the
                number of test days)
    """
    # First a random uniformly distributed sample from the same length is
    # computed. After that the histogram is computed by using existing code.
    n = len(A)
    if random == 'random':
        X = np.random.uniform(0,1,n)
    elif random == 'fixed':
        X = [0]
        for i in range(n):
            X.append(X[i] + (1 / n))
    histo = eva.RankHistogram(rank_data=X, rank=r)
    if method == 'compare':
        return histo
    else:
        histo.plot(sample = A, n = n, output_file = output_file)
        return n

def main(filename, startdates, n_test, copula, n_gen, dia,
         proj = 'scalar', method = 'wasserstein', emd_type = 'sort', p = 1,
         rank = 10, output_file = None, random = 'fixed'):
    """
   The main function which puts all the steps for the algorithm given in
   the report together. Furthermore the main function gives the
   opportunity to specify all the arguments used in this algorithm.

   Args:
        filename (string): The data file.
        startdates (list(strings)): A list of startdates. Each date must be a
                string with the format "YYYY-MM-DD:HH:00". Furthermore, each
                date in this list must have the same date, only the time have
                to differ.
        n_test (int): Number of test days.
        copula (copula object): The tested copula class.
        n_gen (int): Number of realizations from the copula each day.
        dia (int): The diagonal projected to.
        proj (string): The projection method. "scalar" and "vector" are
                provided. (WARNING: Only "scalar" available at the moment)
        method (string): Evaluation method. "wasserstein" (Wasserstein
                distance), "emd" (EMD), "histogram" (rank histogram),
                "histoEMD" (histogram and EMD), "histoWD" (histogram
                and Wasserstein distance) and "all" (all methods) are
                provided.
        emd_type (string): If the method includes EMD, one have to define the
                way of computing it. "pyomo" and "sort" are provided.
        p (int): If the method includes EMD, the p-th EMD is computed.
        rank (int): If the method includes rank histogram, one have to define
                the number of ranks for the rank histogram.
        output_file (string): If the method includes rank histogram, one can
                decide if the histogram shall be saved or just shown. For
                saving a filename has to be provided. Otherwise it is set to
                None and the plot is just shown.
        random (string): If the method includes the Wasserstein distance,
                one can decide if it is computed using a random sample
                ("random") or using a fixed sample ("fixed"). Both are
                uniformly distributed.
    """
    data, dates, ig, d , first_day = read_data(filename, startdates, n_test)
    S, avg_marg, avg_cop, ws = cdf_sample_mp(dates, data, copula, n_gen, d, proj, dia)
    if method == 'wasserstein':
        wasserstein_distance, number = wasserstein(S, random)
        wd = round(wasserstein_distance, 4)
        return wd, number, ig, avg_marg, avg_cop, first_day, ws
    elif method == 'emd':
        E, number = emd_uniform(S, emd_type, p, random)
        emd = round(E, 4)
        return emd, number, ig, avg_marg, avg_cop, first_day, ws
    elif method == 'histogram':
        number = histo_uniform(S, rank, output_file, method, random)
        return number, avg_marg, avg_cop, first_day, ws
    elif method == 'histoEMD':
        E, number1 = emd_uniform(S, emd_type, p, random)
        emd = round(E, 4)
        number2 = histo_uniform(S, rank, output_file, method, random)
        if number1 != number2:
            raise ValueError('The number of test days are not the same for '
                             'both methods!')
        return emd, number1, ig, avg_marg, avg_cop, first_day, ws
    elif method == 'histoWD':
        wasserstein_distance, number1 = wasserstein(S, random)
        wd = round(wasserstein_distance, 4)
        number2 = histo_uniform(S, rank, output_file, method, random)
        if number1 != number2:
            raise ValueError('The number of test days are not the same for '
                             'both methods!')
        return wd, number1, ig, avg_marg, avg_cop, first_day, ws
    elif method == 'all':
        E, number1 = emd_uniform(S, emd_type, p, random)
        emd = round(E, 4)
        wasserstein_distance, number2 = wasserstein(S, random)
        wd = round(wasserstein_distance, 4)
        number3 = histo_uniform(S, rank, output_file, method, random)
        if (number1 != number2) or (number2 != number3) or (number1 != number3):
            raise ValueError('The number of test days are not the same for '
                             'both methods!')
        return emd, wd, number1, ig, avg_marg, avg_cop, first_day, ws
    elif method == 'compare':
        wasserstein_distance, number = wasserstein(S, random)
        wd = round(wasserstein_distance, 4)
        histo = histo_uniform(S, rank, output_file, method, random)
        return wd, number, ig, avg_marg, avg_cop, histo, S, first_day, ws

    else:
        raise ValueError('The given method does not match any provided method.')





if __name__ == "__main__":
    #filename = '../../sim/gosm_test/all_bpa_data.csv'
    filename = '../../../../prescient_cases/bpa_wind/bpa_wind_forecasts_actuals_2012_2017.csv'
    #startdates = ['2016-11-03:18:00', '2016-11-03:08:00']
    startdates = ['2013-01-01:12:00', '2013-01-01:13:00']

    """
    n_test = 1000
    copula = cop.StudentCopula
    n_gen = 10000
    dia = 0
    method = 'histoWD'
    #proj = 'scalar'
    #random = 'fixed'
    #output = 'test2.png'



    wd, number, ig , avg = main(filename, startdates, n_test, copula, n_gen, dia, method = method)

    #print('EMD(' + emd_type + '):', E)
    print('Wasserstein distance:', wd)
    print('number of test days:', number)
    print('average number of training days', avg)
    print('number of ignored days:', len(ig))
    print('list of ignored days:', ig)
    """

    n_test = 100

    data, dates, ig, d, first_day = read_data(filename, startdates, n_test)
    #print(len(data))
    #print(data.index.tolist()[3600:3800])
    #print(dates[0][0])
    source, actual_error = split_data(data, dates[0][0])
    print(source.dayahead_data)
    print(source.historic_data)
