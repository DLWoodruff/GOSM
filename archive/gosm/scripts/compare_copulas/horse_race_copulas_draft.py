"""
horse_race_copulas_draft.py

***FIRST DRAFT OF horse_race_copulas.py***

This script provides two different methods to horse race different copulas
in modeling some given data. After running this script one can decide which
copula models the given data the best.

Method 1 uses the Earth Mover Distance of the copula to the uniform
distribution. Method 2 uses a rank histogram computed of the copula. Because
both methods only work for one dimension, we consider only the diagonals.

For more theoretical details see the article about this horse race.
"""

import gosm.copula_experiments.copula_diagonal as diag
import numpy as np
import gosm.copula_experiments.copula_evaluate as evaluate
import prescient.distributions.distributions as dist

def read_data():
    pass

def projection(copula, data, n, d, i, marginals, actuals, proj = 'scalar'):
    distr = copula.fit(data)
    #print(distr.cor_matrix)
    U = distr.generates_U(n)
    diagonal = diag.diag(d)
    #print('list:', diagonal.list_of_diag[i])
    #print('directions:', diagonal.directions[i])
    #print('projection:', diagonal.projections[i])
    if proj == 'scalar':
        P = diagonal.proj_scalar(U, i)
    elif proj == 'vector':
        P = diagonal.proj_vector(U, i)
    else:
        raise ValueError('The projection method does not match any provided '
                         'method.')
    var = np.var(P)
    mean = np.mean(P)
    #print('var:', var)
    #print('mean:', mean)
    Q = []
    #print('actuals:', actuals)
    for j in range(d):
        #Q.append(marginals[j].cdf(actuals[j]))
        Q.append(actuals[j])
    #print('Q:', Q)
    if proj == 'scalar':
        R = diagonal.proj_scalar(Q, i)
    elif proj == 'vector':
        R = diagonal.proj_vector(Q, i)

    return P, R


def cdf_sample(days, copula, data, n, d, i, marginals, proj = 'scalar'):
    S = []
    R2 = []
    for day in days:
        P, R = projection(copula, data, n, d, i, marginals, day, proj)
        R2.append(R)
        S.append((sum(1 for i in range(n) if P[i] <= R))/n)
    print('var:', np.var(R2))
    print('mean:', np.mean(R2))
    return S


def emd_uniform(A, emd = 'sort', p = 1):
    n = len(A)
    X = np.random.uniform(0,1,n)
    if emd == 'sort':
        return evaluate.emd_sort(X, A, p = p)
    elif emd == 'pyomo':
        return evaluate.emd_pyomo(X, A, p = p)
    else:
        raise ValueError('The EMD type does not match any provided type.')

def histo_uniform(A, r, output_file):
    n = len(A)
    X = np.random.uniform(0,1,n)
    histo = evaluate.RankHistogram(rank_data=X, rank=r)
    histo.plot(sample = A, n = n, output_file = output_file)


def main(method, days, copula, data, n, d, i, marginals, emd,
         proj = 'scalar', p = 1, r = 10, output_file = None):
        S = cdf_sample(days, copula, data, n, d, i, marginals, proj)
        print(len(S))
        if method == 'emd':
            return emd_uniform(S, emd, p)
        elif method == 'histogram':
            histo_uniform(S, r, output_file)
        else:
            raise ValueError('Only emd and histogram are available methods.')


