import gosm.copula_experiments.copula_evaluate as test
import prescient.distributions.distributions as dist
import horse_race_copulas as hr
import numpy as np


data = dist.UnivariateUniformDistribution(0,1).generates_X(n=10000)

histo = test.RankHistogram(distr=dist.UnivariateNormalDistribution(0.1,0), rank_data=data, rank=10)

print(histo.Rank)

print('done')

histo.plot()

"""

A = np.random.normal(0,4,10000)
B = dist.UnivariateNormalDistribution(4,0).generates_X(10000)

hr.histo_uniform(A,10)

hr.histo_uniform(B,10)

"""