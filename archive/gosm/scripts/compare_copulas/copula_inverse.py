import prescient.distributions.copula as cop
import prescient.distributions.distributions as dist
import numpy as np
import gosm.copula_experiments.copula_evaluate as eva

sample = dist.UnivariateNormalDistribution(0.1,0).generates_X(10000)
rank_data = dist.UnivariateUniformDistribution(0,1).generates_X(10000)

histo = eva.RankHistogram(rank_data=rank_data, rank=10)

histo.plot(sample= sample)