import horse_race_copulas_draft as hr
import prescient.distributions.copula as cop
import numpy as np
import prescient.distributions.distributions as dist
import gosm.copula_experiments.copula_diagonal as diag
import scipy.stats as stats
import scipy.special as sps


#testing hr.projection

"""
copula = cop.GaussianCopula
raw_data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 10000).tolist()
data = [[], []]
for b in raw_data:
    data[0].append(b[0])
    data[1].append(b[1])

n = 10000
d = 2
i = 0
marginals = [dist.UnivariateUniformDistribution(1,0), dist.UnivariateUniformDistribution(1,0)]
raw_actuals = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 1)
actuals = [raw_actuals[0,0], raw_actuals[0,1]]
proj = 'vector'
P, R = hr.projection(copula, data, n, d, i, marginals, actuals, proj)
print('P:', P)
print('R:', R)


#testing hr.cdf_sample


copula = copula.GaussianCopula
raw_data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 10000).tolist()
data = [[], []]
for b in raw_data:
    data[0].append(b[0])
    data[1].append(b[1])
n = 10000
d = 2
i = 0
marginals = [dist.UnivariateUniformDistribution(0,1), dist.UnivariateUniformDistribution(0,1)]
proj = 'scalar'
days = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 10).tolist()

S = hr.cdf_sample(days, copula, data, n, d, i, marginals, proj)
print(S)

"""

#testing hr.main

copula1 = cop.GaussianCopula
copula2 = cop.GumbelCopula


raw_data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 10).T.tolist()
#raw_data = [[],[]]
#raw_data[0] = stats.gumbel_l.rvs(size=10000)
#raw_data[1] = stats.gumbel_l.rvs(size=10000)

#print(raw_data)
#print(np.cov(data))

marginals = [dist.UnivariateEmpiricalDistribution(raw_data[0]), dist.UnivariateEmpiricalDistribution(raw_data[1])]

data=[[],[]]
for x,y in zip(raw_data[0],raw_data[1]):
    data[0].append(marginals[0].cdf(x))
    data[1].append(marginals[1].cdf(y))

n = 10
d = 2
i = 1
proj = 'scalar'
#days = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)
copula = copula1.fit(data)
days = copula.generates_U(10)

method = 'histogram'
emd = 'sort'
p = 1
r = 10


A = hr.main(method, days, copula1, data, n, d, i, marginals, emd,
         proj, p, r, 'test.png')
#B = hr.main(method, days, copula2, data, n, d, i, marginals, emd,
 #        proj, p, r)
print(str(copula1), type(A))
#print(str(copula2), B)