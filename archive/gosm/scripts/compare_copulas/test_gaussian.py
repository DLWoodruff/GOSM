import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

mean = [0, 0]
cov = [[1, 0], [0, 100]]

x, y = np.random.multivariate_normal(mean, cov, 5000).T
computed_cov = np.cov(x, y)
computed_cor = np.corrcoef(x, y)
print('cov', computed_cov)
print('cor', computed_cor)
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

a, b = np.random.multivariate_normal(mean, computed_cov, 5000).T
plt.plot(a, b, 'x')
plt.axis('equal')
plt.show()
"""

Z = np.random.multivariate_normal([0, 0], [[0.8, 0.2], [0.2, 0.8]], 10000)
print(np.corrcoef((Z.T)))
print(np.cov(Z.T))

x, y = np.random.multivariate_normal(np.zeros(2), np.corrcoef(Z.T), 10000).T
a, b = np.random.multivariate_normal(np.zeros(2),np.cov(Z.T), 10000).T


res1 = sps.ndtr(x)
res2 = sps.ndtr(y)
res3 = sps.ndtr(a)
res4 = sps.ndtr(b)
print(res1)
print(res2)


# Three subplots sharing both x/y axes
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(x, y, 'x')
ax2.plot(a, b, 'x')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)

plt.show()



plt.figure(1)
plt.subplot(211)
plt.plot(x, y, 'x')
plt.axis('equal')

plt.subplot(212)
plt.plot(a, b, 'x')
plt.axis('equal')

plt.show()

"""