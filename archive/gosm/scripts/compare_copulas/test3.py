import horse_race_copulas as hr
import prescient.distributions.copula as cop
import numpy as np
import matplotlib.pyplot as plt
import time

filename = '../../../../prescient_cases/bpa_wind/bpa_wind_forecasts_actuals_2012_2017.csv'
filename_out = 'bpa\_wind\_forecasts\_actuals\_2012\_2017.csv'

startdates = ['2015-01-02:12:00', '2015-01-02:13:00']
starts_out = ['2015-01-02_12_00', '2015-01-02_13_00']

n_test = 10000
t = str(1)
n_gen = 10000

method = 'compare'
random = 'fixed'

dia = 1

t13 = time.time()

wd1_i, number1_i, ig1_i, avgm1_i, avgc1_i, \
histo1_i, S1_i, fd12, ws12 = hr.main(filename, startdates, n_test, cop.IndependenceCopula,
                         n_gen, dia, method = method, random = random)
t14 = time.time()
print('done independece diagonal 1,', t13-t14)

X = S1_i
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo1_i.r - 1):
        j += 1
        bool = X[i] > histo1_i.Rank[j]
    A[i] = (
           j + 0.5) / histo1_i.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo1_i.r + 1)) / histo1_i.r
C = np.asarray(range(histo1_i.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('independence')

plt.savefig('test3.png')
#plt.show()
plt.close()

print('done')