"""
This script runs experiments for horse_race_copulas.py.
"""

import horse_race_copulas as hr
import prescient.distributions.copula as cop
import numpy as np
import matplotlib.pyplot as plt
import time

#Running the experiments with same parameters but different copulas

#filename = '../../sim/gosm_test/all_bpa_data.csv'
#filename_out = 'all\_bpa\_data.csv'

#filename = '../../sim/gosm_test/2012-2013_BPA_forecasts_actuals.csv'
#filename_out = '2012-2013\_BPA\_forecasts\_actuals.csv'

#filename = 'RTS_data/WIND_forecasts_actuals.csv'
#filename_out = 'WIND\_forecasts\_actuals.csv'

#filename = 'RTS_data/122_WIND_1_forecasts_actuals.csv'
#filename_out = '122\_WIND\_1\_forecasts\_actuals.csv'

filename = '../../../../prescient_cases/bpa_wind/bpa_wind_forecasts_actuals_2012_2017.csv'
filename_out = 'bpa\_wind\_forecasts\_actuals\_2012\_2017.csv'


#startdates = ['2016-02-21:08:00', '2016-02-21:09:00']
#starts_out = ['2016-02-21_08_00', '2016-02-21_09_00']

#startdates = ['2013-08-12:21:00', '2013-08-12:22:00']
#starts_out = ['2013-08-12_21_00', '2013-08-12_22_00']

#startdates = ['2020-11-11:02:00', '2020-11-11:03:00']
#starts_out = ['2020-11-11_02_00', '2020-11-11_03_00']

startdates = ['2015-01-02:12:00', '2015-01-02:13:00']
starts_out = ['2015-01-02_12_00', '2015-01-02_13_00']

n_test = 2000
t = str(1)
n_gen = 10000

method = 'compare'
random = 'random'

dia = 0

t1 = time.time()

wd0_g, number0_g, ig0_g, avgm0_g, avgc0_g, \
histo0_g, S0_g, fd , ws= hr.main(filename, startdates, n_test, cop.GaussianCopula,
                         n_gen, dia, method = method, random = random)

t3 = time.time()
print('done gaussian diagonal 0,', t3-t1)
wd0_gu, number0_gu, ig0_gu, avgm0_gu, avgc0_gu, \
histo0_gu, S0_gu, fd1, ws1 = hr.main(filename, startdates, n_test, cop.GumbelCopula,
                           n_gen, dia, method = method, random = random)
t4 = time.time()
print('done gumebl diagonal 0,', t3-t4)
wd0_f, number0_f, ig0_f, avgm0_f, avgc0_f, \
histo0_f, S0_f, fd2, ws2 = hr.main(filename, startdates, n_test, cop.FrankCopula, n_gen,
                         dia, method = method, random = random)
t5 = time.time()
print('done frank diagonal 0,', t4-t5)
wd0_c, number0_c, ig0_c, avgm0_c, avgc0_c,  \
histo0_c, S0_c, fd3, ws3 = hr.main(filename, startdates, n_test, cop.ClaytonCopula,
                         n_gen, dia, method = method, random = random)
t6 = time.time()
print('done clayton diagonal 0,',t5-t6)
wd0_e, number0_e, ig0_e, avgm0_e, avgc0_e, \
histo0_e, S0_e, fd4, ws4 = hr.main(filename, startdates, n_test, cop.EmpiricalCopula,
                         n_gen, dia, method = method, random = random)
t7 = time.time()
print('done empirical diagonal 0,', t6-t7)
wd0_i, number0_i, ig0_i, avgm0_i, avgc0_i, \
histo0_i, S0_i,fd5, ws5 = hr.main(filename, startdates, n_test, cop.IndependenceCopula,
                         n_gen, dia, method = method, random = random)
t8 = time.time()
print('done independence diagonal 0,', t7-t8)
"""

wd0_s, number0_s, ig0_s, avgm0_s, avgc0_s, \
histo0_s, S0_s, fd6, ws6 = hr.main(filename, startdates, n_test, cop.StudentCopula, 
                         n_gen, dia, method = method, random = random)
"""

dia = 1

wd1_g, number1_g, ig1_g, avgm1_g, avgc1_g, \
histo1_g, S1_g, fd7, ws7  = hr.main(filename, startdates, n_test, cop.GaussianCopula,
                         n_gen, dia, method = method, random = random)
t9 = time.time()
print('done gaussian diagonal 1,', t8-t9)
wd1_gu, number1_gu, ig1_gu, avgm1_gu, avgc1_gu, \
histo1_gu, S1_gu, fd8, ws8 = hr.main(filename, startdates, n_test, cop.GumbelCopula,
                           n_gen, dia, method = method, random = random)
t10 = time.time()
print('done gumbel diagonal 1,', t9-t10)
wd1_f, number1_f, ig1_f, avgm1_f, avgc1_f, \
histo1_f, S1_f, fd9, ws9 = hr.main(filename, startdates, n_test, cop.FrankCopula, n_gen,
                         dia, method = method, random = random)
t11 = time.time()
print('done frank diagonal 1,', t10-t11)
wd1_c, number1_c, ig1_c, avgm1_c, avgc1_c, \
histo1_c, S1_c, fd10, ws10 = hr.main(filename, startdates, n_test, cop.ClaytonCopula,
                         n_gen, dia, method = method, random = random)
t12 = time.time()
print('done clayton diagonal 1,', t11-t12)
wd1_e, number1_e, ig1_e, avgm1_e, avgc1_e, \
histo1_e, S1_e, fd11, ws11 = hr.main(filename, startdates, n_test, cop.EmpiricalCopula,
                         n_gen, dia, method = method, random = random)
t13 = time.time()
print('done empirical diagonal 1,', t12-t13)
wd1_i, number1_i, ig1_i, avgm1_i, avgc1_i, \
histo1_i, S1_i, fd12, ws12 = hr.main(filename, startdates, n_test, cop.IndependenceCopula,
                         n_gen, dia, method = method, random = random)
t14 = time.time()
print('done independece diagonal 1,', t13-t14)
"""
wd1_s, number1_s, ig1_s, avgm1_s, avgc1_s, \
histo1_s, S1_s, fd13, ws13 = hr.main(filename, startdates, n_test, cop.StudentCopula, 
                         n_gen, dia, method = method, random = random)
"""

print('gaussian avg', avgm0_g, avgc0_g, avgm1_g, avgc1_g)
print('gumbel avg', avgm0_gu, avgc0_gu, avgm1_gu, avgc1_gu)
print('frank avg', avgm0_f, avgc0_f, avgm1_f, avgc1_f)
print('clayton avg', avgm0_c, avgc0_c, avgm1_c, avgc1_c)
print('empirical avg', avgm0_e, avgc0_e, avgm1_e, avgc1_e)
print('independence avg', avgm0_i, avgc0_i, avgm1_i, avgc1_i)

print('gaussian number', number0_g, number1_g)
print('gumbel number', number0_gu, number1_gu)
print('frank number', number0_f, number1_f)
print('clayton number', number0_c, number1_c)
print('empirical number', number0_e, number1_e)
print('independence number', number0_i, number1_i)

# Plotting the rank histograms of every experiment.

plt.subplot(2,3,1)

X = S0_g
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo0_g.r - 1):
        j += 1
        bool = X[i] > histo0_g.Rank[j]
    A[i] = (
           j + 0.5) / histo0_g.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo0_g.r + 1)) / histo0_g.r
C = np.asarray(range(histo0_g.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('gaussian')
plt.ylabel('probability density')

plt.subplot(2,3,2)

X = S0_gu
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo0_gu.r - 1):
        j += 1
        bool = X[i] > histo0_gu.Rank[j]
    A[i] = (
           j + 0.5) / histo0_gu.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo0_gu.r + 1)) / histo0_gu.r
C = np.asarray(range(histo0_gu.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('gumbel')

plt.subplot(2,3,3)

X = S0_f
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo0_f.r - 1):
        j += 1
        bool = X[i] > histo0_f.Rank[j]
    A[i] = (
           j + 0.5) / histo0_f.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo0_f.r + 1)) / histo0_f.r
C = np.asarray(range(histo0_f.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('frank')

plt.subplot(2,3,4)

X = S0_c
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo0_c.r - 1):
        j += 1
        bool = X[i] > histo0_c.Rank[j]
    A[i] = (
           j + 0.5) / histo0_c.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo0_c.r + 1)) / histo0_c.r
C = np.asarray(range(histo0_c.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('clayton')
plt.ylabel('probability density')

plt.subplot(2,3,5)

X = S0_e
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo0_e.r - 1):
        j += 1
        bool = X[i] > histo0_e.Rank[j]
    A[i] = (
           j + 0.5) / histo0_e.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo0_e.r + 1)) / histo0_e.r
C = np.asarray(range(histo0_e.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('empirical')

plt.subplot(2,3,6)

X = S0_i
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo0_i.r - 1):
        j += 1
        bool = X[i] > histo0_i.Rank[j]
    A[i] = (
           j + 0.5) / histo0_i.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo0_i.r + 1)) / histo0_i.r
C = np.asarray(range(histo0_i.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('independence')

#plt.savefig('test1_0.png')
#plt.savefig('Report/BPA_2012_2017/' + starts_out[0] + '-' + starts_out[1] + '-' + str(0)
#            + '.png')
plt.savefig('test_more_h/' + starts_out[0] + '-' + starts_out[1] + '-' + str(0)
            + '.png')
#plt.show()

plt.close()


# Plotting the rank histograms of every experiment.

plt.subplot(2,3,1)

X = S1_g
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo1_g.r - 1):
        j += 1
        bool = X[i] > histo1_g.Rank[j]
    A[i] = (
           j + 0.5) / histo1_g.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo1_g.r + 1)) / histo1_g.r
C = np.asarray(range(histo1_g.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('gaussian')
plt.ylabel('probability density')

plt.subplot(2,3,2)

X = S1_gu
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are 1visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo1_gu.r - 1):
        j += 1
        bool = X[i] > histo1_gu.Rank[j]
    A[i] = (
           j + 0.5) / histo1_gu.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo1_gu.r + 1)) / histo1_gu.r
C = np.asarray(range(histo1_gu.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('gumbel')

plt.subplot(2,3,3)

X = S1_f
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo1_f.r - 1):
        j += 1
        bool = X[i] > histo1_f.Rank[j]
    A[i] = (
           j + 0.5) / histo1_f.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo1_f.r + 1)) / histo1_f.r
C = np.asarray(range(histo1_f.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('frank')

plt.subplot(2,3,4)

X = S1_c
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo1_c.r - 1):
        j += 1
        bool = X[i] > histo1_c.Rank[j]
    A[i] = (
           j + 0.5) / histo1_c.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo1_c.r + 1)) / histo1_c.r
C = np.asarray(range(histo1_c.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('clayton')
plt.ylabel('probability density')

plt.subplot(2,3,5)

X = S1_e
n = len(X)
A = np.zeros(n)

for i in range(n):
    # for all data we have we look for its rank.
    # i indicates the float in input_data whose rank we evaluate
    j = -1  # j indicates the rank we are visiting
    bool = True  # bool tells us if our data is inferior to the data of rank j

    while bool and j < (histo1_e.r - 1):
        j += 1
        bool = X[i] > histo1_e.Rank[j]
    A[i] = (
           j + 0.5) / histo1_e.r  # A[i] will increase the height of the column j by one.
B = np.asarray(range(histo1_e.r + 1)) / histo1_e.r
C = np.asarray(range(histo1_e.r))

# bins are the indexes of the histograms
# each column i of the histograms has its value between bins[i] and bins[i+1]
count, bins, ignored = plt.hist(A, B, normed='True')
# count is a list with the value of each column
# count[i] will be the height of the column i
plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
plt.title('empirical')

plt.subplot(2,3,6)

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

#plt.savefig('test1_1.png')
#plt.savefig('Report/BPA_2012_2017/' + starts_out[0] + '-' + starts_out[1] + '-' + str(1)
#            + '.png')
plt.savefig('test_more_h/' + starts_out[0] + '-' + starts_out[1] + '-' + str(1)
            + '.png')
#plt.show()
plt.close()

# Writing the results into a Latex Table format

#file = open('Report/BPA_2012_2017/' + starts_out[0] + '-' + starts_out[1] + '.tex', 'w')
file = open('test_more_h/' + starts_out[0] + '-' + starts_out[1] + '.tex', 'w')

file.write('\\begin{table}[h] \n'
           '    \centering \n'
           '    \\begin{tabular}{|l|l|l|l|l|l|l|l|} \hline \n'
           '        \\textbf{Diagonal} & \multicolumn{7}{c|}{\\textbf{Copula}}'
           ' \\\\ \hline \n'
           '        & Gaussian & Gumbel & Frank & Clayton & Empirical & '
           'Independence & Student \\\\ \hline \n'
           '        $[0,0], [1,1]$ & ' + str(wd0_g) + ' & ' + str(wd0_gu) +
           ' & ' + str(wd0_f) + ' & ' + str(wd0_c) + ' & ' + str(wd0_e) +
           ' & ' + str(wd0_i) + ' &  \\\\ \hline \n'
           '        $[0,1], [1,0]$ & ' + str(wd1_g) + ' & ' + str(wd1_gu) +
           ' & ' + str(wd1_f) + ' & ' + str(wd1_c) + ' & ' + str(wd1_e) +
           ' & ' + str(wd1_i) + ' &  \\\\ \hline \n'
           '    \end{tabular} \n'
           '    \caption{Wasserstein distances for ' + str(number0_c) +
           ' observations of pairs of BPA wind forecast errors beginning '
           + startdates[0] + ', ' + startdates[1] + '. The copulas were fit '
           ' using an average of ' + str(avgc0_c) + ' observations beginning '
           + str(fd) + '. The marginals were fit using an average of '
           + str(avgm0_c) + ' observations of respective hours (i.e. the '
           ' marginals for the two hours were computed separately) with a MW '
           'segmentation filter for the marginals of ' + str(ws) + '. The '
           'data file used was ' + filename_out + '.} \n'
           '\end{table}')
file.close()

"""
file = open('test_' + t + '.txt', 'w')

file.write('Gaussian 0:' + str(wd0_g) + '\n'
           'Gumbel 0:' + str(wd0_gu) + '\n'
           'Frank 0:' + str(wd0_f) + '\n'
           'Clayton 0:' + str(wd0_c) + '\n'
           'Empirical 0:' + str(wd0_e) + '\n'
           'Independence 0:' + str(wd0_i) + '\n'
           'Gaussian 1:' + str(wd1_g) + '\n'
           'Gumbel 1:' + str(wd1_gu) + '\n'
           'Frank 1:' + str(wd1_f) + '\n'
           'Clayton 1:' + str(wd1_c) + '\n'
           'Empirical 1:' + str(wd1_e) + '\n'
           'Independence 1:' + str(wd1_i) + '\n')
file.close()
"""

t2 = time.time()

print(t2-t1)
