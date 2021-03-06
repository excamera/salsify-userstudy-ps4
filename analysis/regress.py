#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm

data = np.loadtxt('salsify-user-study-ps4.csv', delimiter=',')

y = data[:,5]

# dmin, dmax = min(data[:,2]), max(data[:,2])
# data[:,2] = (data[:,2] - dmin) / (dmax - dmin)

# dmin, dmax = min(data[:,4]), max(data[:,4])
# data[:,4] = (data[:,4] - dmin) / (dmax - dmin)

#x = data[:,2]
#x = data[:,4]
x = data[:,2:5:2]
x[:,0] = 50*x[:,0] + 50
print(x)
x = sm.add_constant(x)

results = sm.OLS(endog=y, exog=x).fit()

print(results.summary())
print(results.params)

# exact_count = 0
# one_off_count = 0
# fun = lambda x: 3.2136 + -3.7337*x[0] + 0.8454*x[1] 
# for i in range(len(y)):
#     xx = x[i,:]
#     yy = y[i]
    
#     res = abs(yy - fun(xx[1:]))
#     if res < 0.5:
#         exact_count += 1
#     elif res < 1.5:
#         one_off_count += 1

# print('total', len(y))
# print('exact', exact_count, exact_count/len(y))
# print('off_by_one', one_off_count, one_off_count/len(y))
# print('less_than_equal_one', exact_count+one_off_count, (exact_count+one_off_count)/len(y))
    
