#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import seaborn as sns
# import pandas as pd

# sns.set(color_codes=True)
# sns.set_style("white")

matplotlib.rcParams.update({'font.size': 15})
plt.figure(figsize=(10,5))

data = np.loadtxt('salsify-user-study-ps4.csv', delimiter=',')

# dmin, dmax = min(data[:,1]), max(data[:,1])
# data[:,1] = (data[:,1] - dmin) / (dmax - dmin)

# dmin, dmax = min(data[:,3]), max(data[:,3])
# data[:,3] = (data[:,3] - dmin) / (dmax - dmin)

y = data[:,5]

# print('plotting delay')
# x = 33*data[:,1]
# plot_delay(x, y, data[:,3], 'delay.png')
# plot_delay(x, y, data[:,3], 'delay.svg')

#print('plotting quality')
#x = data[:,3]
#plot_quality(x, y, 'quality.png')
#plot_quality(x, y, 'quality.svg')

x = data[:,2:5:2]
x[:,0] = 50*x[:,0] + 50
x = sm.add_constant(x)
results = sm.OLS(endog=y, exog=x).fit()
print(results.params)
print(results.summary())

# groupings
delays = [1,5,10,20]

mean, std = [],[]
for d in delays:
    delay = 50*d + 50
    mean_ = []
    std_ = []

    for quality in [8, 11, 14]:
        select_y = []
        for j in range(len(y)):
            if abs(x[j,1] - delay)<10 and abs(x[j,2] - quality) < 2:
                select_y.append(y[j])
            
        mean_.append( np.mean(select_y) )
        std_.append( np.std(select_y) )
        
    mean.append(mean_)
    std.append(std_)

first = True
ebar = []
padding = [-25,0,25]
for d,m,s in zip(delays, mean, std):
    c = 50*d + 50
    for p_,m_,s_,q,color in zip(padding, m, s,[10,14,18],['#4c72b0', '#55a868', '#c44e52']):

        plot = plt.errorbar([c+p_], [m_], yerr=[s_],
                         fmt='s', color=color, ecolor=color, capsize=2, capthick=2, lw=2,
                         label='mean ± std')

        if first:
            plt.text(c+p_-40, m_+s_+0.15,str(q), fontsize=9,color=color)

        ebar.append(plot)
        
    if first:
        first = False
        
# lines of best fit
q = [8, 11, 14]
x = [0, 1150]
#x = [0, 18000]
lines = []
for qq,color in zip(q,['#4c72b0', '#55a868', '#c44e52']):
    y = [results.params[0] + x[0]*results.params[1] + qq*results.params[2], results.params[0] + x[1]*results.params[1] + qq*results.params[2]]
    pp = plt.plot(x, y, color=color,label='regression line')
    lines.append(pp)
    
#print(mean, std)

patch = mpatches.Patch(color='white', label='R² = ' + str(round(results.rsquared,2)))
#plt.legend(handles=[p, lines[1][0], patch], labels=['mean ± std', '-x/'+str(round(-1/results.params[1],2))+' + ' + str(round(results.params[0] + q[1]*results.params[2],2)), 'R² = ' + str(round(results.rsquared,3))])
plt.legend(handles=[ebar[1], lines[1][0], patch], labels=['mean ± std', 'best-fit QoE model', 'R² = ' + str(round(results.rsquared,3))])

# add labels for the groupings
c = 2.7
x = 45
plt.plot([x,x+125],[c,c],lw=1,color=(0.33,0.33,0.33))
plt.plot([x,x],[c,c+.1],lw=1,color=(0.33,0.33,0.33))
plt.plot([x+125,x+125],[c,c+.1],lw=1,color=(0.33,0.33,0.33))

plt.text(x-10, c-.55, 'Video Quality\n  (SSIM dB)', fontsize=12, color=(0,0,0))

plt.yticks([1,2,3,4,5])
#plt.xticks(list(map(lambda x: 66*x+250, [1,15,30,60])))
plt.xticks([100,300,550,1050])

plt.axis([-50, 1200, 0.75, 5.75])
#plt.axis([-100, 20000, -20, 6.25])
#plt.title('QoE User Study (Video Call)')
plt.ylabel('QoE score', labelpad=10)
plt.xlabel('Video Delay (ms)', labelpad=10)
plt.tight_layout()
plt.savefig('delay.png')
plt.savefig('delay.svg')

