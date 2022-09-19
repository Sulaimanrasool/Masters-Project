import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
fig, ax = plt.subplots(figsize=(12,8), facecolor='white')

V = np.linspace(0,1,100)
Rhigh = 4.5*np.ones_like(V)
Rlow = 2.5*np.ones_like(V)

N =0.24
cols = ['C0','C1','C2','C3','C4','C5']
for idx, E in enumerate([0.3,0.5,0.7,0.8]):

     R = 1/(1-N-(1-N)*E*V)
     print(E)
     ax.plot(V*100,R,label='Eff. {}'.format(E),c=cols[idx])

ax.fill_between(V*100,Rlow,Rhigh,color='k',alpha=0.3, label='Basic R')
ax.axvline(68,c='k',ls='--',alpha=0.5,zorder=-10)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.tick_params(which='both')
ax.set_xlabel('Vaccinated Fraction [%]')
ax.set_ylabel('Approx. Sustainable R Number')
ax.legend(ncol=2,title='HQ SJC(UK) IMG\n{}% naturallyinfected'.format(N*100))
ax.set_xlim(left=0,right=100)
ax.grid(which='both')
plt.tight_layout()
plt.show()




#
#

def my_search(R_0, vac_ef, rel_acc_wanted=0.01):
    population = 1000e3
    high = 998e3
    low = 1e3
    mid = 0.5(low+high)

    vac_frac = high/population
    max_R_t = run_model(population, R_0, vac_frac, vac_ef)

    if max_R_t > 1:
        return None
    else:
        vac_frac = mid/population
        max_R_t = run_model(population, R_0, vac_frac, vac_ef)
        rel_acc = abs(max_R_t - 1)
        while rel_acc > rel_acc_wanted:
            if max_R_t > 1:
                low = mid
                mid = 0.5(low+high)
                vac_frac = mid/population
                max_R_t = run_model(population, R_0, vac_frac, vac_ef)
                rel_acc = abs(max_R_t - 1)
            else:
                high = mid
                mid = 0.5*(low+high)
                vac_frac = mid/population
                max_R_t = run_model(population, R_0, vac_frac, vac_ef)
                rel_acc = abs(max_R_t - 1)
        return vac_frac


#R_t = [((1 / g) * ((ln_I[i + 1] - ln_I[i]) / (t[i + 1] - t[i])) + 1) for i in range(len(I) - 1)]