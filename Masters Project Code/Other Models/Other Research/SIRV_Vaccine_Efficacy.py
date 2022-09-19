#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
fig, ax = plt.subplots(figsize=(12,8), facecolor='white')

def seir_odes(t, x, b,a, g, N):
    "In this model, vaccinated need to begin with a certain percentage of the population, it does not increase"
    S = x[0]
    V = x[1]
    I = x[2]
    R = x[3]

    dSdt = -(b / N) * S * I
    dVdt = (-V * I * b * (1-a))/N
    dIdt = b * S * (I / N) - g * I + (V * I * b * (1-a))/N
    dRdt = g * I

    return [dSdt, dVdt, dIdt, dRdt]


Ntotal = 1000000

t_span = np.array([0,200]) # Time limits
t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)  # Time series
Inf = 0.3 * Ntotal # initial number that is infected
N = Ntotal -  Inf # population infectable

g = 1/10 # gamma variable, rate of Infected to Recovered
R_0 = np.linspace(1,8,60) # R_0 values for which we need to find the percent of population that needs to be vaccinated
# efficacy = [0.3,0.5,0.7,0.8,0.9,1] # vaccine efficacy
efficacy = [0.8]
vac_pop1 = np.linspace(0,0.49,30)
vac_pop2 = np.linspace(0.5,1,100)
vac_pop = np.concatenate((vac_pop1,vac_pop2), axis= 0 )


V = np.linspace(0,1,100)
Rhigh = 4.5*np.ones_like(V)
Rlow = 2.5*np.ones_like(V)
Nsus = 0
# N = Ntotal - Inf
for idx, E in enumerate([0.3, 0.5, 0.7, 0.8, 0.9, 1]):
    R = 1 / (1 - Nsus - (1 - Nsus) * E * V)
    ax.plot(R, V, label='Theoretical Eff. {}'.format(E))


for a in efficacy:
    vac_amount = list()
    R_0_amount = list()
    print(a)
    for r in R_0:
        for ele in vac_pop:
            vp = N * ele
            x_0 = np.array([N - vp - Inf , vp, Inf, 0])
            b = r * g

            args = (b, a, g, N + Inf)  # Arguments for our model parameters: \beta, \alpha = vaccine efficacy, \gamma, N
            sol = solve_ivp(seir_odes, t_span, x_0, args=args, t_eval=t)
            S = sol.y[0]
            V = sol.y[1]
            I = sol.y[2]
            R = sol.y[3]
            ln_I = np.log(I)


            R_t = (1/g) * np.diff(ln_I)/ np.diff(t) + 1

            if max(R_t) <= 1:
                vac_amount.append(ele)
                R_0_amount.append(r)
                break



    plt.plot( R_0_amount,vac_amount, marker = '*' , ls = ' ', label = "Efficacy " + " " + str(a))

plt.grid()
plt.legend()
plt.ylabel("Vaccinated Proportion")
plt.xlabel("Approx. Sustainable R Number")
plt.show()


def run_model(population, R_0, vac_frac, vac_ef):


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



x_0 = np.array([4000, 5000,1000, 0])  # Initial conditions for model variables: S, V, I, R respectively


args = (0.4, 0.006 , 0.6, 0.1, np.sum(x_0))  # Arguments for our model parameters: \beta, v, \alpha = vaccine efficacy, \gamma, N

sol = solve_ivp(seir_odes, t_span, x_0, args=args, t_eval = t)

R_0 = args[0]/args[1]
S = sol.y[0]
V = sol.y[1]
I = sol.y[2]
R = sol.y[3]
N = S + V + I + R
vaccinated_prop = (V / N)
ln_I = np.log(I)
I_m = (R+V) / N
R_t = [((1/args[3])*((ln_I[i + 1] - ln_I[i])/(t[i+1] -t[i])) + 1) for i in range(len(I)-1)]
print(R_t)
(1/(1-I_m[i]))*


plt.figure(1)
plt.plot(t, S, label='Susceptible')
plt.plot(t, V, label='Vaccinated')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, N, label ='Population')
plt.xlabel('Time [days]')
plt.ylabel('Number')
plt.legend()

#print("R_0 is:", round(R_0, 3))
print("End-population:", int(round(N[-1],0)))
print("Efficacy of Vaccine:", (1-args[2])*100,"%")
plt.figure(2)
# plt.plot(vaccinated_prop[:t_span[1]*10], R_t)
plt.plot(t[:2000], R_t)

plt.show()

a = [1,2,3]
b = -1*np.array(a)
print(b)