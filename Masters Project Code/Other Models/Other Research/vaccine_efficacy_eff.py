import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
fig, ax = plt.subplots(figsize=(12,8), facecolor='white')


def sirv_odes(t, x, b,a, g, N):
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


def run_model(N, R_0, vac_frac, vac_ef):
    vp = N * vac_frac
    x_0 = np.array([N - vp - Inf, vp, Inf, 0])
    b = R_0 * g

    args = (b, vac_ef, g, Ntotal)  # Arguments for our model parameters: \beta, \alpha = vaccine efficacy, \gamma, N
    sol = solve_ivp(sirv_odes, t_span, x_0, args=args, t_eval=t)
    S = sol.y[0]
    V = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]
    ln_I = np.log(I)

    R_t = (1 / g) * np.diff(ln_I) / np.diff(t) + 1
    return   max(R_t)


def my_search(R_0, vac_ef,  rel_acc_wanted=0.01):
    population = N # Population suceptible
    high = N
    low = Inf
    mid = 0.5*(low+high)

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
                mid = 0.5*(low+high)
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


V = np.linspace(0,1,100)
Rhigh = 4.5*np.ones_like(V)
Rlow = 2.5*np.ones_like(V)


"Parameter Setup"
Ntotal = 1000000 # Total Number of people
Inf = 1000
t_span = np.array([0, 200])  # Time limits
t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)  # Time series
N = Ntotal - Inf  # population infectable
g = 1 / 10  # gamma variable, rate of Infected to Recovered
R_0 = np.linspace(1, 8, 60)  # R_0 values for which we need to find the percent of population that needs to be vaccinated
efficacy = [0.3, 0.5, 0.7, 0.8, 0.9, 1]  # vaccine efficacy

vac_pop = np.linspace(0,1,100)

# Inf1 = np.array([1000, 0.01*Ntotal, 0.1*Ntotal ])
#
# for Inf in Inf1:
Nsus = 0
# N = Ntotal - Inf
for idx, E in enumerate([0.3, 0.5, 0.7, 0.8, 0.9, 1]):
    R = 1 / (1 - Nsus - (1 - Nsus) * E * V)
    ax.plot(R, V, label='Theoretical Eff. {}'.format(E))

for a in efficacy:
    vac_amount = list()
    print(a)
    for R in R_0:
        vac_amount.append(my_search(R,a,0.01))

    plt.plot(R_0, vac_amount, marker='*', ls=' ', label="Efficacy " + " " + str(a))

plt.legend()
plt.grid()
plt.ylabel("Vaccinated Proportion")
plt.xlabel("Approx. Sustainable R Number")
plt.show()





