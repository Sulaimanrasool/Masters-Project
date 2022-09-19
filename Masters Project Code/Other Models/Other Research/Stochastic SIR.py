import random

import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm

import pickle

def sir_odes(t, x, b, g, N):
    "data = signal + noise"

    S = x[0]
    I = x[1]
    R = x[2]


    dSdt = -(b/N)*S*I
    dIdt = (b/N)*S*I - g*I
    dRdt = g*I

    "Introduction of noise based on d = s + n, where n is gaussian white noise"



    return dSdt, dIdt, dRdt

# def stochastic_var(x, std, gen):
#     """
#     This creates arrays of normal distributed variables
#
#     :param x: Array of arguments we want to make stochastic which would be mean value
#     :param e: Array of sigma for each variable
#     :param n: Number of generations
#     :return: 2D array of varaibles that have been random distributed
#     """
#
#     stochstic_array = abs(np.random.normal(x, std, (gen,len(x))))
#     return stochstic_array
#
# def stochastic_model(t_span, x_0,args,t,std,gen, N):
#
#
#     args = stochastic_var(args[:-1], std,gen)
#     number_array = numpy.full(shape = (gen,1), fill_value= N)
#     args = numpy.concatenate((args, number_array), axis=1)
#     t_len = len(t)
#     # S_array = np.zeros((gen, t_len))
#     I_array = np.zeros((gen, t_len))
#     # R_array = np.zeros((gen, t_len))
#
#
#
#     for i in range(gen):
#
#         sol = solve_ivp(sir_odes, t_span, x_0, args=(args[i]), t_eval=t)
#
#         S = sol.y[0]
#         I = sol.y[1]
#         R = sol.y[2]
#
#
#         I_array[i] = sol.y[1]
#         # S_array[i] = sol.y[0]
#         # R_array[i] = sol.y[2]
#
#
#         if i == 0:
#             # plt.plot(t, S,label='Susceptible, Black', c="black",alpha=0.1)
#             plt.plot(t, I, label='Infected, Magenta', c="m", alpha=0.1)
#             # plt.plot(t, R, label='Recovered, Green', c="g", alpha=0.1)
#             plt.plot(t, (I+R+S))
#
#         else:
#             plt.plot(t, S, c="black",alpha=0.1)
#             plt.plot(t, I, c="m", alpha=0.1)
#             plt.plot(t, R, c="g", alpha=0.1)
#
#
#
#
#     I_pos = I_array[:,5]
#     fith = np.percentile(I_array, 5, axis = 0)
#     ninefive = np.percentile(I_array, 95, axis = 0)
#     mean = np.mean(I_array, axis = 0)
#     plt.plot(t,fith, label = "Fith percentile")
#     plt.plot(t,ninefive, label = "ninty Fith percentile")
#     plt.plot(t, mean, label="Mean")
#     plt.xlabel("Time")
#     plt.ylabel("Population")
#     plt.legend()
#     plt.show()
#
#
#     mu,std = norm.fit(I_pos)
#     plt.hist(I_pos, bins = 20 ,density= True)
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x,mu,std)
#
#     """density = counts / (sum(counts) * np.diff(bins))), so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1"""
#
#     plt.plot(x,p,label = "Normal Curve, mu =" + str(round(mu,2)) + " " + "std" + " "+str(round(std,2)))
#     plt.legend()
#     plt.show()
#
#
# def data_generator(t_span, x_0,args,t):
#
#     sol = solve_ivp(sir_odes, t_span, x_0, args=(args), t_eval=t)
#     S = sol.y[0]
#     I = sol.y[1]
#     R = sol.y[2]
#
#     S_data = np.random.normal(S,0.1*S)
#     I_data = np.random.normal(I, 0.1 * I)
#     R_data = np.random.normal(R, 0.1 * R)
#
#     return I_data, I
#
# def bay_engine(t_span, x_0,args,t,std,gen, N):
#     data, signal = data_generator(t_span, x_0, args, t)
#     args = (3, 2, N)  # Arguments f
#     std = np.array([0.5*3, 0.5*2])
#     args = stochastic_var(args[:-1], std, gen)
#     number_array = numpy.full(shape=(gen, 1), fill_value=N)
#     args = numpy.concatenate((args, number_array), axis=1)
#     t_len = len(t)
#     # S_array = np.zeros((gen, t_len))
#     I_array = np.zeros((gen, t_len))
#     # R_array = np.zeros((gen, t_len))
#
#
#
#
#
#
#     for i in range(gen):
#         print(i)
#
#         sol = solve_ivp(sir_odes, t_span, x_0, args=(args[i]), t_eval=t)
#
#         S = sol.y[0]
#         I = sol.y[1]
#         R = sol.y[2]
#
#         I_array[i] = sol.y[1]
#         # S_array[i] = sol.y[0]
#         # R_array[i] = sol.y[2]
#
#         if i == 0:
#             # plt.plot(t, S,label='Susceptible, Black', c="black",alpha=0.1)
#             plt.plot(t, I, label='Infected, Magenta', c="m", alpha=0.1)
#             # plt.plot(t, R, label='Recovered, Green', c="g", alpha=0.1)
#             # plt.plot(t, (I + R + S))
#
#         else:
#             # plt.plot(t, S, c="black", alpha=0.1)
#             plt.plot(t, I, c="m", alpha=0.1)
#             # plt.plot(t, R, c="g", alpha=0.1)
#
#         if i == 0:
#             chi_value = np.sum(np.square((data - I)))
#             chi_store = chi_value
#             arg_store = args[i]
#             pos = 0
#
#
#
#         chi_value = np.sum(np.square((data-I)))
#         if chi_value < chi_store:
#             chi_store = chi_value
#             arg_store = args[i]
#             pos = i
#     print(chi_store)
#     plt.plot(t, data, label = "Data")
#     plt.plot(t, signal, label="signal")
#     plt.plot(t, I_array[pos], label = "minim chi sqaured")
#     plt.xlabel("Time,[days]")
#     plt.ylabel("Population")
#     plt.legend()
#     plt.show()
#     return data, arg_store




def chain_markov(data,t_span, x_0,t,gen, N):
    "markov process for SIR model only for now"

    b_store = []
    g_store = []
    all_chi_vals = []
    p = 0.8
    for j in range(1):
        print("Chain: ", 1)

        args = np.random.uniform(0,10,2)
        args = np.append(args, N)
        b_store.append(args[0]) # creating our lsit
        g_store.append(args[1])

        sol = solve_ivp(sir_odes, t_span, x_0, args=args, t_eval=t)

        I = sol.y[1]
        I_store = np.array(I)

        chi_value = np.sum(np.square((data - I)))
        chi_store = chi_value
        all_chi_vals.append(chi_store)

        arg_store = np.array(args)

        args[0] = np.random.normal(args[0], 0.1, 1)
        args[1] = np.random.normal(args[1], 0.1, 1)



        for i in range(gen):


            sol = solve_ivp(sir_odes, t_span, x_0, args= args, t_eval=t)

            I = sol.y[1]

            chi_value = np.sum(np.square((data - I)))
            if chi_value < chi_store:
                chi_store = chi_value
                arg_store = np.array(args)


                args[0] = np.random.normal(args[0], 0.01, 1)
                args[1] = np.random.normal(args[1], 0.01, 1)

                b_store.append(arg_store[0])  # creating our lsit
                g_store.append(arg_store[1])
                all_chi_vals.append(chi_store)

                I_store = I

            else:

                bolt = np.random.uniform(0,1)
                if bolt > p :
                    chi_store = chi_value
                    arg_store = np.array(args)


                    args[0] = np.random.normal(args[0], 0.01, 1)
                    args[1] = np.random.normal(args[1], 0.01, 1)

                    b_store.append(arg_store[0])  # creating our lsit
                    g_store.append(arg_store[1])
                    all_chi_vals.append(chi_store)
                    I_store = I
                else:
                    args[0] = np.random.normal(arg_store[0], 0.01, 1)
                    args[1] = np.random.normal(arg_store[1], 0.01, 1)



        print(arg_store)
        print((chi_store))

        plt.plot(t, I_store, label="minim chi sqaured"+ str(j))


    plt.plot(t, data, label="Data" + str(j))
    plt.legend()
    plt.show()

    return b_store, g_store , all_chi_vals





"----------------------------------------------------------------------------------------------------------------------"
"Set up of Model parameters"
t_span = np.array([0, 25])  # Time limits
t = np.linspace(t_span[0], t_span[1], t_span[1] * 10 )  # Time series
x_0 = np.array([2000, 2, 0])  # Initial conditions for model variables: S, I, R respectively
N = np.sum(x_0)
args = (1.25, 0.5, N)  # Arguments for our model parameters: \beta, \gamma, N
std = np.array([0.1*1.25, 0.1*0.5])
gen = 100

"----------------------------------------------------------------------------------------------------------------------"
# stochastic_model(t_span,x_0,args,t,std,gen, N)
#
# data , arg_store = bay_engine(t_span, x_0,args,t,std,gen, N)
# print(arg_store)
# fid = open('training_data_sir', 'wb')
# pickle.dump(data, fid)
# fid.close()


fid = open('training_data_sir', 'rb')  # Load the degree distribution
data = pickle.load(fid)
fid.close()

"----------------------------------------------------------------------------------------------------------------------"
gen = 30000
b,g,chi = chain_markov(data, t_span, x_0,t,gen, N)
# b,g = chain_markov(data, t_span, x_0,t,gen, N)

data = np.column_stack([chi, b , g])
datafile_path = "chain"
np.savetxt(datafile_path , data, fmt=['%f','%f', '%f'])


ele = np.loadtxt('chain')
print(ele)
























# model = pm.Model()
# with model:
#
#     # Define the prior of the parameter lambda.
#     sigma = pm.HalfNormal('sigma' ,sd = 10)
#
#     b_0 = pm.Normal('b_0', mu = arg_store[0], sigma= 5)
#     g_0 = pm.Normal('g_0', mu = arg_store[1], sigma= 5)
#     print(b_0, g_0)
#     arg_monte = (b_0,g_0, N)
#     sol = solve_ivp(sir_odes, t_span, x_0, args=arg_monte, t_eval=t)
#
#     I = sol.y[1]
#
#     likelihood = pm.Normal('y', mu = I, sd = sigma, observed = data )
#
#     trace = pm.sample(2000, chains = 1, tune = 2000, step = pm.Metropolis())
#
# pm.traceplot(trace)
# pm.plot_posterior(trace)