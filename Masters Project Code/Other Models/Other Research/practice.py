import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt



xarray = np.array([0, 1, 2, 3, 4, 5])
yarray = np.array([0, 10, 20, 30, 40, 50])
# here is your data, in two numpy arrays

data = np.column_stack([xarray, yarray])
datafile_path = "Practice data"
np.savetxt(datafile_path , data, fmt=['%d','%d'])



























# warnings.simplefilter(action="ignore", category=FutureWarning)
#
#
#
# with pm.Model() as model:
#     mu = pm.Normal("mu", mu=0, sigma=1)
#     obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))
#
#
#







#
#
# np.random.seed(7)
# S = 50
# x = np.linspace(2,10,S)
# y = 3 * x + 4
# y += np.random.normal(scale=2,size=S)
#
# with pm.Model() as model:
#     sigma = pm.HalfNormal('sigma', sd = 10)
#     a = pm.Uniform('a', 0, 10)
#     b = pm.Uniform('b', 0, 10)
#
#     y_hat = a * x + b
#
#     likelihood = pm.Normal('y', mu= y_hat, sd = sigma, observed = y)
#
#     trace = pm.sample(10000, chains = 1, tune = 1000)
#
# pm.traceplot(trace)
# pm.plot_posterior(trace)