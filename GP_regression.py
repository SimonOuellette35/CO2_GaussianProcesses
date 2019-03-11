import pymc3 as pm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

ice = pd.read_csv("merged_ice_core_yearly.csv", header=26)
ice.columns = ["year", "CO2"]
ice["CO2"] = ice["CO2"].astype(np.float)

#### DATA AFTER 1958 is an average of ice core and mauna loa data, so remove it
ice = ice[ice["year"] <= 1958]
print("Number of data points:", len(ice))

fig = plt.figure(figsize=(9,4))
ax = plt.gca()

ax.plot(ice.year.values, ice.CO2.values, '.k')
ax.set_xlabel("Year")
ax.set_ylabel("CO2 (ppm)")

fig = plt.figure(figsize=(8,5))
ax = plt.gca()
ax.hist(100 * pm.Normal.dist(mu=0.0, sd=0.02).random(size=10000), 100)
ax.set_xlabel("$\Delta$ time (years)")
ax.set_title("time offset prior")

t = ice.year.values
y = ice.CO2.values

# normalize the CO2 readings prior to fitting the model
y_mu, y_sd = np.mean(y[0:50]), np.std(y)
y_n = (y - y_mu) / y_sd

# scale t to have units of centuries
t_n = t / 100

fig = plt.figure(figsize=(8,5))
ax = plt.gca()
ax.hist(pm.Gamma.dist(alpha=2, beta=0.25).random(size=10000), 100)
ax.set_xlabel("Time (centuries)")
ax.set_title("Lengthscale prior")
plt.show()

with pm.Model() as model:
    gamma = pm.Gamma("gamma", alpha=4, beta=2)
    alpha = pm.Gamma("alpha", alpha=3, beta=1)
    cov = pm.gp.cov.RatQuad(1, alpha, gamma)

    gp = pm.gp.Marginal(cov_func=cov)

    # x location uncertainty
    # - sd = 0.02 says the uncertainty on the point is about two years
    t_diff = pm.Normal("t_diff", mu=0.0, sd=0.02, shape=len(t))
    t_uncert = t_n - t_diff

    # white noise variance
    sigma = pm.HalfNormal("sigma", sd=5, testval=1)

    # y = f(x) + E
    y_ = gp.marginal_likelihood("y", X=t_uncert[:,None], y=y_n, noise=sigma)

    tr = pm.sample(1000, tune=1000, chains=2, cores=1, nuts_kwargs={"target_accept":0.95})

tnew = np.linspace(-100, 2150, 2000) * 0.01
with model:
    fnew = gp.conditional("fnew", Xnew=tnew[:,None])

with model:
    ppc = pm.sample_ppc(tr, samples=100, vars=[fnew])

samples = y_sd * ppc["fnew"] + y_mu

fig = plt.figure(figsize=(12,5))
ax = plt.gca()
pm.gp.util.plot_gp_dist(ax, samples, tnew * 100, plot_samples=True, palette="Blues")
ax.plot(t, y, "k.")
ax.set_xlim([-100, 2200])
ax.set_ylabel("CO2")
ax.set_xlabel("Year")

plt.show()