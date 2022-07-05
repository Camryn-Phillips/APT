#%%
import copy
import numpy as np
import astropy.units as u
import pint.models as pm
import pint.toa as pt
import pint.fitter as pf
import pint.residuals as pr
import pint.simulation as ps
import pint.logging as pl

pl.setup()


def make_random_models(fitter, Nmodels=100, params="all"):
    cov_matrix = fitter.parameter_covariance_matrix
    # this is a list of the parameter names in the order they appear in the covariance matrix
    param_names = cov_matrix.get_label_names(axis=0)
    # this is a dictionary with the parameter values, but it might not be in the same order
    # and it leaves out the Offset parameter
    param_values = fitter.model.get_params_dict("free", "value")
    mean_vector = np.array([param_values[x] for x in param_names if x != "Offset"])
    if params == "all":
        # remove the first column and row (absolute phase)
        if param_names[0] == "Offset":
            cov_matrix = cov_matrix.get_label_matrix(param_names[1:])
            fac = fitter.fac[1:]
            param_names = param_names[1:]
        else:
            fac = fitter.fac
    else:
        # only select some parameters
        # need to also select from the fac array and the mean_vector array
        idx, labels = cov_matrix.get_label_slice(params)
        cov_matrix = cov_matrix.get_label_matrix(params)
        index = idx[0].flatten()
        fac = fitter.fac[index]
        # except mean_vector does not have the 'Offset' entry
        # so may need to subtract 1
        if param_names[0] == "Offset":
            mean_vector = mean_vector[index - 1]
        else:
            mean_vector = mean_vector[index]
        param_names = cov_matrix.get_label_names(axis=0)

    f_rand = copy.deepcopy(fitter)
    # scale by fac
    mean_vector = mean_vector * fac
    scaled_cov_matrix = ((cov_matrix.matrix * fac).T * fac).T
    random_models = []
    for _ in range(Nmodels):
        # create a set of randomized parameters based on mean vector and covariance matrix
        # dividing by fac brings us back to real units
        rparams_num = (
            np.random.multivariate_normal(mean_vector, scaled_cov_matrix) / fac
        )
        rparams = dict(zip(param_names, rparams_num))
        f_rand.set_params(rparams)
        random_models.append(f_rand.model)
        f_rand = copy.deepcopy(fitter)  # is this necessary?
    return random_models


def compute_random_model_phases(fitter, models, toas, return_time=False):
    Nmodels = len(models)
    Nmjd = len(toas)
    phases_i = np.zeros((Nmodels, Nmjd))
    phases_f = np.zeros((Nmodels, Nmjd))
    for ii, model in enumerate(models):
        phase = model.phase(toas, abs_phase=True)
        # return phase
        # raise Exception("test stop")
        phases_i[ii] = phase.int
        phases_f[ii] = phase.frac
    phases = phases_i + phases_f
    phases0 = fitter.model.phase(toas, abs_phase=True)
    dphase = phases - (phases0.int + phases0.frac)
    if return_time:
        r = pr.Residuals(toas, fitter.model)
        dphase /= r.get_PSR_freq(calctype="taylor")
    return dphase


def compute_random_model_resids(fitter, models, toas, return_time=False):
    resids = np.zeros((len(models), toas.ntoas), dtype=np.float)
    r0 = pr.Residuals(tnew, fitter.model, subtract_mean=False)
    for ii, model in enumerate(models):
        rn = pr.Residuals(tnew, model, subtract_mean=False)
        resids[ii] = (
            rn.time_resids - r0.time_resids
            if return_time
            else rn.phase_resids - r0.phase_resids
        )
    return resids << u.s if return_time else resids


m, t = pm.get_model_and_toas("1748-2446Y.par", "Ter5Y.tim", usepickle=True)
# Must do a fit to get parameter covariance matrix
# f = pf.DownhillWLSFitter(t, m)
f = pf.WLSFitter(t, m)
f.fit_toas()
#%%


#%%

import time

# make fake TOAs covering more than the full range
tic = time.perf_counter()
tnew = ps.make_fake_toas_uniform(
    t.get_mjds().min().value - 500,
    t.get_mjds().max().value + 500,
    1000,
    freq=2000.0 * u.MHz,
    model=f.model,
)
toc = time.perf_counter()
print(f"Creating the fake TOAs took {toc-tic:0.4f} sec")

# now make random models
tic = time.perf_counter()
rms = make_random_models(f, 20)
toc = time.perf_counter()
print(f"Making the random models took {toc-tic:0.4f} sec")

tic = time.perf_counter()
rrs = compute_random_model_phases(f, rms, tnew, return_time=True).to(u.us)
toc = time.perf_counter()
print(f"Computing the random resids1 took {toc-tic:0.4f} sec")

tic = time.perf_counter()
rrs2 = compute_random_model_resids(f, rms, tnew, return_time=True).to(u.us)
toc = time.perf_counter()
print(f"Computing the random resids2 took {toc-tic:0.4f} sec")


#%%

tic = time.perf_counter()
dmat, dmat_lab, dmat_units = m.designmatrix(tnew, incfrozen=False, incoffset=False)
toc = time.perf_counter()
print(f"Computing the design matrix took {toc-tic:0.4f} sec")

#%%

tic = time.perf_counter()
fastrs = np.zeros((len(rms), tnew.ntoas), dtype=np.float)
dparam = np.zeros(len(dmat_lab), dtype=np.float128)
for ii, rm in enumerate(rms):
    # Do the differencing
    for jj, pp in enumerate(dmat_lab):
        if type(getattr(m, pp)) == pm.parameter.MJDParameter:
            dparam[jj] = (getattr(m, pp).value - getattr(rm, pp).value) * u.d
        else:
            dparam[jj] = getattr(m, pp).quantity - getattr(rm, pp).quantity
    fastrs[ii] = np.dot(dparam, dmat.T)
fastrs <<= u.s
toc = time.perf_counter()
print(f"Computing the random resids via design matrix {toc-tic:0.4f} sec")

#%%

import matplotlib.pyplot as plt

# Turn on support for plotting quantities
from astropy.visualization import quantity_support

quantity_support()
fig, ax = plt.subplots(figsize=(8, 4.5))
# plot the random models
for ii in range(len(rms)):
    ax.plot(tnew.get_mjds(), rrs[ii], "k-", alpha=0.2)
    # ax.plot(tnew.get_mjds(), rrs2[ii], "bx", alpha=0.2)
    ax.plot(tnew.get_mjds(), fastrs[ii], "r-", alpha=0.2)
# plot the best-fit model resids
# ax.errorbar(t.get_mjds(), f.resids.time_resids.to(u.us), t.get_errors().to(u.us), fmt="x")

ax.set_title(f"{m.PSR.value} Timing Residuals")
ax.set_xlabel("MJD")
ax.set_ylabel("Residual (us)")
ax.grid()
plt.show()

# %%
