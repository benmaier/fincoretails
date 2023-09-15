from fincoretails import lognormal, general_algpareto, loglikelihood_ratio, aic

alpha = 2
xmin = 4.5
beta = alpha
Nsample = 10_000
data = general_algpareto.sample(Nsample, alpha, xmin, beta)

alpha, xmin, beta = general_algpareto.fit_params(data, beta_initial_values=(2.5,))
mu, sigma = lognormal.fit_params(data)

ll_fcp = general_algpareto.loglikelihoods(data, alpha, xmin, beta)
ll_logn = lognormal.loglikelihoods(data, mu, sigma)
logL_fcp = ll_fcp.sum()
logL_logn = ll_logn.sum()

R, p = loglikelihood_ratio(ll_fcp, ll_logn)
AIC_fcp = aic(logL_fcp, number_of_free_parameters=3, nsamples=Nsample)
AIC_logn = aic(logL_logn, number_of_free_parameters=2, nsamples=Nsample)

print(f"finite-core pareto: {alpha=:4.2f}, {xmin=:4.2f}, {beta=:4.2f} ")
print(f"lognormal: {mu=:4.2f}, {sigma=:4.2f}")
print(f"logL finite-core pareto = {logL_fcp:4.2f}")
print(f"logL lognormal = {logL_logn:4.2f}")
print(f"log-likelihood ratio R={R:4.2f} with significance level p={p:4.2e}")
print(f"AIC finite-core pareto = {AIC_fcp:4.2f}")
print(f"AIC lognormal = {AIC_logn:4.2f}")
