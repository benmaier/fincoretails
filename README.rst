fincoretails
============

Fit some heavy-tailed distributions to data. Has a focus on piece-wise
defined probability densities that have finite values on the core (are
nonzero and finite in the support region x <= xmin).

This should not be considered a fully functional fledged, smooth-edged
library, it just contains the bare necessities to fit some stuff. Please
feel free to fork and tinker with this project.

-  repository: https://github.com/benmaier/fincoretails/

|Example distributions of the piecewise models| |Example fits of the
generalized models| |Example fits of the forced models|

Example
-------

.. code:: python

   from fincoretails import lognormal, general_algpareto, loglikelihood_ratio, aic

   alpha = 2
   xmin = 4.5
   beta = alpha
   Nsample = 10_000
   data = general_algpareto.sample(Nsample, alpha, xmin, beta)

   alpha, xmin, beta = general_algpareto.fit_params(data)
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

With output:

.. code:: bash

   finite-core pareto: alpha=2.01, xmin=4.47, beta=1.97
   lognormal: mu=1.16, sigma=1.44
   logL finite-core pareto = -28833.94
   logL lognormal = -29474.10
   log-likelihood ratio R=11.24 with significance level p=2.65e-29
   AIC finite-core pareto = 57673.89
   AIC lognormal = 58952.20

Install
-------

.. code:: bash

   pip install fincoretails

``fincoretails`` was developed and tested for

-  Python 3.6
-  Python 3.7
-  Python 3.8

So far, the package's functionality was tested on Mac OS X and CentOS
only.

Dependencies
------------

``fincoretails`` directly depends on the following packages which will
be installed by ``pip`` during the installation process

-  ``numpy>=1.23``
-  ``scipy>=1.9``
-  ``sympy>=1.12``

Documentation
-------------

Distributions
~~~~~~~~~~~~~

Every distribution is treated as a separate module in this package.
There's one list that makes all these distributions accessible.

.. code:: python

   from fincoretails import distributions

will give you this list:

.. code:: python

   distributions = [
                # constant core (uniform distribution), power-law tail
                fincoretails.unipareto,

                # core: (2-(x/xmin)^alpha), tail: (xmin/x)^alpha
                fincoretails.algpareto,

                # core: (x/xmin)^alpha, tail: (xmin/x)^alpha
                fincoretails.powpareto,

                # core: exp[-alpha(x/xmin-1)], tail: (xmin/x)^alpha
                fincoretails.expareto,

                # core: (2-(x/xmin)^beta), tail: (xmin/x)^alpha
                fincoretails.general_algpareto,
                
                # core: (x/xmin)^beta, tail: (xmin/x)^alpha
                fincoretails.general_powpareto,

                # core: exp[-beta(x/xmin-1)], tail: (xmin/x)^alpha
                fincoretails.general_expareto,

                # log-normal as reference
                fincoretails.lognormal,

                # see Appendix D of the paper
                fincoretails.santafe,
              ]

We'll comment on each of them further below.

Each distribution module can also be imported as e.g.

.. code:: python

   from fincoretails import lognormal

   rvs = lognormal.sample(Nsample=1000,mu=1,sigma=1)

The distribution modules all contain similar functions, we'll list some
in the following.

Distribution Properties
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def sample(Nsample, alpha, xmin, beta):
       """Samples from the distribution based on parameters."""

   def quantile(q, *parameters):
       """
       Computes the quantile of the distribution for given parameters
       and probability q.
       """

   def get_normalization_constant(alpha, xmin, beta):
       """Computes the normalization constant."""

   def Pcrit(alpha, xmin, beta):
       """Computes the CDF at xmin given distribution parameters."""

   def cdf(x, alpha, xmin, beta):
       """
       Compute the cumulative distribution function (CDF)
       for single value or array-like x.
       """

   def cdf_left(x, alpha, xmin, beta):
       """
       Compute the CDF for values less than xmin.
       (single data point or array-like)
       """

   def cdf_right(x, alpha, xmin, beta):
       """
       Compute the CDF for values greater than xmin.
       (single data point or array-like)
       """

   def ccdf(x, *args, **kwargs):
       """
       Compute the complementary CDF of this distribution
       for single value or array-like x.
       """

   def pdf(x, alpha, xmin, beta):
       """
       Compute the probability density function at x
       (single data point or array-like).
       """

   def pdf_left(x, alpha, xmin, beta):
       """
       Compute the probability density function for values less than xmin.
       (single data point or array-like).
       """

   def pdf_right(x, alpha, xmin, beta):
       """
       Compute the probability density function for values greater than xmin
       (single data point or array-like).
       """

   def mean(alpha, xmin, beta):
       """Calculate the mean of the distribution"""

   def second_moment(alpha, xmin, beta):
       """Calculate the second moment of the distribution"""

   def variance(*args, **kwargs):
       """Calculate the variance of the distribution"""

   def neighbor_degree(*args, **kwargs):
       """Returns second_moment / mean"""

Distribution Fitting
^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def fit_params(data, beta_initial_values=(1.,), minxmin=None, maxxmin=None):
       """
       Fits distribution parameters based on data.

       Parameters
       ----------
       data : array-like
           Data points.
       beta_initial_values : tuple, optional
           Initial values to start searching for beta hat.
           Default is (1.,). Usually you already have an idea
           what beta might be (or even if it's gonna be negative
           or positive) from what the data looks like and
           providing a sensible guess speeds up the fitting.
       minxmin : float, optional
           Minimum xmin value. Default is None.
       maxxmin : float, optional
           Maximum xmin value. Default is None.

       Returns
       -------
       alpha_hat : float

       xmin_hat : float

       beta_hat : float
       """

   def loglikelihood(data, *parameters):
       """
       Computes the total log-likelihood of the distribution
       for given data and parameters.
       """

   def loglikelihoods(data, *parameters):
       """
       Computes the individual log-likelihood of the distribution
       for each data point (expects an array, returns an array)
       """

   def alpha_and_log_likelihood_fixed_xmin(data, xmin, beta):
       """
       Calculates alpha and log likelihood for a fixed minimum value
       and fixed beta

       Parameters
       ----------
       data : array-like
           Data for computation.
       xmin : float
           Fixed minimum value.
       beta : float, optional
           beta parameter of the distribution

       Returns
       -------
       tuple
           alpha, log-likelihood.
       """


   def alpha_beta_and_log_likelihood_fixed_xmin(data, xmin, b0=1.5):
       """
       Calculates alpha, beta, and log likelihood for a fixed minimum value.

       Parameters
       ----------
       data : array-like
           Data for computation.
       xmin : float
           Fixed minimum value.
       b0 : float, optional
           Initial value of beta. Default is 1.5.

       Returns
       -------
       tuple
           alpha, beta, and log-likelihood.
       """

   def alpha_xmin_beta_and_log_likelihood_fixed_xmin_index(data, j, xmins=None, beta0=[2.,]):
       """
       Computes alpha, xmin, beta, and log likelihood values for a fixed minimum value index.

       Parameters
       ----------
       data : array-like
           Data for computation.
       j : int
           Index of the fixed minimum value.
       xmins : array-like, optional
           Array of minimum values. If None, will consider unique sorted data values.
       beta0 : list, optional
           Initial beta values to try. Default is [2.,].

       Returns
       -------
       tuple
           alpha, xmin, beta, and log likelihood values.
       """

   def alpha_xmin_beta_and_log_likelihood(data, beta0=[2.,], stop_at_first_max=False, minxmin=None, maxxmin=None):
       """
       Computes optimal alpha, xmin, beta, and log likelihood.

       Parameters
       ----------
       data : array-like
           Data for computation.
       beta0 : list, optional
           Initial beta values. Default is [2.,].
       stop_at_first_max : bool, optional
           If True, stops the search at the first maximum value. Default is False.
       minxmin : float, optional
           Minimum boundary for xmin. Default is None.
       maxxmin : float, optional
           Maximum boundary for xmin. Default is None.

       Returns
       -------
       tuple
           Optimal alpha, xmin, beta, and log likelihood.
       """

Some other functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can do a complete fit analysis with the experimental analysis
function

.. code:: python

   from fincoretails import distributions, unipareto
   from fincoretails.analysis import analysis

   Nsample = 2000
   atrue = 2
   ytrue = 3
   data = unipareto.sample(Nsample,atrue,ytrue)
   dists = distributions[:-1] # disregard Santa Fe distribution

   analysis(data, dists)

There's also functionalities to do goodness-of-fit:

.. code:: python

   from fincoretails.tools import loglikelihood_ratio, aic

with function headers

.. code:: python

   def loglikelihood_ratio(loglikelihoodsA, loglikelihoodsB, normalized_ratio=True):
       """
       Compute the log-likelihood ratio and the significance level.
       Return the log-likelihood ratio R and the probability p
       that a random sample from a normally distributed R-value
       distribution would be larger or equal to the computed
       R-value.

       Parameters
       ----------
       loglikelihoodsA : array-like
           Log-likelihoods under null hypothesis.
       loglikelihoodsB : array-like
           Log-likelihoods under alternative hypothesis.
       normalized_ratio : bool, optional, default = True
           If True (default), return the normalized log-likelihood ratio.

       Returns
       -------
       R : float
           The log-likelihood ratio. It is normalized if `normalized_ratio` is True.
       p : float
           The significance level, i.e., the probability that a random sample
           from a normally distributed R-value distribution would be larger or
           equal to the computed R-value.

       Notes
       -----
       Typically, one classifies the R-value as 'significant' if p < 0.05. However,
       also consider the context of the problem rather than strictly following this rule.
       """

   def aic(logLL, number_of_free_parameters, nsamples=None):
       """
       Compute the Akaike Information Criterion (AIC).

       Parameters
       ----------
       logLL : float
           The log-likelihood.
       number_of_free_parameters : int
           The number of free parameters in the model.
       nsamples : int, optional, default = None
           The number of samples. If not provided or if the sample size is too small,
           the function will return the regular AIC.

       Returns
       -------
       AIC : float
           The Akaike Information Criterion.

       Notes
       -----
       If the number of samples and the number of free parameters are large enough,
       the function will return the corrected AIC.
       """

Changelog
---------

Changes are logged in a `separate
file <https://github.com/benmaier/fincoretails/blob/main/CHANGELOG.md>`__.

License
-------

This project is licensed under the `MIT
License <https://github.com/benmaier/fincoretails/blob/main/LICENSE>`__.
Note that this excludes any images/pictures/figures shown here or in the
documentation.

Contributing
------------

If you want to contribute to this project, please make sure to read the
`code of
conduct <https://github.com/benmaier/fincoretails/blob/main/CODE_OF_CONDUCT.md>`__
and the `contributing
guidelines <https://github.com/benmaier/fincoretails/blob/main/CONTRIBUTING.md>`__.
In case you're wondering about what to contribute, we're always
collecting ideas of what we want to implement next in the `outlook
notes <https://github.com/benmaier/fincoretails/blob/main/OUTLOOK.md>`__.

|Contributor Covenant|

Dev notes
---------

Fork this repository, clone it, and install it in dev mode.

.. code:: bash

   git clone git@github.com:YOURUSERNAME/fincoretails.git
   make

If you want to upload to PyPI, first convert the new ``README.md`` to
``README.rst``

.. code:: bash

   make readme

It will give you warnings about bad ``.rst``-syntax. Fix those errors in
``README.rst``. Then wrap the whole thing

.. code:: bash

   make pypi

It will probably give you more warnings about ``.rst``-syntax. Fix those
until the warnings disappear. Then do

.. code:: bash

   make upload

.. |Example distributions of the piecewise models| image:: https://github.com/benmaier/fincoretails/blob/main/cookbook/example_plots/example_plots.png?raw=true
.. |Example fits of the generalized models| image:: https://github.com/benmaier/fincoretails/blob/main/cookbook/example_plots/fit_plots_0.png?raw=true
.. |Example fits of the forced models| image:: https://github.com/benmaier/fincoretails/blob/main/cookbook/example_plots/fit_plots_1.png?raw=true
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg
   :target: code-of-conduct.md
