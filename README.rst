fincoretails
============

Fit some heavy-tailed distributions to data. Has a focus on piece-wise
defined probability densities that have finite values on the core (are
nonzero and finite in the support region x <= xmin).

This should not be considered a fully functional fledged, smooth-edged
library, it just contains the bare necessities to fit some stuff. Please
feel free to fork and tinker with this project.

-  repository: https://github.com/benmaier/fincoretails/

.. code:: python

   from fincoretails import lognormal, fincorepareto, loglikelihood_ratio, aic

   alpha = 2
   xmin = 4.5
   beta = alpha
   Nsample = 10_000
   data = fincorepareto.sample(Nsample, alpha, xmin, beta)

   alpha, xmin, beta = fincorepareto.fit_params(data)
   mu, sigma = lognormal.fit_params(data)

   ll_fcp = fincorepareto.loglikelihoods(data, alpha, xmin, beta)
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

   finite-core pareto: alpha=2.00, xmin=4.50, beta=2.04
   lognormal: mu=1.18, sigma=1.43
   logL finite-core pareto = -29026.04
   logL lognormal = -29568.56
   log-likelihood ratio R=12.28 with significance level p=1.15e-34
   AIC finite-core pareto = 58058.07
   AIC lognormal = 59141.11

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

Sorry. No docs really, no time. Look at the code and the docstrings.

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

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg
   :target: code-of-conduct.md
