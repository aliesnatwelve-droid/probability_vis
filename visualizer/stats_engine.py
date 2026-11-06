"""
Statistical computation engine for probability distributions and analysis.

Provides functions for PDF, CDF, random sampling, and statistical calculations
across a wide range of continuous and discrete distributions.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import (
    norm, uniform, expon, poisson, binom, bernoulli, geom, gamma, beta,
    chi2, t as t_dist, f as f_dist, lognorm, weibull_min, pareto, nbinom,
    hypergeom, rayleigh, laplace, cauchy
)

# ---------------------------------------------------------------------------
# Distribution registry — metadata for every supported distribution
# ---------------------------------------------------------------------------
DISTRIBUTIONS = {
    'normal': {
        'name': 'Normal (Gaussian)',
        'type': 'continuous',
        'params': [
            {'name': 'mu', 'label': 'Mean (μ)', 'default': 0.0, 'min': -10.0, 'max': 10.0, 'step': 0.1},
            {'name': 'sigma', 'label': 'Std Dev (σ)', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
        ],
        'formula': r'f(x|\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}',
        'description': 'The most widely used distribution in statistics. Describes data that clusters around a mean.',
        'applications': 'Height/weight measurements, test scores, measurement errors, natural phenomena.',
        'slug': 'normal',
    },
    'uniform': {
        'name': 'Uniform',
        'type': 'continuous',
        'params': [
            {'name': 'a', 'label': 'Lower bound (a)', 'default': 0.0, 'min': -10.0, 'max': 10.0, 'step': 0.5},
            {'name': 'b', 'label': 'Upper bound (b)', 'default': 1.0, 'min': -10.0, 'max': 10.0, 'step': 0.5},
        ],
        'formula': r'f(x|a,b) = \frac{1}{b-a} \quad \text{for } a \le x \le b',
        'description': 'All outcomes are equally likely within the range [a, b].',
        'applications': 'Random number generation, Monte Carlo simulation, dice rolls.',
        'slug': 'uniform',
    },
    'exponential': {
        'name': 'Exponential',
        'type': 'continuous',
        'params': [
            {'name': 'rate', 'label': 'Rate (λ)', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
        ],
        'formula': r'f(x|\lambda) = \lambda e^{-\lambda x} \quad \text{for } x \ge 0',
        'description': 'Models waiting times between events in a Poisson process.',
        'applications': 'Queueing theory, reliability analysis, radioactive decay.',
        'slug': 'exponential',
    },
    'poisson': {
        'name': 'Poisson',
        'type': 'discrete',
        'params': [
            {'name': 'lam', 'label': 'Rate (λ)', 'default': 3.0, 'min': 0.1, 'max': 50.0, 'step': 0.5},
        ],
        'formula': r'P(X=k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}',
        'description': 'Models the number of events in a fixed interval of time or space.',
        'applications': 'Call center arrivals, website traffic, rare diseases, insurance claims.',
        'slug': 'poisson',
    },
    'binomial': {
        'name': 'Binomial',
        'type': 'discrete',
        'params': [
            {'name': 'n', 'label': 'Trials (n)', 'default': 10, 'min': 1, 'max': 100, 'step': 1, 'integer': True},
            {'name': 'p', 'label': 'Prob. (p)', 'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        ],
        'formula': r'P(X=k|n,p) = \binom{n}{k} p^k (1-p)^{n-k}',
        'description': 'Models the number of successes in n independent trials with success probability p.',
        'applications': 'A/B testing, quality control, election polling, coin flips.',
        'slug': 'binomial',
    },
    'bernoulli': {
        'name': 'Bernoulli',
        'type': 'discrete',
        'params': [
            {'name': 'p', 'label': 'Prob. (p)', 'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        ],
        'formula': r'P(X=k|p) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0,1\}',
        'description': 'Models a single trial with two possible outcomes (success/failure).',
        'applications': 'Coin flips, pass/fail tests, click-through rates, binary classification.',
        'slug': 'bernoulli',
    },
    'geometric': {
        'name': 'Geometric',
        'type': 'discrete',
        'params': [
            {'name': 'p', 'label': 'Prob. (p)', 'default': 0.3, 'min': 0.01, 'max': 1.0, 'step': 0.05},
        ],
        'formula': r'P(X=k|p) = (1-p)^{k-1} p \quad \text{for } k = 1,2,3,\dots',
        'description': 'Models the number of trials needed to get the first success.',
        'applications': 'Waiting times, number of attempts until first win, failure analysis.',
        'slug': 'geometric',
    },
    'gamma': {
        'name': 'Gamma',
        'type': 'continuous',
        'params': [
            {'name': 'alpha', 'label': 'Shape (α)', 'default': 2.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
            {'name': 'beta', 'label': 'Rate (β)', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
        ],
        'formula': r'f(x|\alpha,\beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}',
        'description': 'A flexible distribution for non-negative continuous data, generalizing the exponential.',
        'applications': 'Rainfall amounts, insurance claims, queueing times, Bayesian priors.',
        'slug': 'gamma',
    },
    'beta': {
        'name': 'Beta',
        'type': 'continuous',
        'params': [
            {'name': 'a', 'label': 'Alpha (α)', 'default': 2.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
            {'name': 'b', 'label': 'Beta (β)', 'default': 2.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
        ],
        'formula': r'f(x|\alpha,\beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha,\beta)}',
        'description': 'Models probabilities and proportions on [0, 1]. Very flexible shapes.',
        'applications': 'Bayesian statistics, A/B testing conversion rates, project completion times.',
        'slug': 'beta',
    },
    'chi2': {
        'name': 'Chi-Square (χ²)',
        'type': 'continuous',
        'params': [
            {'name': 'df', 'label': 'Degrees of freedom', 'default': 3, 'min': 1, 'max': 30, 'step': 1, 'integer': True},
        ],
        'formula': r'f(x|k) = \frac{x^{k/2-1} e^{-x/2}}{2^{k/2} \Gamma(k/2)} \quad \text{for } x>0',
        'description': 'Sum of squared independent standard normal variables.',
        'applications': 'Chi-square test for independence, goodness of fit, variance estimation.',
        'slug': 'chi2',
    },
    'lognormal': {
        'name': 'Log-Normal',
        'type': 'continuous',
        'params': [
            {'name': 'mu', 'label': 'Mean (log-μ)', 'default': 0.0, 'min': -3.0, 'max': 3.0, 'step': 0.1},
            {'name': 'sigma', 'label': 'Std Dev (log-σ)', 'default': 0.5, 'min': 0.1, 'max': 3.0, 'step': 0.1},
        ],
        'formula': r'f(x|\mu,\sigma) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x-\mu)^2}{2\sigma^2}}',
        'description': 'A distribution of a variable whose logarithm is normally distributed.',
        'applications': 'Stock prices, income distribution, chemical concentrations, human metabolism.',
        'slug': 'lognormal',
    },
    'weibull': {
        'name': 'Weibull',
        'type': 'continuous',
        'params': [
            {'name': 'shape', 'label': 'Shape (k)', 'default': 1.5, 'min': 0.1, 'max': 5.0, 'step': 0.1},
            {'name': 'scale', 'label': 'Scale (λ)', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
        ],
        'formula': r'f(x|k,\lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}',
        'description': 'Models time-to-event data with varying hazard rates.',
        'applications': 'Survival analysis, failure times, weather forecasting, wind speed modeling.',
        'slug': 'weibull',
    },
    'student_t': {
        'name': "Student's t",
        'type': 'continuous',
        'params': [
            {'name': 'df', 'label': 'Degrees of freedom', 'default': 5, 'min': 1, 'max': 30, 'step': 1, 'integer': True},
        ],
        'formula': r'f(t|\nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}',
        'description': 'Similar to normal but with heavier tails. Becomes standard normal as df → ∞.',
        'applications': 't-tests, confidence intervals with small samples, robust statistics.',
        'slug': 'student_t',
    },
    'f_dist': {
        'name': 'F-distribution',
        'type': 'continuous',
        'params': [
            {'name': 'd1', 'label': 'DF1 (numerator)', 'default': 5, 'min': 1, 'max': 50, 'step': 1, 'integer': True},
            {'name': 'd2', 'label': 'DF2 (denominator)', 'default': 20, 'min': 1, 'max': 50, 'step': 1, 'integer': True},
        ],
        'formula': r'f(F|d_1,d_2) = \frac{\sqrt{\frac{(d_1 F)^{d_1} d_2^{d_2}}{(d_1 F+d_2)^{d_1+d_2}}}}{F\,B(d_1/2,d_2/2)}',
        'description': 'Distribution of ratios of two independent chi-square variables.',
        'applications': 'ANOVA, comparing variances, regression model significance.',
        'slug': 'f_dist',
    },
    'negative_binomial': {
        'name': 'Negative Binomial',
        'type': 'discrete',
        'params': [
            {'name': 'r', 'label': 'Successes (r)', 'default': 5, 'min': 1, 'max': 30, 'step': 1, 'integer': True},
            {'name': 'p', 'label': 'Prob. (p)', 'default': 0.5, 'min': 0.01, 'max': 1.0, 'step': 0.05},
        ],
        'formula': r'P(X=k|r,p) = \binom{k+r-1}{r-1} (1-p)^k p^r',
        'description': 'Models the number of failures before r successes.',
        'applications': 'Overdispersed count data, accident rates, disease incidence.',
        'slug': 'negative_binomial',
    },
    'rayleigh': {
        'name': 'Rayleigh',
        'type': 'continuous',
        'params': [
            {'name': 'scale', 'label': 'Scale (σ)', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1},
        ],
        'formula': r'f(x|\sigma) = \frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)} \quad \text{for } x \ge 0',
        'description': 'Distribution of the magnitude of a 2D vector with independent normal components.',
        'applications': 'Wind speed modeling, signal processing, wireless communications.',
        'slug': 'rayleigh',
    },
    'laplace': {
        'name': 'Laplace (Double Exponential)',
        'type': 'continuous',
        'params': [
            {'name': 'mu', 'label': 'Location (μ)', 'default': 0.0, 'min': -5.0, 'max': 5.0, 'step': 0.1},
            {'name': 'b', 'label': 'Scale (b)', 'default': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1},
        ],
        'formula': r'f(x|\mu,b) = \frac{1}{2b} e^{-|x-\mu|/b}',
        'description': 'Symmetric with sharper peak and heavier tails than normal.',
        'applications': 'Robust regression (L1), signal denoising, Bayesian sparse models.',
        'slug': 'laplace',
    },
    'cauchy': {
        'name': 'Cauchy (Lorentzian)',
        'type': 'continuous',
        'params': [
            {'name': 'x0', 'label': 'Location (x₀)', 'default': 0.0, 'min': -5.0, 'max': 5.0, 'step': 0.5},
            {'name': 'gamma', 'label': 'Scale (γ)', 'default': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1},
        ],
        'formula': r'f(x|x_0,\gamma) = \frac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}',
        'description': 'Heavy-tailed distribution with undefined mean and variance.',
        'applications': 'Spectroscopy, financial modeling, robust estimation, physics.',
        'slug': 'cauchy',
    },
}

# Map slugs to scipy.stats objects
SCIPY_MAP = {
    'normal': norm,
    'uniform': uniform,
    'exponential': expon,
    'poisson': poisson,
    'binomial': binom,
    'bernoulli': bernoulli,
    'geometric': geom,
    'gamma': gamma,
    'beta': beta,
    'chi2': chi2,
    'lognormal': lognorm,
    'weibull': weibull_min,
    'student_t': t_dist,
    'f_dist': f_dist,
    'negative_binomial': nbinom,
    'rayleigh': rayleigh,
    'laplace': laplace,
    'cauchy': cauchy,
}


def get_distribution(slug):
    """Return the scipy distribution object for a given slug."""
    return SCIPY_MAP.get(slug)


def _build_params(slug, **kwargs):
    """
    Convert user-facing param names to scipy-compatible arguments.
    Uses a plain if/elif chain — no lambdas (avoids bytecode cache issues).
    Returns (dist_obj, fitted_args).
    """
    dist = get_distribution(slug)
    if dist is None:
        raise ValueError(f"Unknown distribution: {slug}")

    def _f(k):
        return float(kwargs.get(k, 0))

    def _i(k):
        return int(kwargs.get(k, 1))

    scipy_args = {}

    if slug == 'normal':
        scipy_args = {'loc': _f('mu'), 'scale': _f('sigma')}
    elif slug == 'uniform':
        a, b = _f('a'), _f('b')
        scipy_args = {'loc': a, 'scale': b - a}
    elif slug == 'exponential':
        scipy_args = {'scale': 1.0 / _f('rate')}
    elif slug == 'poisson':
        scipy_args = {'mu': _f('lam')}
    elif slug == 'binomial':
        scipy_args = {'n': _i('n'), 'p': _f('p')}
    elif slug == 'bernoulli':
        scipy_args = {'p': _f('p')}
    elif slug == 'geometric':
        scipy_args = {'p': _f('p')}
    elif slug == 'gamma':
        scipy_args = {'a': _f('alpha'), 'scale': 1.0 / _f('beta')}
    elif slug == 'beta':
        scipy_args = {'a': _f('a'), 'b': _f('b')}
    elif slug == 'chi2':
        scipy_args = {'df': _i('df')}
    elif slug == 'lognormal':
        mu, sigma = _f('mu'), _f('sigma')
        scipy_args = {'s': sigma, 'scale': np.exp(mu)}
    elif slug == 'weibull':
        scipy_args = {'c': _f('shape'), 'scale': _f('scale')}
    elif slug == 'student_t':
        scipy_args = {'df': _i('df')}
    elif slug == 'f_dist':
        scipy_args = {'dfn': _i('d1'), 'dfd': _i('d2')}
    elif slug == 'negative_binomial':
        scipy_args = {'n': _i('r'), 'p': _f('p')}
    elif slug == 'rayleigh':
        scipy_args = {'scale': _f('scale')}
    elif slug == 'laplace':
        scipy_args = {'loc': _f('mu'), 'scale': _f('b')}
    elif slug == 'cauchy':
        scipy_args = {'loc': _f('x0'), 'scale': _f('gamma')}

    return dist, scipy_args


def pdf(slug, x_values=None, **params):
    """Compute PDF/PMF over given x values."""
    dist, args = _build_params(slug, **params)
    if x_values is None:
        x_values = np.linspace(-4, 4, 500)
    y = dist.pdf(x_values, **args) if hasattr(dist, 'pdf') else dist.pmf(x_values, **args)
    return np.array(x_values), np.array(y)


def cdf(slug, x_values=None, **params):
    """Compute CDF over given x values."""
    dist, args = _build_params(slug, **params)
    if x_values is None:
        x_values = np.linspace(-4, 4, 500)
    y = dist.cdf(x_values, **args)
    return np.array(x_values), np.array(y)


def sample(slug, n_samples=1000, **params):
    """Generate random samples from a distribution."""
    dist, args = _build_params(slug, **params)
    return dist.rvs(size=int(n_samples), **args)


def descriptive_stats(data):
    """
    Compute descriptive statistics for a 1D array.
    Returns a dict with mean, variance, std, skewness, kurtosis, min, max, median, etc.
    """
    arr = np.array(data, dtype=float)
    from scipy.stats import skew, kurtosis
    return {
        'count': int(len(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'variance': float(np.var(arr, ddof=1)),
        'std': float(np.std(arr, ddof=1)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'skewness': float(skew(arr)),
        'kurtosis': float(kurtosis(arr, fisher=True)),
        'range': float(np.max(arr) - np.min(arr)),
    }


def correlation_stats(x, y, method='pearson'):
    """
    Compute correlation between two variables.
    method: 'pearson', 'spearman', or 'both'.
    Returns dict with correlation coefficient and p-value.
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    results = {}
    if method in ('pearson', 'both'):
        from scipy.stats import pearsonr
        r, p_val = pearsonr(x_arr, y_arr)
        results['pearson'] = {'r': float(r), 'p_value': float(p_val)}

    if method in ('spearman', 'both'):
        from scipy.stats import spearmanr
        r, p_val = spearmanr(x_arr, y_arr)
        results['spearman'] = {'r': float(r), 'p_value': float(p_val)}

    # Covariance
    cov = float(np.cov(x_arr, y_arr)[0, 1])
    results['covariance'] = cov

    return results


def compute_probability(slug, x_value=None, lower=None, upper=None, **params):
    """
    Compute tail probabilities for a distribution:
    - If x_value given: P(X = x) for discrete, or P(X <= x) for continuous
    - If lower and upper: P(lower <= X <= upper)
    """
    dist, args = _build_params(slug, **params)
    results = {}

    if lower is not None and upper is not None:
        low_cdf = float(dist.cdf(lower, **args))
        up_cdf = float(dist.cdf(upper, **args))
        results['probability'] = up_cdf - low_cdf
        results['label'] = f'P({lower} ≤ X ≤ {upper})'
    elif x_value is not None:
        if hasattr(dist, 'pdf'):
            # continuous: P(X <= x)
            results['probability'] = float(dist.cdf(x_value, **args))
            results['label'] = f'P(X ≤ {x_value})'
            # Also give survival
            results['survival'] = 1 - results['probability']
        else:
            # discrete: P(X = x)
            results['probability'] = float(dist.pmf(int(x_value), **args))
            results['label'] = f'P(X = {int(x_value)})'
    else:
        results['probability'] = 0
        results['label'] = 'No value specified'

    # Include quantiles for common percentiles
    quantiles = {}
    for q in [0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99]:
        try:
            quantiles[f'q{int(q*100)}'] = float(dist.ppf(q, **args))
        except Exception:
            quantiles[f'q{int(q*100)}'] = None
    results['quantiles'] = quantiles

    return results
