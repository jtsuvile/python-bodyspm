# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Modified by Juulia Suvilehto
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import pearsonr, spearmanr, kendalltau
from pingouin.power import power_corr
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_esci
from pingouin.utils import remove_na, _perm_pval
from pingouin.bayesian import bayesfactor_pearson
from scipy.spatial.distance import pdist, squareform


__all__ = ["partial_corr_vector", 'corr']

def partial_corr_vector(data=None, x=None, y=None, covar=None, tail='two-sided', method='pearson'):
    """Partial and semi-partial correlation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe. Note that this function can also directly be used as a
        :py:class:`pandas.DataFrame` method, in which case this argument is
        no longer needed.
    x, y : string
        x and y. Must be names of columns in ``data``.
    covar : string or list
        Covariate(s). Must be a names of columns in ``data``. Use a list if
        there are two or more covariates.
    x_covar : string or list
        Covariate(s) for the ``x`` variable. This is used to compute
        semi-partial correlation (i.e. the effect of ``x_covar`` is removed
        from ``x`` but not from ``y``). Note that you cannot specify both
        ``covar`` and ``x_covar``.
    y_covar : string or list
        Covariate(s) for the ``y`` variable. This is used to compute
        semi-partial correlation (i.e. the effect of ``y_covar`` is removed
        from ``y`` but not from ``x``). Note that you cannot specify both
        ``covar`` and ``y_covar``.
    tail : string
        Specify whether to return the 'one-sided' or 'two-sided' p-value.
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::

        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
        'skipped' : skipped correlation (robust Spearman, requires sklearn)

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'n' : Sample size (after NaN removal)
        'outliers' : number of outliers (only for 'shepherd' or 'skipped')
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value
        'BF10' : Bayes Factor of the alternative hypothesis (Pearson only)
        'power' : achieved power of the test (= 1 - type II error).

    Notes
    -----
    From [4]_:

    “With *partial correlation*, we find the correlation between :math:`x`
    and :math:`y` holding :math:`C` constant for both :math:`x` and
    :math:`y`. Sometimes, however, we want to hold :math:`C` constant for
    just :math:`x` or just :math:`y`. In that case, we compute a
    *semi-partial correlation*. A partial correlation is computed between
    two residuals. A semi-partial correlation is computed between one
    residual and another raw (or unresidualized) variable.”

    Note that if you are not interested in calculating the statistics and
    p-values but only the partial correlation matrix, a (faster)
    alternative is to use the :py:func:`pingouin.pcorr` method (see example 4).

    Rows with missing values are automatically removed from data. Results have
    been tested against the `ppcor` R package.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Partial_correlation

    .. [2] https://cran.r-project.org/web/packages/ppcor/index.html

    .. [3] https://gist.github.com/fabianp/9396204419c7b638d38f

    .. [4] http://faculty.cas.usf.edu/mbrannick/regression/Partial.html

    """
    from pingouin.utils import _flatten_list
    # Check arguments
    # Drop rows with NaN
    col = ['x', 'y', 'covar']
    # data_orig = data.copy()
    data = pd.DataFrame(list(zip(x,y)), columns=['x','y'])
    data['covar'] = covar

    data = data[col].dropna()
    if data.shape[0] < 2 or np.unique(data['x']).shape[0]<2:
        stats = {'n': np.nan,
                 'r': np.nan,
                 'r2': np.nan,
                 'adj_r2': np.nan,
                 'CI95%': [np.nan,],
                 'p-val': np.nan,
                 'power': np.nan
                 }
        return stats

    # Standardize (= no need for an intercept in least-square regression)
    C = (data[col] - data[col].mean(axis=0)) / data[col].std(axis=0)

    if covar is not None:
        # PARTIAL CORRELATION
        cvar = np.transpose(np.atleast_2d(C['covar'].values))
        beta_x = np.linalg.lstsq(cvar, C['x'].values, rcond=None)[0]
        beta_y = np.linalg.lstsq(cvar, C['y'].values, rcond=None)[0]
        res_x = C['x'].values - np.dot(cvar, beta_x)
        res_y = C['y'].values - np.dot(cvar, beta_y)
    else:
        # SEMI-PARTIAL CORRELATION
        # Initialize "fake" residuals
        res_x, res_y = data[x].values, data[y].values
    return corr(res_x, res_y, method=method, tail=tail)


def corr(x, y, tail='two-sided', method='pearson'):
    """(Robust) correlation between two variables.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::

        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
        'skipped' : skipped correlation (robust Spearman, requires sklearn)

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'n' : Sample size (after NaN removal)
        'outliers' : number of outliers (only for 'shepherd' or 'skipped')
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value
        'BF10' : Bayes Factor of the alternative hypothesis (Pearson only)
        'power' : achieved power of the test (= 1 - type II error).

    See also
    --------
    pairwise_corr : Pairwise correlation between columns of a pandas DataFrame
    partial_corr : Partial correlation

    Notes
    -----
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Correlations of -1 or +1 imply
    an exact linear relationship.

    The Spearman correlation is a nonparametric measure of the monotonicity of
    the relationship between two datasets. Unlike the Pearson correlation,
    the Spearman correlation does not assume that both datasets are normally
    distributed. Correlations of -1 or +1 imply an exact monotonic
    relationship.

    Kendall’s tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate
    strong disagreement.

    The percentage bend correlation [1]_ is a robust method that
    protects against univariate outliers.

    The Shepherd's pi [2]_ and skipped [3]_, [4]_ correlations are both robust
    methods that returns the Spearman's rho after bivariate outliers removal.
    Note that the skipped correlation requires that the scikit-learn
    package is installed (for computing the minimum covariance determinant).

    Please note that rows with NaN are automatically removed.

    If ``method='pearson'``, The Bayes Factor is calculated using the
    :py:func:`pingouin.bayesfactor_pearson` function.

    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

    .. [2] Schwarzkopf, D.S., De Haas, B., Rees, G., 2012. Better ways to
       improve standards in brain-behavior correlation analysis. Front.
       Hum. Neurosci. 6, 200. https://doi.org/10.3389/fnhum.2012.00200

    .. [3] Rousselet, G.A., Pernet, C.R., 2012. Improving standards in
       brain-behavior correlation analyses. Front. Hum. Neurosci. 6, 119.
       https://doi.org/10.3389/fnhum.2012.00119

    .. [4] Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust correlation
       analyses: false positive and power validation using a new open
       source matlab toolbox. Front. Psychol. 3, 606.
       https://doi.org/10.3389/fpsyg.2012.00606
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Check size
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Remove NA
    x, y = remove_na(x, y, paired=True)
    nx = x.size

    # Compute correlation coefficient
    if method == 'pearson':
        r, pval = pearsonr(x, y)
    elif method == 'spearman':
        r, pval = spearmanr(x, y)
    elif method == 'kendall':
        r, pval = kendalltau(x, y)
    else:
        raise ValueError('Method not recognized.')

    assert not np.isnan(r), 'Correlation returned NaN. Check your data.'

    # Compute r2 and adj_r2
    r2 = r**2
    adj_r2 = 1 - (((1 - r2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval and power
    if r2 < 1:
        ci = compute_esci(stat=r, nx=nx, ny=nx, eftype='r')
        pr = round(power_corr(r=r, n=nx, power=None, alpha=0.05, tail=tail), 3)
    else:
        ci = [1., 1.]
        pr = np.inf

    # Create dictionnary
    stats = {'n': nx,
             'r': round(r, 3),
             'r2': round(r2, 3),
             'adj_r2': round(adj_r2, 3),
             'CI95%': [ci],
             'p-val': pval if tail == 'two-sided' else .5 * pval,
             'power': pr
             }

    # Compute the BF10 for Pearson correlation only
    if method == 'pearson':
        if r2 < 1:
            stats['BF10'] = bayesfactor_pearson(r, nx, tail=tail)
        else:
            stats['BF10'] = str(np.inf)

    # Convert to DataFrame
    stats = pd.DataFrame.from_records(stats, index=[method])

    # Define order
    col_keep = ['n', 'outliers', 'r', 'CI95%', 'r2', 'adj_r2', 'p-val',
                'BF10', 'power']
    col_order = [k for k in col_keep if k in stats.keys().tolist()]
    return stats[col_order]
