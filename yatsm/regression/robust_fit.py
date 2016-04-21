"""
Perform an iteratively re-weighted least squares 'robust regression'. Basically
a clone of `statsmodels.robust.robust_linear_model.RLM` without all the lovely,
but costly, creature comforts.

Reference:
    http://statsmodels.sourceforge.net/stable/rlm.html
    http://cran.r-project.org/web/packages/robustreg/index.html
    http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-robust-regression.pdf

Run this file to test performance gains. Implementation is ~3x faster than
statsmodels and can reach ~4x faster if Numba is available to accelerate.

"""
import inspect

# Don't alias to ``np`` until fix is implemented
# https://github.com/numba/numba/issues/1559
import numpy
import six
import sklearn

from yatsm.accel import try_jit

EPS = numpy.finfo('float').eps


# Weight scaling methods
@try_jit(nopython=True)
def bisquare(resid, c=4.685):
    """
    Returns weighting for each residual using bisquare weight function

    Args:
        resid (np.ndarray): residuals to be weighted
        c (float): tuning constant for Tukey's Biweight (default: 4.685)

    Returns:
        weight (ndarray): weights for residuals

    Reference:
        http://statsmodels.sourceforge.net/stable/generated/statsmodels.robust.norms.TukeyBiweight.html
    """
    # Weight where abs(resid) < c; otherwise 0
    return (numpy.abs(resid) < c) * (1 - (resid / c) ** 2) ** 2


@try_jit(nopython=True)
def mad(resid, c=0.6745):
    """
    Returns Median-Absolute-Deviation (MAD) for residuals

    Args:
        resid (np.ndarray): residuals
        c (float): scale factor to get to ~standard normal (default: 0.6745)
                 (i.e. 1 / 0.75iCDF ~= 1.4826 = 1 / 0.6745)

    Returns:
        float: MAD 'robust' variance estimate

    Reference:
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    # Return median absolute deviation adjusted sigma
    return numpy.median(numpy.fabs(resid)) / c


# UTILITY FUNCTIONS
# np.any prevents nopython
@try_jit()
def _check_converge(x0, x, tol=1e-8):
    return not numpy.any(numpy.fabs(x0 - x > tol))


# Broadcast on sw prevents nopython
# TODO: check implementation https://github.com/numba/numba/pull/1542
@try_jit()
def _weight_fit(X, y, w):
    """
    Apply a weighted OLS fit to data

    Args:
        X (ndarray): independent variables
        y (ndarray): dependent variable
        w (ndarray): observation weights

    Returns:
        tuple: coefficients and residual vector

    """
    sw = numpy.sqrt(w)

    Xw = X * sw[:, None]
    yw = y * sw

    beta, _, _, _ = numpy.linalg.lstsq(Xw, yw)

    resid = y - numpy.dot(X, beta)

    return beta, resid


# Robust regression
class RLM(sklearn.base.BaseEstimator):
    """ Robust Linear Model using Iterative Reweighted Least Squares (RIRLS)

    Perform robust fitting regression via iteratively reweighted least squares
    according to weight function and tuning parameter.

    Basically a clone from `statsmodels` that should be much faster and follows
    the scikit-learn __init__/fit/predict paradigm.

    Args:
        scale_est (callable): function for scaling residuals
        tune (float): tuning constant for scale estimate
        maxiter (int, optional): maximum number of iterations (default: 50)
        tol (float, optional): convergence tolerance of estimate
            (default: 1e-8)
        scale_est (callable): estimate used to scale the weights
            (default: `mad` for median absolute deviation)
        scale_constant (float): normalization constant (default: 0.6745)
        update_scale (bool, optional): update scale estimate for weights
            across iterations (default: True)
        M (callable): function for scaling residuals
        tune (float): tuning constant for scale estimate

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        weights (np.ndarray): 1D array of weights for each observation from a
            robust iteratively reweighted least squares

    """

    def __init__(self, M=bisquare, tune=4.685,
                 scale_est=mad, scale_constant=0.6745,
                 update_scale=True, maxiter=50, tol=1e-8):
        self.M = M
        self.tune = tune
        self.scale_est = scale_est
        self.scale_constant = scale_constant
        self.update_scale = update_scale
        self.maxiter = maxiter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """ Fit a model predicting y from X design matrix

        Args:
            X (np.ndarray): 2D (n_obs x n_features) design matrix
            y (np.ndarray): 1D independent variable

        Returns:
            object: return `self` with model results stored for method
                chaining

        """
        self.coef_, resid = _weight_fit(X, y, numpy.ones_like(y))
        self.scale = self.scale_est(resid, c=self.scale_constant)

        iteration = 1
        converged = 0
        while not converged and iteration < self.maxiter:
            _coef = self.coef_.copy()
            self.weights = self.M(resid / self.scale, c=self.tune)
            self.coef_, resid = _weight_fit(X, y, self.weights)
            if self.update_scale:
                self.scale = max(EPS,
                                 self.scale_est(resid, c=self.scale_constant))
            iteration += 1
            converged = _check_converge(self.coef_, _coef, tol=self.tol)

        return self

    def predict(self, X):
        """ Predict yhat using model

        Args:
            X (np.ndarray): 2D (n_obs x n_features) design matrix

        Returns:
            np.ndarray: 1D yhat prediction

        """
        return numpy.dot(X, self.coef_) + self.intercept_

    def __str__(self):
        return (("%s:\n"
                 " * Coefficients: %s\n"
                 " * Intercept = %.5f\n") %
                (self.__class__.__name__,
                 numpy.array_str(self.coef_, precision=4),
                 self.intercept_))
