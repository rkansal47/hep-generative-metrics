"""
Collection of methods for calculating distances and divergences between two distributions.

Author: Raghav Kansal
"""

import itertools


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable
from numpy.typing import ArrayLike

import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance, linregress, iqr
from scipy.optimize import curve_fit

import sklearn.metrics
from sklearn.linear_model import LinearRegression

import time
import logging

from tqdm import tqdm

import numba_scipy
from numba import jit, njit, prange

# from jetnet.datasets.normalisations import FeaturewiseLinearBounded


def normalise_features(X: ArrayLike, Y: ArrayLike = None) -> ArrayLike:
    maxes = np.max(np.abs(X), axis=0)

    return (X / maxes, Y / maxes) if Y is not None else X / maxes


def multi_batch_evaluation(
    X: ArrayLike,
    Y: ArrayLike,
    num_batches: int,
    batch_size: int,
    metric: Callable,
    seed: int = 42,
    normalise: bool = True,
    timing: bool = False,
    **metric_args,
):
    np.random.seed(seed)

    if normalise:
        X, Y = normalise_features(X, Y)

    times = []

    vals = []
    for _ in range(num_batches):
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        t0 = time.time()
        val = metric(rand_sample1, rand_sample2, normalise=False, **metric_args)
        t1 = time.time()
        vals.append(val)
        times.append(t1 - t0)

    mean_std = (np.mean(vals, axis=0), np.std(vals, axis=0))

    return (mean_std, times) if timing else mean_std


def multi_batch_evaluation_median(
    X: ArrayLike,
    Y: ArrayLike,
    num_batches: int,
    batch_size: int,
    metric: Callable,
    seed: int = 42,
    normalise: bool = True,
    timing: bool = False,
    **metric_args,
):
    np.random.seed(seed)

    if normalise:
        X, Y = normalise_features(X, Y)

    times = []

    vals = []
    for _ in range(num_batches):
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        t0 = time.time()
        val = metric(rand_sample1, rand_sample2, normalise=False, **metric_args)
        t1 = time.time()
        vals.append(val)
        times.append(t1 - t0)

    median_err = [np.median(vals), iqr(vals, rng=(16.275, 83.725)) / 2]

    return (median_err, times) if timing else median_err


# @njit
def _average_batches_mmd(X, Y, num_batches, batch_size, seed, degree=4):
    vals_point = []
    for i in range(num_batches):
        np.random.seed(seed + i * 1000)
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = mmd_poly_quadratic_unbiased(
            rand_sample1, rand_sample2, normalise=False, degree=degree
        )
        vals_point.append(val)

    return vals_point


def multi_batch_evaluation_mmd(
    X: ArrayLike,
    Y: ArrayLike,
    num_batches: int = 10,
    batch_size: int = 5000,
    seed: int = 42,
    normalise: bool = True,
    degree: int = 4,
    timing: bool = False,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    num_batches = 10

    t0 = time.time()
    vals_point = _average_batches_mmd(X, Y, num_batches, batch_size, seed, degree=degree)
    t1 = time.time()
    
    med_err = [np.median(vals_point), iqr(vals_point, rng=(16.275, 83.725)) / 2]

    return (med_err, t1 - t0) if timing else med_err

    # return [np.mean(vals_point), np.std(vals_point)]


# based on https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
def one_over_n_extrapolation(
    X: ArrayLike,
    Y: ArrayLike,
    metric: Callable,
    min_samples: int = 20_000,
    max_samples: int = 50_000,
    num_batches: int = 20,
    num_points: int = 10,
    seed: int = 42,
    normalise: bool = True,
    **metric_args,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    # Choose the number of images to evaluate FID_N at regular intervals over N
    batches = np.linspace(min_samples, max_samples, num_points).astype("int32")

    np.random.seed(seed)

    vals = []

    # Evaluate for different Ns
    for batch_size in batches:
        for _ in range(num_batches):
            rand1 = np.random.choice(len(X), size=batch_size)
            rand2 = np.random.choice(len(Y), size=batch_size)

            rand_sample1 = X[rand1]
            rand_sample2 = Y[rand2]

            val = metric(rand_sample1, rand_sample2, normalise=False, **metric_args)
            vals.append(val)

    vals = np.array(vals)

    # return vals

    # Fit linear regression
    result = linregress(1 / np.repeat(batches, num_batches), vals)
    return [
        result.intercept,
        result.intercept_stderr,
        np.repeat(batches, num_batches),
        vals,
        result.slope,
    ]


def linear(x, intercept, slope):
    return intercept + slope * x


def _average_batches(X, Y, metric, batch_size, num_batches, seed):
    vals_point = []
    for _ in range(num_batches):
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = metric(rand_sample1, rand_sample2, normalise=False)
        vals_point.append(val)

    return [np.mean(vals_point), np.std(vals_point)]


# based on https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
# @jit(parallel=True)
def one_over_n_extrapolation_repeated_measurements(
    X: ArrayLike,
    Y: ArrayLike,
    min_samples: int = 20_000,
    max_samples: int = 50_000,
    num_batches: int = 20,
    num_points: int = 10,
    seed: int = 42,
    normalise: bool = True,
    inverse_intervals: bool = True,
    timing: bool = False,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    # Choose the number of images to evaluate FID_N at regular intervals over N
    if inverse_intervals:
        batches = (1 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)).astype("int32")
    else:
        batches = np.linspace(min_samples, max_samples, num_points).astype("int32")

    # assert num_batches >= 5, "Needs at least 5 estimates per point"

    np.random.seed(seed)

    vals = []

    t0 = time.time()

    for i, batch_size in enumerate(batches):
        val_points = []
        for _ in range(num_batches):
            rand1 = np.random.choice(len(X), size=batch_size)
            rand2 = np.random.choice(len(Y), size=batch_size)

            rand_sample1 = X[rand1]
            rand_sample2 = Y[rand2]

            mu1 = np.mean(rand_sample1, axis=0)
            sigma1 = np.cov(rand_sample1, rowvar=False)
            mu2 = np.mean(rand_sample2, axis=0)
            sigma2 = np.cov(rand_sample2, rowvar=False)

            val = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            val_points.append(val)

        vals.append(np.mean(val_points))

    params, covs = curve_fit(linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf]))

    t1 = time.time()

    res = [params[0], np.sqrt(np.diag(covs)[0]), batches, vals, params[1]]

    return res if not timing else (res[:2], t1 - t0)
    return res[:2]  # , batches, vals, params[1]]  #


def wasserstein1d(X: ArrayLike, Y: ArrayLike, normalise: bool = True) -> float:
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

    if normalise:
        X, Y = normalise_features(X, Y)

    return wasserstein_distance(X, Y)


def wasserstein(X: ArrayLike, Y: ArrayLike, normalise: bool = True) -> float:
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

    import ot

    if normalise:
        X, Y = normalise_features(X, Y)

    n = X.shape[0]

    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
    M = ot.dist(X, Y)
    return ot.emd2(a, b, M)


def trace_covmean(sigma1, sigma2):
    return


# from https://github.com/mseitzer/pytorch-fid
# @jit
def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # msg = (
        #     f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        # )
        # logging.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def frechet_gaussian_distance(X: ArrayLike, Y: ArrayLike, normalise: bool = True) -> float:
    if normalise:
        X, Y = normalise_features(X, Y)

    mu1 = np.mean(X, axis=0)
    sigma1 = np.cov(X, rowvar=False)
    mu2 = np.mean(Y, axis=0)
    sigma2 = np.cov(Y, rowvar=False)

    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def _sqeuclidean(X: ArrayLike, Y: ArrayLike) -> np.ndarray:
    return np.sum((X - Y) ** 2, axis=1)


def _rbf_kernel_elementwise(X: ArrayLike, Y: ArrayLike, kernel_sigma: float) -> np.ndarray:
    return np.exp(-_sqeuclidean(X, Y) / (2.0 * (kernel_sigma**2)))


@njit
def _poly_kernel_pairwise(X: ArrayLike, Y: ArrayLike, degree: int) -> np.ndarray:
    gamma = 1.0 / X.shape[-1]
    return (X @ Y.T * gamma + 1.0) ** degree


def _poly_kernel_elementwise(X: ArrayLike, Y: ArrayLike, degree: int) -> np.ndarray:
    gamma = 1.0 / X.shape[-1]
    return (np.sum(X * Y, axis=-1) * gamma + 1.0) ** degree


# @njit
def _get_mmd_quadratic_arrays(X: ArrayLike, Y: ArrayLike, kernel_func: Callable, **kernel_args):
    XX = kernel_func(X, X, **kernel_args)
    YY = kernel_func(Y, Y, **kernel_args)
    XY = kernel_func(X, Y, **kernel_args)
    return XX, YY, XY


def _mmd_quadratic_biased(XX: ArrayLike, YY: ArrayLike, XY: ArrayLike):
    return XX.mean() + YY.mean() - 2 * XY.mean()


@njit
def _mmd_quadratic_unbiased(XX: ArrayLike, YY: ArrayLike, XY: ArrayLike):
    m, n = XX.shape[0], YY.shape[0]
    # subtract diagonal 1s
    return (
        (XX.sum() - np.trace(XX)) / (m * (m - 1))
        + (YY.sum() - np.trace(YY)) / (n * (n - 1))
        - 2 * np.mean(XY)
    )


def _mmd_linear(X: ArrayLike, Y: ArrayLike, kernel_func: Callable, **kernel_args):
    N = (len(X) // 2) * 2  # even N

    h = (
        kernel_func(X[0:N:2], X[1:N:2], **kernel_args)
        + kernel_func(Y[0:N:2], Y[1:N:2], **kernel_args)
        - kernel_func(X[0:N:2], Y[1:N:2], **kernel_args)
        - kernel_func(X[1:N:2], Y[0:N:2], **kernel_args)
    )

    return 2 * h.sum() / N  # average


# REMEMBER SKLEARN DEPENDENCY!!
def mmd_gaussian_quadratic_biased(
    X: ArrayLike, Y: ArrayLike, kernel_sigma: float, normalise: bool = True
) -> float:
    if normalise:
        X, Y = normalise_features(X, Y)

    gamma = 1.0 / (2.0 * (kernel_sigma**2))
    XX, YY, XY = _get_mmd_quadratic_arrays(X, Y, sklearn.metrics.pairwise.rbf_kernel, gamma=gamma)
    return _mmd_quadratic_biased(XX, YY, XY)


def mmd_gaussian_quadratic_unbiased(
    X: ArrayLike, Y: ArrayLike, kernel_sigma: float, normalise: bool = True
) -> float:
    if normalise:
        X, Y = normalise_features(X, Y)

    gamma = 1.0 / (2.0 * (kernel_sigma**2))
    # this can maybe be optimized - only need to calculate half of these
    XX, YY, XY = _get_mmd_quadratic_arrays(X, Y, sklearn.metrics.pairwise.rbf_kernel, gamma=gamma)
    return _mmd_quadratic_unbiased(XX, YY, XY)


@njit
def mmd_poly_quadratic_unbiased(
    X: ArrayLike, Y: ArrayLike, degree: int = 3, normalise: bool = True
) -> float:

    # if normalise:
    #     X, Y = normalise_features(X, Y)

    # XX, YY, XY = _get_mmd_quadratic_arrays(X, Y, _poly_kernel_pairwise, degree=degree)

    XX = _poly_kernel_pairwise(X, X, degree=degree)
    YY = _poly_kernel_pairwise(Y, Y, degree=degree)
    XY = _poly_kernel_pairwise(X, Y, degree=degree)
    return _mmd_quadratic_unbiased(XX, YY, XY)


def mmd_gaussian_linear(
    X: ArrayLike, Y: ArrayLike, kernel_sigma: float, normalise: bool = True
) -> float:
    assert len(X) == len(Y), "Linear estimate assumes equal number of samples"

    if normalise:
        X, Y = normalise_features(X, Y)

    return _mmd_linear(X, Y, _rbf_kernel_elementwise, kernel_sigma=kernel_sigma)


def mmd_poly_linear(X: ArrayLike, Y: ArrayLike, degree: int = 3, normalise: bool = True) -> float:
    assert len(X) == len(Y), "Linear estimate assumes equal number of samples"

    if normalise:
        X, Y = normalise_features(X, Y)

    return _mmd_linear(X, Y, _poly_kernel_elementwise, degree=degree)


# https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
def _compute_pairwise_distance(data_x: ArrayLike, data_y: ArrayLike = None) -> np.ndarray:
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric="euclidean")
    return dists


# https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
def _get_kth_value(unsorted: ArrayLike, k: int, axis=-1):
    """
    Args:
        unsorted: array of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


# https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
def compute_nearest_neighbour_distances(X: ArrayLike, nearest_k: int = 5):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = _compute_pairwise_distance(X)
    radii = _get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


# based on # https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
def pr(
    X: ArrayLike,
    Y: ArrayLike,
    X_nearest_neighbour_distances: ArrayLike = None,
    nearest_k: int = 5,
    normalise: bool = True,
):
    if X_nearest_neighbour_distances is None:
        if normalise:
            X = normalise_features(X)

        X_nearest_neighbour_distances = compute_nearest_neighbour_distances(X, nearest_k)
    else:
        if isinstance(X_nearest_neighbour_distances, dict):
            X_nearest_neighbour_distances = X_nearest_neighbour_distances[len(Y)]

    if normalise:
        Y = normalise_features(Y)

    Y_nearest_neighbour_distances = compute_nearest_neighbour_distances(Y, nearest_k)
    distance_XY = _compute_pairwise_distance(X, Y)

    precision = (
        (distance_XY < np.expand_dims(X_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
    )

    recall = (
        (distance_XY < np.expand_dims(Y_nearest_neighbour_distances, axis=0)).any(axis=1).mean()
    )

    return precision, recall


# based on # https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
def dc(
    X: ArrayLike,
    Y: ArrayLike,
    X_nearest_neighbour_distances: ArrayLike = None,
    nearest_k: int = 5,
    normalise: bool = True,
):
    if X_nearest_neighbour_distances is None:
        if normalise:
            X = normalise_features(X)

        X_nearest_neighbour_distances = compute_nearest_neighbour_distances(X, nearest_k)
    else:
        if isinstance(X_nearest_neighbour_distances, dict):
            X_nearest_neighbour_distances = X_nearest_neighbour_distances[len(Y)]

    if normalise:
        Y = normalise_features(Y)

    distance_XY = _compute_pairwise_distance(X, Y)

    density = (1.0 / float(nearest_k)) * (
        distance_XY < np.expand_dims(X_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (distance_XY.min(axis=1) < X_nearest_neighbour_distances).mean()

    return density, coverage
