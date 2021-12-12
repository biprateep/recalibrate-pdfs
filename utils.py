from scipy.stats import binom
from matplotlib import pyplot as plt
import numpy as np
from xgboost import XGBClassifier, XGBRegressor


def get_pit(cdes: np.ndarray, z_grid: np.ndarray, z_test: np.ndarray) -> np.ndarray:
    """
    Calculates PIT based on CDE

    cdes: a numpy array of conditional density estimates;
        each row corresponds to an observation, each column corresponds to a grid
        point
    z_grid: a numpy array of the grid points at which cde_estimates is evaluated
    z_test: a numpy array of the true z values corresponding to the rows of cde_estimates

    returns: A numpy array of values

    """
    # flatten the input arrays to 1D
    z_grid = np.ravel(z_grid)
    z_test = np.ravel(z_test)

    # Sanity checks
    nrow_cde, ncol_cde = cdes.shape
    n_samples = z_test.shape[0]
    n_grid_points = z_grid.shape[0]

    if nrow_cde != n_samples:
        raise ValueError(
            "Number of samples in CDEs should be the same as in z_test."
            "Currently %s and %s." % (nrow_cde, n_samples)
        )
    if ncol_cde != n_grid_points:
        raise ValueError(
            "Number of grid points in CDEs should be the same as in z_grid."
            "Currently %s and %s." % (nrow_cde, n_grid_points)
        )

    z_min = np.min(z_grid)
    z_max = np.max(z_grid)
    z_delta = (z_max - z_min) / (n_grid_points - 1)

    # Vectorized implementation using masked arrays
    pit = np.ma.masked_array(cdes, (z_grid > z_test[:, np.newaxis]))
    pit = z_delta * np.sum(pit, axis=-1)

    return np.array(pit)


def plot_pit(pit_values, ci_level, n_bins=30, ax=None, **fig_kw):
    """
    Plots the PIT/HPD histogram and calculates the confidence interval for the bin values, were the PIT/HPD values follow an uniform distribution

    @param values: a numpy array with PIT/HPD values
    @param ci_level: a float between 0 and 1 indicating the size of the confidence level
    @param x_label: a string, populates the x_label of the plot
    @param n_bins: an integer, the number of bins in the histogram
    @param figsize: a tuple, the plot size (width, height)
    @param ylim: a list of two elements, including the lower and upper limit for the y axis

    @returns The matplotlib figure object with the histogram of the PIT/HPD values and the CI for the uniform distribution
    """

    # Extract the number of CDEs
    n = pit_values.shape[0]

    # Creating upper and lower limit for selected uniform band
    ci_quantity = (1 - ci_level) / 2
    low_lim = binom.ppf(q=ci_quantity, n=n, p=1 / n_bins)
    upp_lim = binom.ppf(q=ci_level + ci_quantity, n=n, p=1 / n_bins)

    # Creating figure

    if ax is None:
        fig, ax = plt.subplots(1, 2, **fig_kw)

    # plot PIT histogram
    ax[0].hist(pit_values, bins=n_bins)
    ax[0].axhline(y=low_lim, color="grey")
    ax[0].axhline(y=upp_lim, color="grey")
    ax[0].axhline(y=n / n_bins, label="Uniform Average", color="red")
    ax[0].fill_between(
        x=np.linspace(0, 1, 100),
        y1=np.repeat(low_lim, 100),
        y2=np.repeat(upp_lim, 100),
        color="grey",
        alpha=0.2,
    )
    ax[0].set_xlabel("PIT Values")
    ax[0].legend(loc="best")

    # plot P-P plot
    prob_theory = np.linspace(0, 1, 100)
    prob_data = [np.sum(pit_values < i) / len(pit_values) for i in prob_theory]
    # plot Q-Q
    quants = np.linspace(0, 100, 100)
    quant_theory = quants/100.
    quant_data = np.percentile(pit_values,quants)

    ax[1].scatter(quant_theory, quant_data, marker=".")
    ax[1].plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), "k--")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Theoretical quantile")
    ax[1].set_ylabel("Empirical quantile")
    ax[1].set_aspect("equal")

    return fig, ax
