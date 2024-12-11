import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_traces(samples, x, facet, hue, quantiles=True, n_cols=2):

    samples_long = samples.stack().rename('value').reset_index()
    g = sns.FacetGrid(samples_long, col=facet, hue=hue, col_wrap=n_cols, sharey=False, aspect=16/(5 * n_cols))
    if quantiles:
        g = g.map(lambda x, color, **kwargs: plt.axhspan(*x.quantile([0.05, 0.95]), color=color, alpha=0.1), 'value')
        g = g.map(lambda x, color, **kwargs: plt.axhspan(*x.quantile([0.25, 0.75]), color=color, alpha=0.2), 'value')
        g = g.map(lambda x, color, **kwargs: plt.axhline(x.median(), color=color, alpha=0.4), 'value')
    g = g.map(plt.plot, x, 'value', linewidth=0.5)
    plt.show()


def plot_marginals(samples, facet, hue, quantiles=True, n_cols=2):

    samples_long = samples.stack().rename('value').reset_index()
    g = sns.FacetGrid(samples_long, col=facet, hue=hue, col_wrap=n_cols, sharex=False, sharey=False, aspect=16/(5 * n_cols))
    g = g.map(sns.kdeplot, 'value', fill=True)
    if quantiles:
        g = g.map(lambda x, color, **kwargs: sns.rugplot(x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]), color=color), 'value')
    plt.xlabel('value')
    plt.show()


def plot_acf(samples, facet, hue, n_cols=2, n_lags=32):

    acf = samples.apply(lambda x: est_acf(x.values, n_lags), 1, False, 'expand').rename_axis(columns='lag')
    acf_long = acf.stack().rename('autocorrelation').reset_index()
    g = sns.FacetGrid(acf_long, col=facet, hue=hue, col_wrap=n_cols, sharey=False, aspect=16/(5 * n_cols))
    g = g.map(lambda x, y, **kwargs: plt.plot(x, y, **kwargs), 'lag', 'autocorrelation')
    g = g.map(lambda x, y, **kwargs: plt.scatter(x, y, **kwargs), 'lag', 'autocorrelation')
    g = g.map(lambda x, y, **kwargs: plt.axhline(0, linewidth=0.5, alpha=.5, color='k'), 'lag', 'autocorrelation')
    plt.show()


def est_acf(series, max_lag):

    mean = np.mean(series)
    var = np.mean(series ** 2) - mean ** 2

    if var == 0:
        return np.array([np.nan])

    demeaned = series - mean
    acf = np.array([
        sum(demeaned[lag:] * demeaned[:-lag])
        for lag in range(1, min(len(series), max_lag))]) / var / len(series)

    return np.hstack([1, acf])


def est_int_autocor(acf, tradeoff_par=8):

    int_autocor = 0.5
    for i in range(1, len(acf)):
        int_autocor += acf[i]
        if not bool(i % 2) and i >= tradeoff_par * int_autocor:
            return int_autocor
    return int_autocor
