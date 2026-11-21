"""Characterise the noise level of dF/F traces, as Cascade does.

Cascade's noise level is a robust, frame-rate-independent measure of the
high-frequency noise in a dF/F trace::

    noise_level = median(|ΔdF/F|) / sqrt(frame_rate) * 100     # percent * s^(1/2)

The median absolute difference between subsequent samples is an outlier-robust
estimate of the trace's standard deviation (it converges to the standard
deviation for uncorrelated, outlier-free traces); dividing by ``sqrt(fs)``
makes the value comparable across recordings with different frame rates.  The
result is reported in percent, with units ``% * s^(1/2)``.

This mirrors ``cascade2p.utils.calculate_noise_levels`` and
``cascade2p.utils.plot_noise_level_distribution`` from the Cascade repository
(https://github.com/HelmchenLabSoftware/Cascade), so a trace's noise level is
directly comparable to the ``NoiseLevel`` of the pretrained models.

Example
-------
::

    from cascade_data import noise_level, noise_levels, plot_noise_distribution

    level = noise_level(dff, frame_rate=30.0)          # one trace -> float
    levels = noise_levels(traces, frame_rate=30.0)     # [n_neurons, time] -> [n_neurons]
    plot_noise_distribution(traces, 30.0, "noise.png") # histogram -> PNG
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def noise_level(dff, frame_rate):
    """Cascade's noise level of a single dF/F trace, in percent.

    ``median(|ΔdF/F|) / sqrt(frame_rate) * 100``.  Returns ``0.0`` for a trace
    with fewer than two samples (where the difference is undefined).
    """
    dff = np.asarray(dff, dtype=float).ravel()
    if dff.size < 2:
        return 0.0
    return float(np.nanmedian(np.abs(np.diff(dff))) / np.sqrt(frame_rate) * 100.0)


def noise_levels(neurons_x_time, frame_rate):
    """Noise levels of a ``[n_neurons, time]`` matrix, one per neuron.

    A 1-D input is treated as a single neuron and returns a length-1 array.
    """
    traces = np.asarray(neurons_x_time, dtype=float)
    if traces.ndim == 1:
        traces = traces[None, :]
    if traces.shape[-1] < 2:
        return np.zeros(traces.shape[0], dtype=float)
    return (np.nanmedian(np.abs(np.diff(traces, axis=-1)), axis=-1)
            / np.sqrt(frame_rate) * 100.0)


def frame_rate_from_time(time):
    """Estimate the sampling rate [Hz] from a (possibly irregular) time axis.

    Uses the median inter-sample interval, so a few dropped frames do not skew
    the estimate.  Returns ``0.0`` when the rate cannot be determined.
    """
    t = np.asarray(time, dtype=float).ravel()
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else 0.0


def plot_noise_levels(levels, save_path=None, bins=100, percentile=99.9,
                      title=None):
    """Histogram of precomputed noise levels, saved as a PNG.

    Mirrors Cascade's ``plot_noise_level_distribution``: the x-axis is clipped
    at the ``percentile``-th noise level so a few very noisy neurons do not
    squash the bulk of the distribution.  Returns the levels unchanged.
    """
    levels = np.asarray(levels, dtype=float).ravel()
    finite = levels[np.isfinite(levels)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite, bins=bins, density=True, color="tab:blue", alpha=0.8)
    if finite.size:
        hi = float(np.nanpercentile(finite, percentile))
        if np.isfinite(hi) and hi > 0:
            ax.set_xlim(0, hi)
    ax.set_xlabel("noise level (% s$^{1/2}$)")
    ax.set_ylabel("density")
    ax.set_title(title or "Noise level distribution across neurons")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return levels


def plot_noise_distribution(traces, frame_rate, save_path=None, bins=100,
                            percentile=99.9, title=None):
    """Noise-level histogram of a ``[n_neurons, time]`` matrix, saved as PNG.

    Convenience wrapper: computes :func:`noise_levels` and delegates to
    :func:`plot_noise_levels`.  Returns the per-neuron noise levels.
    """
    return plot_noise_levels(noise_levels(traces, frame_rate), save_path,
                             bins=bins, percentile=percentile, title=title)
