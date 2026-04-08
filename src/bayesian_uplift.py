from __future__ import annotations

from typing import Any, Sequence

import arviz as az
import matplotlib.axes
import numpy as np
import pandas as pd
import pymc as pm


def fit_bayesian_views_model(
    df: pd.DataFrame,
    views_col: str = "first_7d_views",
    day_col: str = "publish_weekday",
    format_col: str = "video_type",
    alpha_sigma: float = 2.0,
    sigma_day_prior: float = 0.15,
    sigma_format_prior: float = 0.7,
    sigma_obs_prior: float = 1.0,
    prior_samples: int = 1000,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.92,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Fit a hierarchical Bayesian model for log(1 + views) with weekday and format effects.

    Returns a dict with the fitted PyMC model, inference data, predictive samples,
    encoded category names, and a copy of the prepared dataframe.
    """
    data = df.copy()
    data["log_views"] = np.log1p(data[views_col])

    day_codes, day_uniques = pd.factorize(data[day_col], sort=True)
    format_codes, format_uniques = pd.factorize(data[format_col], sort=True)

    y = data["log_views"].to_numpy(dtype="float64")
    y_mean = y.mean()

    coords = {
        "obs_id": np.arange(len(data)),
        "day": day_uniques,
        "format": format_uniques,
    }

    with pm.Model(coords=coords) as model:
        day_idx = pm.Data("day_idx", day_codes, dims="obs_id")
        format_idx = pm.Data("format_idx", format_codes, dims="obs_id")

        alpha = pm.Normal("alpha", mu=y_mean, sigma=alpha_sigma)

        sigma_day = pm.HalfNormal("sigma_day", sigma=sigma_day_prior)
        sigma_format = pm.HalfNormal("sigma_format", sigma=sigma_format_prior)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_obs_prior)

        day_raw = pm.Normal("day_raw", mu=0.0, sigma=1.0, dims="day")
        day_centered = day_raw - pm.math.mean(day_raw)
        day_effect = pm.Deterministic(
            "day_effect",
            day_centered * sigma_day,
            dims="day",
        )

        format_raw = pm.Normal("format_raw", mu=0.0, sigma=1.0, dims="format")
        format_centered = format_raw - pm.math.mean(format_raw)
        format_effect = pm.Deterministic(
            "format_effect",
            format_centered * sigma_format,
            dims="format",
        )

        mu = pm.Deterministic(
            "mu",
            alpha + day_effect[day_idx] + format_effect[format_idx],
            dims="obs_id",
        )

        pm.Normal(
            "log_views_obs",
            mu=mu,
            sigma=sigma_obs,
            observed=y,
            dims="obs_id",
        )

        prior_pred = pm.sample_prior_predictive(
            samples=prior_samples,
            random_seed=random_seed,
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )

        post_pred = pm.sample_posterior_predictive(
            idata,
            random_seed=random_seed,
        )

    return {
        "model": model,
        "idata": idata,
        "prior_pred": prior_pred,
        "post_pred": post_pred,
        "day_uniques": day_uniques,
        "format_uniques": format_uniques,
        "data": data,
    }


def build_uplift_tables(
    idata: az.InferenceData,
    day_names: Sequence[str],
    format_names: Sequence[str],
    day_label: str = "publish_weekday",
    format_label: str = "video_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build uplift tables for weekday and format effects from posterior samples.

    Uplift is computed as 100 * (exp(effect) - 1).
    """
    posterior_day = idata.posterior["day_effect"].values
    posterior_format = idata.posterior["format_effect"].values

    day_uplift = np.expm1(posterior_day)
    format_uplift = np.expm1(posterior_format)

    day_uplift_df = pd.DataFrame(
        {
            day_label: list(day_names),
            "median_uplift_pct": 100 * np.median(day_uplift, axis=(0, 1)),
            "low_95_pct": 100 * np.quantile(day_uplift, 0.025, axis=(0, 1)),
            "high_95_pct": 100 * np.quantile(day_uplift, 0.975, axis=(0, 1)),
        }
    ).sort_values("median_uplift_pct", ascending=False)

    format_uplift_df = pd.DataFrame(
        {
            format_label: list(format_names),
            "median_uplift_pct": 100 * np.median(format_uplift, axis=(0, 1)),
            "low_95_pct": 100 * np.quantile(format_uplift, 0.025, axis=(0, 1)),
            "high_95_pct": 100 * np.quantile(format_uplift, 0.975, axis=(0, 1)),
        }
    ).sort_values("median_uplift_pct", ascending=False)

    return day_uplift_df, format_uplift_df


def build_pairwise_uplift(
    idata: az.InferenceData,
    coord_name: str,
    baseline: str,
    comparison: str,
) -> pd.DataFrame:
    """
    Compute pairwise uplift between two levels of the same posterior effect.

    Example:
        coord_name='day', baseline='Monday', comparison='Friday'
    """
    effect_name = f"{coord_name}_effect"
    posterior = idata.posterior[effect_name]

    coord_values = list(posterior.coords[coord_name].values)
    idx_map = {name: i for i, name in enumerate(coord_values)}

    comp_idx = idx_map[comparison]
    base_idx = idx_map[baseline]

    values = posterior.values
    delta = values[:, :, comp_idx] - values[:, :, base_idx]
    uplift = np.expm1(delta)

    return pd.DataFrame(
        {
            "comparison": [f"{comparison} vs {baseline}"],
            "median_uplift_pct": [100 * np.median(uplift)],
            "low_95_pct": [100 * np.quantile(uplift, 0.025)],
            "high_95_pct": [100 * np.quantile(uplift, 0.975)],
            "prob_gt_zero": [float((uplift > 0).mean())],
        }
    )


def plot_uplift_errorbars(
    df: pd.DataFrame,
    category_col: str,
    ax: matplotlib.axes.Axes,
    order: Sequence[str] | None = None,
    xlabel: str = "Expected Uplift (%)",
    ylabel: str | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot uplift medians with 95% interval error bars for a category column.

    Expected columns in df:
      - category_col
      - median_uplift_pct
      - low_95_pct
      - high_95_pct
    """
    config = {
        "point_color": "black",
        "error_color": "black",
        "zero_line_color": "red",
        "grid_alpha": 0.5,
        "elinewidth": 1.8,
        "capsize": 4,
        "capthick": 1.8,
        "markersize": 7,
        "text_offset": 0.5,
        "text_y_offset": 0.15,
        "positive_color": "green",
        "negative_color": "red",
        "fontsize": 10,
        "xlabel_fontsize": 12,
        "ylabel_fontsize": 12,
        "title_fontsize": 13,
        "title_pad": 10,
    }

    plot_df = df.copy()

    if order is not None:
        plot_df[category_col] = pd.Categorical(
            plot_df[category_col],
            categories=list(order),
            ordered=True,
        )
        plot_df = plot_df.sort_values(category_col)

    y = plot_df[category_col]
    x = plot_df["median_uplift_pct"]
    err_low = x - plot_df["low_95_pct"]
    err_high = plot_df["high_95_pct"] - x

    ax.errorbar(
        x,
        y,
        xerr=[err_low, err_high],
        fmt="o",
        color=config["point_color"],
        ecolor=config["error_color"],
        elinewidth=config["elinewidth"],
        capsize=config["capsize"],
        capthick=config["capthick"],
        markersize=config["markersize"],
    )
    ax.axvline(0, color=config["zero_line_color"], linewidth=config["elinewidth"])
    ax.grid(alpha=config["grid_alpha"])
    ax.set_ylim((-0.5, len(y) - 0.5))

    for i, xi in enumerate(x):
        text_color = config["positive_color"] if xi > 0 else config["negative_color"]
        ax.text(
            xi + (config["text_offset"] if xi >= 0 else -config["text_offset"]),
            i + config["text_y_offset"],
            f"{xi:+.1f}%",
            va="center",
            ha="left" if xi >= 0 else "right",
            fontsize=config["fontsize"],
            color=text_color,
            fontweight="bold",
        )

    ax.set_xlabel(xlabel, fontsize=config["xlabel_fontsize"])
    ax.set_ylabel(ylabel or category_col, fontsize=config["ylabel_fontsize"])
    if title is not None:
        ax.set_title(
            title,
            fontsize=config["title_fontsize"],
            pad=config["title_pad"],
        )
    return ax
