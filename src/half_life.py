from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_half_life_regression_data(
    video_cohort_data: pd.DataFrame,
    min_duration_months: int = 4,
    ratio_eps: float = 1e-6,
    group_cols: Sequence[str] = (
        "video_id",
        "platform",
        "publish_month",
        "data_month",
        "months_since_publish",
        "video_type",
    ),
) -> pd.DataFrame:
    """
    Prepare cohort data for half-life linear regression.

    Steps:
    1. Sort trajectories by video, platform, and month.
    2. Build cumulative views by (video_id, platform).
    3. Aggregate duplicated mapped rows using max(cum_views) and sum(monthly metrics).
    4. Keep keys observed for at least min_duration_months.
    5. Build ratio to saturation views and stable log1p transform.
    """
    data = video_cohort_data.sort_values(
        ["video_id", "platform", "months_since_publish"]
    ).copy()

    data["cum_views"] = data.groupby(["video_id", "platform"])["views"].cumsum()

    data = data.groupby(list(group_cols), as_index=False).agg(
        {
            "views": "sum",
            "watch_time_minutes": "sum",
            "cum_views": "max",
        }
    )

    max_month_per_key = (
        data.groupby(["video_id", "platform"], as_index=False)["months_since_publish"]
        .max()
        .rename(columns={"months_since_publish": "max_month"})
    )
    valid_keys = max_month_per_key[
        max_month_per_key["max_month"] >= (min_duration_months - 1)
    ][["video_id", "platform"]]

    data = data.merge(valid_keys, on=["video_id", "platform"], how="inner")

    v_sat = (
        data.groupby(["video_id", "platform"])["cum_views"]
        .max()
        .rename("v_sat")
        .reset_index()
    )
    data = data.merge(v_sat, on=["video_id", "platform"], how="left")

    data["ratio_views"] = data["cum_views"] / data["v_sat"]
    data = data[data["ratio_views"] < 1].copy()
    data["ratio_views"] = data["ratio_views"].clip(lower=ratio_eps, upper=1 - ratio_eps)
    data["log_ratio_views"] = np.log1p(-data["ratio_views"])

    data = data[np.isfinite(data["log_ratio_views"])].copy()
    return data


def estimate_life_time_linear(
    prepared_data: pd.DataFrame,
    min_unique_months: int = 2,
    min_abs_slope: float = 1e-3,
    max_life_time_months: float = 24.0,
) -> pd.DataFrame:
    """
    Estimate life_time (in months) from linear fit:
      log(1 - ratio_views) = coef * months_since_publish, with zero intercept.

    Returned columns:
      - video_id
      - platform
      - life_time
      - slope
    """
    rows: list[dict[str, float | str]] = []

    for (video_id, platform), group in prepared_data.groupby(["video_id", "platform"]):
        fit = group[group["months_since_publish"] > 0]
        if fit["months_since_publish"].nunique() < min_unique_months:
            continue

        x = fit["months_since_publish"].to_numpy().reshape(-1, 1)
        y = fit["log_ratio_views"].to_numpy()

        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        slope = float(model.coef_[0])

        if not np.isfinite(slope) or slope >= 0:
            continue
        if abs(slope) < min_abs_slope:
            continue

        life_time = -1.0 / slope
        if not np.isfinite(life_time) or life_time <= 0:
            continue
        if life_time > max_life_time_months:
            continue

        rows.append(
            {
                "video_id": video_id,
                "platform": platform,
                "life_time": life_time,
                "slope": slope,
            }
        )

    return pd.DataFrame(rows)


def build_life_time_plot_data(
    data_with_life_time: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Deduplicate mapped rows and trim extreme life_time tails for plotting."""
    plot_data = (
        data_with_life_time.dropna(subset=["life_time", "platform", "video_type"])
        .drop_duplicates(subset=["video_id", "platform", "video_type"])
        .copy()
    )

    q_low, q_high = plot_data["life_time"].quantile([lower_q, upper_q])
    plot_data = plot_data[plot_data["life_time"].between(q_low, q_high)].copy()
    return plot_data
