from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def prepare_popularity_tiers(
    data: pd.DataFrame,
    views_col: str = "total_views",
    platform_col: str = "platform",
    content_col: str = "content_original_id",
    video_col: str = "video_id",
    q50: float = 0.50,
    q90: float = 0.90,
    q99: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare dataset with popularity tiers from global quantiles of views.

    Returns:
      - prepared dataframe with `popularity_tier` and `log1p_views`
      - dict with quantile cutoffs and tier order metadata
    """
    df = data.copy()
    df[views_col] = pd.to_numeric(df[views_col], errors="coerce")
    df = df.dropna(subset=[platform_col, content_col, video_col, views_col]).copy()
    df = df[df[views_col] >= 0].copy()

    p50 = float(df[views_col].quantile(q50))
    p90 = float(df[views_col].quantile(q90))
    p99 = float(df[views_col].quantile(q99))

    def assign_tier(v: float) -> str:
        if v <= p50:
            return "Low (<=P50)"
        if v <= p90:
            return "Mid (P50-P90)"
        if v <= p99:
            return "High (P90-P99)"
        return "Viral (>P99)"

    tier_order = ["Low (<=P50)", "Mid (P50-P90)", "High (P90-P99)", "Viral (>P99)"]
    df["popularity_tier"] = pd.Categorical(
        df[views_col].apply(assign_tier),
        categories=tier_order,
        ordered=True,
    )
    df["log1p_views"] = np.log1p(df[views_col])

    meta = {
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "tier_order": tier_order,
    }
    return df, meta


def build_tier_platform_metrics(
    data: pd.DataFrame,
    tier_col: str = "popularity_tier",
    platform_col: str = "platform",
    video_col: str = "video_id",
) -> pd.DataFrame:
    """Build per-tier platform share and lift vs global platform share."""
    tier_platform = (
        data.groupby([tier_col, platform_col], observed=True)[video_col]
        .size()
        .rename("videos")
        .reset_index()
    )
    tier_platform["tier_total"] = tier_platform.groupby(tier_col, observed=True)[
        "videos"
    ].transform("sum")
    tier_platform["tier_share"] = tier_platform["videos"] / tier_platform["tier_total"]

    global_share = (
        data[platform_col]
        .value_counts(normalize=True)
        .rename("global_share")
        .rename_axis(platform_col)
        .reset_index()
    )

    tier_platform = tier_platform.merge(global_share, on=platform_col, how="left")
    tier_platform["share_lift_vs_global"] = (
        tier_platform["tier_share"] / tier_platform["global_share"]
    )
    return tier_platform


def build_content_spread_summary(
    data: pd.DataFrame,
    tier_col: str = "popularity_tier",
    content_col: str = "content_original_id",
    platform_col: str = "platform",
) -> pd.DataFrame:
    """Summarize number of platforms per original content within each popularity tier."""
    spread = (
        data.groupby([tier_col, content_col], observed=True)
        .agg(n_platforms=(platform_col, "nunique"))
        .reset_index()
    )

    summary = (
        spread.groupby([tier_col, "n_platforms"], observed=True)
        .size()
        .rename("contents")
        .reset_index()
    )
    summary["tier_total_contents"] = summary.groupby(tier_col, observed=True)[
        "contents"
    ].transform("sum")
    summary["share"] = summary["contents"] / summary["tier_total_contents"]
    return summary
