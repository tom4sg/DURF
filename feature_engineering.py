#%%

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# =========================================================
# 1. LOAD
# =========================================================

songs = pd.read_csv("data/processed_data/songs.csv")
ig = pd.read_csv("data/processed_data/12m_ig_pre_release_imputed.csv")
tt = pd.read_csv("data/processed_data/12m_tt_pre_release_imputed.csv")
yt = pd.read_csv("data/processed_data/12m_yt_pre_release_imputed.csv")
social_handles = pd.read_csv("data/processed_data/social_handles.csv")
ig_post = pd.read_csv("data/processed_data/2w_ig_post_release.csv")
tt_post = pd.read_csv("data/processed_data/2w_tt_post_release.csv")
yt_post = pd.read_csv("data/processed_data/2w_yt_post_release.csv")

songs["release_date"] = pd.to_datetime(songs["release_date"], errors="coerce")

for df in [ig, tt, yt, ig_post, tt_post, yt_post]:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")



# 2. HELPERS

def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    out = np.full_like(a, np.nan, dtype=float)
    valid = (~np.isnan(a)) & (~np.isnan(b)) & (b != 0)
    out[valid] = a[valid] / b[valid]
    return out


def add_platform_flags(handles: pd.DataFrame) -> pd.DataFrame:
    handles = handles.copy()
    handles["has_ig"] = handles["ig_handle"].notna()
    handles["has_tt"] = handles["tt_handle"].notna()
    handles["has_yt"] = handles["yt_handle"].notna()
    handles["has_social_media"] = handles[["has_ig", "has_tt", "has_yt"]].any(axis=1)
    return handles


def get_release_snapshot(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    out = (
        df.loc[df["date"] == df["release_date"], ["song_id", "release_date"] + value_cols]
        .drop_duplicates(["song_id", "release_date"])
        .rename(columns={col: f"{prefix}_{col}_release_date" for col in value_cols})
    )
    return out


def linear_slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or np.all(np.isnan(y)):
        return np.nan
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    return np.polyfit(x, y, 1)[0]


def coeff_var(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return np.nan
    mu = np.mean(y)
    sd = np.std(y, ddof=1)
    if mu == 0:
        return np.nan
    return sd / abs(mu)


def max_drawdown(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return np.nan

    running_max = np.maximum.accumulate(y)
    valid = running_max > 0
    if valid.sum() == 0:
        return np.nan

    dd = np.full_like(y, np.nan, dtype=float)
    dd[valid] = (running_max[valid] - y[valid]) / running_max[valid]

    if np.all(np.isnan(dd)):
        return np.nan

    return np.nanmax(dd)


def longest_streak_positive(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    best = 0
    cur = 0
    for val in arr:
        if val > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def longest_streak_nonpositive(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    best = 0
    cur = 0
    for val in arr:
        if val <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def window_summary(series: pd.Series, prefix: str, window_name: str) -> dict:
    s = series.astype(float).dropna()
    out = {}

    base = f"{prefix}_{window_name}"

    if len(s) == 0:
        return out

    vals = s.values
    diffs = np.diff(vals) if len(vals) >= 2 else np.array([])
    pct = pd.Series(vals).pct_change().replace([np.inf, -np.inf], np.nan).dropna().values

    out[f"{base}_last"] = vals[-1]
    out[f"{base}_mean"] = np.mean(vals)
    out[f"{base}_std"] = np.std(vals, ddof=1) if len(vals) >= 2 else 0.0
    out[f"{base}_min"] = np.min(vals)
    out[f"{base}_max"] = np.max(vals)
    out[f"{base}_range"] = np.max(vals) - np.min(vals)
    out[f"{base}_slope"] = linear_slope(vals)
    out[f"{base}_cv"] = coeff_var(vals)
    out[f"{base}_drawdown"] = max_drawdown(vals)

    out[f"{base}_abs_change"] = vals[-1] - vals[0]
    out[f"{base}_pct_change"] = safe_divide(vals[-1] - vals[0], vals[0])

    if len(diffs) > 0:
        out[f"{base}_diff_mean"] = np.mean(diffs)
        out[f"{base}_diff_std"] = np.std(diffs, ddof=1) if len(diffs) >= 2 else 0.0
        out[f"{base}_diff_max"] = np.max(diffs)
        out[f"{base}_diff_min"] = np.min(diffs)
        out[f"{base}_jump_ratio"] = safe_divide(np.max(diffs), np.std(diffs, ddof=1) if len(diffs) >= 2 else np.nan)
        out[f"{base}_pos_streak"] = longest_streak_positive(diffs)
        out[f"{base}_nonpos_streak"] = longest_streak_nonpositive(diffs)

    if len(pct) > 0:
        out[f"{base}_pct_mean"] = np.nanmean(pct)
        out[f"{base}_pct_std"] = np.nanstd(pct, ddof=1) if len(pct) >= 2 else 0.0
        out[f"{base}_pct_max"] = np.nanmax(pct)
        out[f"{base}_pct_min"] = np.nanmin(pct)

    return out


def build_platform_time_series_features(
    df: pd.DataFrame,
    platform_prefix: str,
    metrics: list[str],
    windows=(30, 90, 180, 365),
) -> pd.DataFrame:
    """
    Builds many pre-release time-series features per song from raw daily histories.
    """
    df = df.copy()
    df = df.sort_values(["song_id", "date"])

    all_rows = []

    for (song_id, release_date), g in df.groupby(["song_id", "release_date"], dropna=False):
        g = g.copy()
        g = g[g["date"] <= g["release_date"]].sort_values("date")

        row = {
            "song_id": song_id,
            "release_date": release_date,
        }

        for metric in metrics:
            metric_series = g[["date", metric]].dropna().sort_values("date")
            if metric_series.empty:
                continue

            metric_series = metric_series.drop_duplicates("date", keep="last")

            for w in windows:
                cutoff = release_date - pd.Timedelta(days=w)
                sub = metric_series[metric_series["date"] >= cutoff][metric]

                feats = window_summary(
                    sub,
                    prefix=f"{platform_prefix}_{metric}",
                    window_name=f"{w}d"
                )
                row.update(feats)

            # All-history-in-window summary
            full = metric_series[metric]
            feats_full = window_summary(
                full,
                prefix=f"{platform_prefix}_{metric}",
                window_name="full"
            )
            row.update(feats_full)

            # Distance from recent max at release
            vals = metric_series[metric].astype(float).values
            if len(vals) > 0:
                row[f"{platform_prefix}_{metric}_release_vs_max_full"] = safe_divide(vals[-1], np.nanmax(vals))
                row[f"{platform_prefix}_{metric}_release_vs_mean_full"] = safe_divide(vals[-1], np.nanmean(vals))

            # Recent trend comparisons
            last_30 = metric_series[metric_series["date"] >= release_date - pd.Timedelta(days=30)][metric].astype(float).dropna()
            last_90 = metric_series[metric_series["date"] >= release_date - pd.Timedelta(days=90)][metric].astype(float).dropna()
            last_180 = metric_series[metric_series["date"] >= release_date - pd.Timedelta(days=180)][metric].astype(float).dropna()

            if len(last_30) > 0 and len(last_90) > 0:
                row[f"{platform_prefix}_{metric}_30d_mean_over_90d_mean"] = safe_divide(last_30.mean(), last_90.mean())
                row[f"{platform_prefix}_{metric}_30d_slope_over_90d_slope"] = safe_divide(
                    linear_slope(last_30.values), linear_slope(last_90.values)
                )

            if len(last_30) > 0 and len(last_180) > 0:
                row[f"{platform_prefix}_{metric}_30d_mean_over_180d_mean"] = safe_divide(last_30.mean(), last_180.mean())

        all_rows.append(row)

    return pd.DataFrame(all_rows)


def build_post_release_features(
    df: pd.DataFrame,
    platform_prefix: str,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Builds summary features from the 2-week post-release social window.
    Reuses window_summary so the feature names are consistent with pre-release ones.
    """
    df = df.copy()
    df = df.sort_values(["song_id", "date"])

    all_rows = []

    for (song_id, release_date), g in df.groupby(["song_id", "release_date"], dropna=False):
        g = g[g["date"] > g["release_date"]].sort_values("date")

        row = {"song_id": song_id, "release_date": release_date}

        for metric in metrics:
            metric_series = g[["date", metric]].dropna().sort_values("date")
            if metric_series.empty:
                continue
            metric_series = metric_series.drop_duplicates("date", keep="last")
            feats = window_summary(
                metric_series[metric],
                prefix=f"{platform_prefix}_{metric}",
                window_name="post2w",
            )
            row.update(feats)

        all_rows.append(row)

    return pd.DataFrame(all_rows)


def add_cross_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Release-level cross-platform ratios
    pairs = [
        ("tt_likes_release_date", "tt_followers_release_date"),
        ("yt_views_release_date", "yt_subs_release_date"),
        ("ig_media_release_date", "ig_followers_release_date"),
        ("tt_likes_release_date", "yt_views_release_date"),
        ("tt_followers_release_date", "ig_followers_release_date"),
        ("yt_subs_release_date", "ig_followers_release_date"),
    ]

    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_over_{b}"] = safe_divide(df[a], df[b])

    # Time-series interaction features from key windows if present
    interaction_pairs = [
        ("tt_likes_30d_slope", "tt_followers_30d_slope"),
        ("yt_views_30d_slope", "yt_subs_30d_slope"),
        ("ig_followers_30d_slope", "tt_followers_30d_slope"),
        ("ig_followers_30d_pct_change", "tt_likes_30d_pct_change"),
        ("yt_views_90d_pct_change", "tt_likes_90d_pct_change"),
    ]

    # Match actual generated names
    resolved_candidates = []
    for a, b in interaction_pairs:
        col_a = None
        col_b = None
        for col in df.columns:
            if col.endswith(a):
                col_a = col
            if col.endswith(b):
                col_b = col
        if col_a is not None and col_b is not None:
            resolved_candidates.append((col_a, col_b))

    for a, b in resolved_candidates:
        df[f"{a}_x_{b}"] = df[a] * df[b]
        df[f"{a}_over_{b}"] = safe_divide(df[a], df[b])

    return df


def build_scaled_linear_pipeline(features: list[str], model):
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                features,
            )
        ]
    )
    return Pipeline([
        ("prep", prep),
        ("model", model),
    ])


def build_tree_pipeline(features: list[str], model):
    prep = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), features)
        ]
    )
    return Pipeline([
        ("prep", prep),
        ("model", model),
    ])


def get_models(features: list[str]):
    return {
        "elastic_net": build_scaled_linear_pipeline(
            features,
            ElasticNet(alpha=0.03, l1_ratio=0.5, random_state=42)
        ),
        "random_forest": build_tree_pipeline(
            features,
            RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "extra_trees": build_tree_pipeline(
            features,
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": build_tree_pipeline(
            features,
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        ),
        "hist_gradient_boosting": build_tree_pipeline(
            features,
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=350,
                min_samples_leaf=20,
                random_state=42,
            )
        ),
    }


# =========================================================
# 3. BASE FEATURE TABLE
# =========================================================

feature_df = songs.copy()

social_handles = add_platform_flags(social_handles)
feature_df = feature_df.merge(
    social_handles[["artist", "has_ig", "has_tt", "has_yt", "has_social_media"]],
    on="artist",
    how="left",
)

ig_release = get_release_snapshot(ig, ["followers", "media"], "ig")
tt_release = get_release_snapshot(tt, ["followers", "uploads", "likes"], "tt")
yt_release = get_release_snapshot(yt, ["subs", "views"], "yt")

for part in [ig_release, tt_release, yt_release]:
    feature_df = feature_df.merge(part, on=["song_id", "release_date"], how="left")

# Simple ratio features
feature_df["tt_likes_per_follower"] = safe_divide(
    feature_df["tt_likes_release_date"], feature_df["tt_followers_release_date"]
)
feature_df["tt_likes_per_upload"] = safe_divide(
    feature_df["tt_likes_release_date"], feature_df["tt_uploads_release_date"]
)
feature_df["yt_views_per_sub"] = safe_divide(
    feature_df["yt_views_release_date"], feature_df["yt_subs_release_date"]
)
feature_df["ig_media_per_follower"] = safe_divide(
    feature_df["ig_media_release_date"], feature_df["ig_followers_release_date"]
)
feature_df["log_tt_likes_per_upload"] = (
    np.log1p(feature_df["tt_likes_release_date"]) -
    np.log1p(feature_df["tt_uploads_release_date"])
)

# Log scale features
for col in [
    "ig_followers_release_date",
    "ig_media_release_date",
    "tt_followers_release_date",
    "tt_uploads_release_date",
    "tt_likes_release_date",
    "yt_subs_release_date",
    "yt_views_release_date",
]:
    if col in feature_df.columns:
        feature_df[f"log_{col}"] = np.log1p(feature_df[col])

# Regression target
feature_df["log_lifespan"] = np.log1p(feature_df["lifespan"])


# =========================================================
# 4. BUILD LARGE TIME-SERIES FEATURE BLOCKS
# =========================================================

ig_ts = build_platform_time_series_features(
    ig,
    platform_prefix="ig",
    metrics=["followers", "media"],
    windows=(30, 90, 180, 365),
)

tt_ts = build_platform_time_series_features(
    tt,
    platform_prefix="tt",
    metrics=["followers", "likes", "uploads"],
    windows=(30, 90, 180, 365),
)

yt_ts = build_platform_time_series_features(
    yt,
    platform_prefix="yt",
    metrics=["subs", "views"],
    windows=(30, 90, 180, 365),
)

for part in [ig_ts, tt_ts, yt_ts]:
    feature_df = feature_df.merge(part, on=["song_id", "release_date"], how="left")

feature_df = add_cross_platform_features(feature_df)

# =========================================================
# 4C. POST-RELEASE FEATURE BLOCK (2-week window)
# =========================================================

ig_post_feats = build_post_release_features(ig_post, "ig", ["followers", "media"])
tt_post_feats = build_post_release_features(tt_post, "tt", ["followers", "likes", "uploads"])
yt_post_feats = build_post_release_features(yt_post, "yt", ["subs", "views"])

for part in [ig_post_feats, tt_post_feats, yt_post_feats]:
    feature_df = feature_df.merge(part, on=["song_id", "release_date"], how="left")

# =========================================================
# 4B. ROUND-2 FEATURE BLOCK: RECENCY, ACCELERATION, STABILITY
# Put this RIGHT AFTER:
# feature_df = add_cross_platform_features(feature_df)
# =========================================================

def add_round2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------- helpers --------
    def add_if_exists(new_col, expr):
        try:
            df[new_col] = expr
        except KeyError:
            pass

    def col_exists(*cols):
        return all(c in df.columns for c in cols)

    # =====================================================
    # A. ACCELERATION FEATURES
    # Compare short-run momentum to medium/long-run momentum
    # =====================================================

    for platform, metric in [
        ("ig", "followers"),
        ("ig", "media"),
        ("tt", "followers"),
        ("tt", "likes"),
        ("tt", "uploads"),
        ("yt", "subs"),
        ("yt", "views"),
    ]:
        p30_slope = f"{platform}_{metric}_30d_slope"
        p90_slope = f"{platform}_{metric}_90d_slope"
        p180_slope = f"{platform}_{metric}_180d_slope"

        p30_pct = f"{platform}_{metric}_30d_pct_change"
        p90_pct = f"{platform}_{metric}_90d_pct_change"
        p180_pct = f"{platform}_{metric}_180d_pct_change"

        p30_diff_mean = f"{platform}_{metric}_30d_diff_mean"
        p90_diff_mean = f"{platform}_{metric}_90d_diff_mean"

        if col_exists(p30_slope, p90_slope):
            add_if_exists(
                f"{platform}_{metric}_accel_slope_30_vs_90",
                df[p30_slope] - df[p90_slope]
            )
            add_if_exists(
                f"{platform}_{metric}_accel_slope_ratio_30_vs_90",
                safe_divide(df[p30_slope], df[p90_slope])
            )

        if col_exists(p30_slope, p180_slope):
            add_if_exists(
                f"{platform}_{metric}_accel_slope_30_vs_180",
                df[p30_slope] - df[p180_slope]
            )
            add_if_exists(
                f"{platform}_{metric}_accel_slope_ratio_30_vs_180",
                safe_divide(df[p30_slope], df[p180_slope])
            )

        if col_exists(p30_pct, p90_pct):
            add_if_exists(
                f"{platform}_{metric}_accel_pct_30_vs_90",
                df[p30_pct] - df[p90_pct]
            )

        if col_exists(p30_pct, p180_pct):
            add_if_exists(
                f"{platform}_{metric}_accel_pct_30_vs_180",
                df[p30_pct] - df[p180_pct]
            )

        if col_exists(p30_diff_mean, p90_diff_mean):
            add_if_exists(
                f"{platform}_{metric}_accel_diffmean_30_vs_90",
                df[p30_diff_mean] - df[p90_diff_mean]
            )

    # =====================================================
    # B. RECENCY-WEIGHTED / LATE PICKUP FEATURES
    # Short window relative to longer window
    # =====================================================

    for platform, metric in [
        ("ig", "followers"),
        ("ig", "media"),
        ("tt", "followers"),
        ("tt", "likes"),
        ("tt", "uploads"),
        ("yt", "subs"),
        ("yt", "views"),
    ]:
        p30_mean = f"{platform}_{metric}_30d_mean"
        p90_mean = f"{platform}_{metric}_90d_mean"
        p180_mean = f"{platform}_{metric}_180d_mean"

        p30_std = f"{platform}_{metric}_30d_std"
        p90_std = f"{platform}_{metric}_90d_std"

        p30_cv = f"{platform}_{metric}_30d_cv"
        p90_cv = f"{platform}_{metric}_90d_cv"

        p30_last = f"{platform}_{metric}_30d_last"
        p90_max = f"{platform}_{metric}_90d_max"
        p180_max = f"{platform}_{metric}_180d_max"

        if col_exists(p30_mean, p90_mean):
            add_if_exists(
                f"{platform}_{metric}_recency_mean_30_over_90",
                safe_divide(df[p30_mean], df[p90_mean])
            )

        if col_exists(p30_mean, p180_mean):
            add_if_exists(
                f"{platform}_{metric}_recency_mean_30_over_180",
                safe_divide(df[p30_mean], df[p180_mean])
            )

        if col_exists(p30_std, p90_std):
            add_if_exists(
                f"{platform}_{metric}_recency_std_30_over_90",
                safe_divide(df[p30_std], df[p90_std])
            )

        if col_exists(p30_cv, p90_cv):
            add_if_exists(
                f"{platform}_{metric}_recency_cv_30_over_90",
                safe_divide(df[p30_cv], df[p90_cv])
            )

        if col_exists(p30_last, p90_max):
            add_if_exists(
                f"{platform}_{metric}_release_vs_90d_max",
                safe_divide(df[p30_last], df[p90_max])
            )

        if col_exists(p30_last, p180_max):
            add_if_exists(
                f"{platform}_{metric}_release_vs_180d_max",
                safe_divide(df[p30_last], df[p180_max])
            )

    # =====================================================
    # C. STABILITY / BURSTINESS FEATURES
    # Measures whether growth is steady vs spiky / stagnant
    # =====================================================

    for platform, metric in [
        ("ig", "followers"),
        ("ig", "media"),
        ("tt", "followers"),
        ("tt", "likes"),
        ("tt", "uploads"),
        ("yt", "subs"),
        ("yt", "views"),
    ]:
        p30_diff_max = f"{platform}_{metric}_30d_diff_max"
        p30_diff_std = f"{platform}_{metric}_30d_diff_std"
        p30_diff_mean = f"{platform}_{metric}_30d_diff_mean"
        p30_abs_change = f"{platform}_{metric}_30d_abs_change"
        p30_range = f"{platform}_{metric}_30d_range"
        p30_mean = f"{platform}_{metric}_30d_mean"
        p30_nonpos = f"{platform}_{metric}_30d_nonpos_streak"
        p30_pos = f"{platform}_{metric}_30d_pos_streak"

        p90_diff_max = f"{platform}_{metric}_90d_diff_max"
        p90_diff_std = f"{platform}_{metric}_90d_diff_std"
        p90_abs_change = f"{platform}_{metric}_90d_abs_change"
        p90_range = f"{platform}_{metric}_90d_range"
        p90_mean = f"{platform}_{metric}_90d_mean"
        p90_nonpos = f"{platform}_{metric}_90d_nonpos_streak"
        p90_pos = f"{platform}_{metric}_90d_pos_streak"

        if col_exists(p30_diff_max, p30_diff_std):
            add_if_exists(
                f"{platform}_{metric}_burstiness_30d",
                safe_divide(df[p30_diff_max], df[p30_diff_std])
            )

        if col_exists(p90_diff_max, p90_diff_std):
            add_if_exists(
                f"{platform}_{metric}_burstiness_90d",
                safe_divide(df[p90_diff_max], df[p90_diff_std])
            )

        if col_exists(p30_abs_change, p30_range):
            add_if_exists(
                f"{platform}_{metric}_trend_efficiency_30d",
                safe_divide(df[p30_abs_change], df[p30_range])
            )

        if col_exists(p90_abs_change, p90_range):
            add_if_exists(
                f"{platform}_{metric}_trend_efficiency_90d",
                safe_divide(df[p90_abs_change], df[p90_range])
            )

        if col_exists(p30_abs_change, p30_mean):
            add_if_exists(
                f"{platform}_{metric}_norm_abs_change_30d",
                safe_divide(df[p30_abs_change], df[p30_mean])
            )

        if col_exists(p90_abs_change, p90_mean):
            add_if_exists(
                f"{platform}_{metric}_norm_abs_change_90d",
                safe_divide(df[p90_abs_change], df[p90_mean])
            )

        if col_exists(p30_nonpos, p30_pos):
            add_if_exists(
                f"{platform}_{metric}_streak_balance_30d",
                safe_divide(df[p30_pos], df[p30_nonpos] + 1)
            )

        if col_exists(p90_nonpos, p90_pos):
            add_if_exists(
                f"{platform}_{metric}_streak_balance_90d",
                safe_divide(df[p90_pos], df[p90_nonpos] + 1)
            )

        if col_exists(p30_nonpos, p30_mean):
            add_if_exists(
                f"{platform}_{metric}_stagnation_over_level_30d",
                safe_divide(df[p30_nonpos], df[p30_mean])
            )

    # =====================================================
    # D. CROSS-PLATFORM CONCENTRATION / DOMINANCE
    # Relative platform strength rather than raw size
    # =====================================================

    # release-level totals
    release_scale_cols = [
        "ig_followers_release_date",
        "tt_followers_release_date",
        "yt_subs_release_date",
    ]
    existing_release_scale_cols = [c for c in release_scale_cols if c in df.columns]

    if len(existing_release_scale_cols) >= 2:
        total_release_audience = df[existing_release_scale_cols].sum(axis=1, min_count=1)
        df["total_release_audience_main3"] = total_release_audience

        for c in existing_release_scale_cols:
            df[f"{c}_share_main3"] = safe_divide(df[c], total_release_audience)

    # platform content / attention shares
    attention_cols = [
        "tt_likes_release_date",
        "yt_views_release_date",
    ]
    existing_attention_cols = [c for c in attention_cols if c in df.columns]

    if len(existing_attention_cols) >= 2:
        total_attention = df[existing_attention_cols].sum(axis=1, min_count=1)
        df["total_release_attention_tt_yt"] = total_attention

        for c in existing_attention_cols:
            df[f"{c}_share_tt_yt"] = safe_divide(df[c], total_attention)

    # =====================================================
    # E. TARGETED MULTIPLICATIONS
    # Only a few, based on patterns already found
    # =====================================================

    mult_pairs = [
        ("log_ig_followers_release_date", "ig_media_per_follower"),
        ("log_tt_uploads_release_date", "tt_likes_per_follower"),
        ("log_yt_views_release_date", "yt_views_per_sub"),
        ("entry_week_pos", "tt_likes_per_follower"),
        ("entry_week_pos", "ig_media_per_follower"),
    ]

    for a, b in mult_pairs:
        if col_exists(a, b):
            df[f"{a}_x_{b}"] = df[a] * df[b]

    return df


feature_df = add_round2_features(feature_df)

# =========================================================
# 5. FEATURE POOLS
# =========================================================

_songs_meta_cols = {
    "song_id", "title", "artist", "release_date", "entry_week_date",
    "entry_week_pos", "peak_pos", "lifespan", "song_length",
}
genre_features = [
    c for c in songs.columns
    if c not in _songs_meta_cols
    and c in feature_df.columns
]

baseline_features = [
    c for c in [
        "song_length",
        "has_ig",
        "has_tt",
        "has_yt",
        "log_ig_followers_release_date",
        "log_ig_media_release_date",
        "log_tt_followers_release_date",
        "log_tt_uploads_release_date",
        "log_tt_likes_release_date",
        "log_yt_subs_release_date",
        "log_yt_views_release_date",
        "tt_likes_per_follower",
        "tt_likes_per_upload",
        "log_tt_likes_per_upload",
        "yt_views_per_sub",
        "ig_media_per_follower",
    ] if c in feature_df.columns
] + genre_features

ts_features = [
    c for c in feature_df.columns
    if (
        any(c.startswith(prefix) for prefix in ["ig_", "tt_", "yt_"])
        and any(tag in c for tag in ["30d", "90d", "180d", "365d", "full", "post2w"])
        and c not in {"song_id", "release_date", "lifespan", "log_lifespan"}
    )
]

# Remove IDs and target-like columns accidentally captured
ts_features = [
    c for c in ts_features
    if c not in {"song_id", "release_date", "lifespan", "log_lifespan"}
]

interaction_features = [
    c for c in feature_df.columns
    if "_x_" in c or "_over_" in c
]
interaction_features = [
    c for c in interaction_features
    if c not in baseline_features and c not in ts_features
]

all_feature_pool = baseline_features + ts_features + interaction_features
all_feature_pool = [c for c in all_feature_pool if c in feature_df.columns]

# Keep only numeric columns
all_feature_pool = [
    c for c in all_feature_pool
    if pd.api.types.is_numeric_dtype(feature_df[c])
]

groups = feature_df["artist"]
y = feature_df["log_lifespan"]

# Optional: drop obviously constant columns
nunique = feature_df[all_feature_pool].nunique(dropna=False)
all_feature_pool = [c for c in all_feature_pool if nunique[c] > 1]


# =========================================================
# 6. UNIVARIATE SCREENING
# =========================================================

screen_rows = []

for col in all_feature_pool:
    x = feature_df[col]
    valid = x.notna() & y.notna()
    if valid.sum() < 30:
        continue

    corr = np.corrcoef(x[valid], y[valid])[0, 1]
    screen_rows.append({
        "feature": col,
        "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
        "corr": corr,
        "missing_rate": x.isna().mean(),
        "n_valid": int(valid.sum()),
    })

screen_df = pd.DataFrame(screen_rows)
screen_df = screen_df[
    (screen_df["n_valid"] >= 100) &
    (screen_df["missing_rate"] <= 0.60)
].sort_values("abs_corr", ascending=False)

print("\nTop univariate correlations:")
print(screen_df.head(40))

# Keep top-N by abs correlation for manageable broad search
top_n = 120
top_screened_features = screen_df["feature"].head(top_n).tolist()

# Force-keep baseline anchor features if available
forced_keep = [c for c in baseline_features if c in all_feature_pool]
screened_feature_pool = list(dict.fromkeys(forced_keep + top_screened_features))


# =========================================================
# 7. FEATURE SET COMPARISONS
# =========================================================

feature_sets = {
    "baseline_only": baseline_features,
    "time_series_only": [c for c in ts_features if c in screened_feature_pool],
    "interactions_only": [c for c in interaction_features if c in screened_feature_pool],
    "baseline_plus_top_ts": list(dict.fromkeys(
        baseline_features + [c for c in ts_features if c in screened_feature_pool]
    )),
    "baseline_plus_top_interactions": list(dict.fromkeys(
        baseline_features + [c for c in interaction_features if c in screened_feature_pool]
    )),
    "baseline_plus_all_screened": screened_feature_pool,
    "all_screened_no_entry_week_pos": [c for c in screened_feature_pool if c != "entry_week_pos"],
}

feature_sets = {
    name: [c for c in cols if c in feature_df.columns]
    for name, cols in feature_sets.items()
}
feature_sets = {name: cols for name, cols in feature_sets.items() if len(cols) > 0}

cv = GroupKFold(n_splits=5)

scoring = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False
    ),
    "r2": make_scorer(r2_score),
}

results = []

for set_name, features in feature_sets.items():
    X_set = feature_df[features]

    for model_name, model in get_models(features).items():
        scores = cross_validate(
            model,
            X_set,
            y,
            cv=cv,
            groups=groups,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append({
            "feature_set": set_name,
            "model": model_name,
            "n_features": len(features),
            "mean_mae": -scores["test_mae"].mean(),
            "std_mae": scores["test_mae"].std(),
            "mean_rmse": -scores["test_rmse"].mean(),
            "std_rmse": scores["test_rmse"].std(),
            "mean_r2": scores["test_r2"].mean(),
            "std_r2": scores["test_r2"].std(),
        })

results_df = pd.DataFrame(results).sort_values(["mean_mae", "mean_r2"], ascending=[True, False])

print("\nFeature set comparison:")
print(results_df)


# =========================================================
# 8. BEST MODEL + PERMUTATION IMPORTANCE
# =========================================================

best_row = results_df.iloc[0]
best_feature_set_name = best_row["feature_set"]
best_model_name = best_row["model"]
best_features = feature_sets[best_feature_set_name]

print("\nBest feature set:", best_feature_set_name)
print("Best model:", best_model_name)
print("Number of features:", len(best_features))

models_for_best = get_models(best_features)
best_model = models_for_best[best_model_name]

X_best = feature_df[best_features]
best_model.fit(X_best, y)

perm = permutation_importance(
    best_model,
    X_best,
    y,
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
    scoring="r2",
)

perm_df = pd.DataFrame({
    "feature": best_features,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

print("\nTop permutation importance:")
print(perm_df.head(40))


# =========================================================
# 9. INCREMENTAL VALUE OVER ENTRY-WEEK BASELINE
# =========================================================

def unique_cols(cols):
    return list(dict.fromkeys(cols))

incremental_sets = {
    "entry_only": unique_cols([c for c in ["entry_week_pos"] if c in feature_df.columns]),
    "entry_plus_baseline": unique_cols([c for c in (["entry_week_pos"] + baseline_features) if c in feature_df.columns]),
    "entry_plus_top_ts": unique_cols([c for c in (["entry_week_pos"] + [x for x in ts_features if x in screened_feature_pool]) if c in feature_df.columns]),
    "entry_plus_top_interactions": unique_cols([c for c in (["entry_week_pos"] + [x for x in interaction_features if x in screened_feature_pool]) if c in feature_df.columns]),
    "entry_plus_all_screened": unique_cols([c for c in (["entry_week_pos"] + screened_feature_pool) if c in feature_df.columns]),
}

incremental_rows = []

for set_name, features in incremental_sets.items():
    if len(features) == 0:
        continue

    X_set = feature_df[features]
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=20,
        random_state=42,
    )
    pipeline = build_tree_pipeline(features, model)

    scores = cross_validate(
        pipeline,
        X_set,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        n_jobs=-1,
    )

    incremental_rows.append({
        "feature_set": set_name,
        "n_features": len(features),
        "mean_mae": -scores["test_mae"].mean(),
        "mean_r2": scores["test_r2"].mean(),
    })

incremental_df = pd.DataFrame(incremental_rows).sort_values("mean_mae")

print("\nIncremental value over entry-week baseline:")
print(incremental_df)


# =========================================================
# 10. OPTIONAL SAVES
# =========================================================

feature_df.to_csv("data/processed_data/feature_df_large_ts_features.csv", index=False)
screen_df.to_csv("data/processed_data/feature_screening_results.csv", index=False)
results_df.to_csv("data/processed_data/time_series_feature_model_results.csv", index=False)
perm_df.to_csv("data/processed_data/time_series_best_model_permutation_importance.csv", index=False)
incremental_df.to_csv("data/processed_data/time_series_incremental_over_entry.csv", index=False)
# %%
