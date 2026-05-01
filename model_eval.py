"""
Comprehensive model comparison across four statistical framings:
  1. Regression        — ElasticNet, QuantileRegressor (median), HistGB (Poisson loss)
  2. Count regression  — PoissonRegressor, NegativeBinomial (statsmodels wrapper)
  3. 4-class           — Very Short (1-3w), Short (4-10w), Medium (11-20w), Long (21+w)
  4. Binary            — short-lived (≤10w) vs long-lived (>10w)

All models use GroupKFold(k=5) split by artist.
Loads from pre-built feature_df_large_ts_features.csv so feature_engineering.py doesn't need to re-run.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    ElasticNet, QuantileRegressor, PoissonRegressor, LogisticRegression, Ridge
)
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, accuracy_score,
)

# =========================================================
# 1. LOAD + SCREEN FEATURES
# =========================================================

feature_df = pd.read_csv("data/processed_data/feature_df_large_ts_features.csv")

NON_FEATURE = {
    "song_id", "title", "artist", "release_date", "entry_week_date",
    "entry_week_pos", "peak_pos", "lifespan", "log_lifespan",
}

candidate_cols = [
    c for c in feature_df.columns
    if c not in NON_FEATURE
    and pd.api.types.is_numeric_dtype(feature_df[c])
    and feature_df[c].nunique(dropna=False) > 1
]

y_log    = np.log1p(feature_df["lifespan"].values.astype(float))
y_raw    = feature_df["lifespan"].values.astype(float)
groups   = feature_df["artist"].values

# Univariate screen against log_lifespan — top 120 by |Pearson r|
screen = []
for col in candidate_cols:
    x = feature_df[col]
    valid = x.notna() & pd.Series(y_log).notna()
    if valid.sum() < 100 or x[valid].std() == 0:
        continue
    r = np.corrcoef(x[valid].values, y_log[valid])[0, 1]
    if np.isfinite(r):
        screen.append((col, abs(r)))

screen.sort(key=lambda t: t[1], reverse=True)
features = [col for col, _ in screen[:120]]
top40    = [col for col, _ in screen[:40]]   # smaller set for NB stability

X    = feature_df[features].values.astype(float)
X40  = feature_df[top40].values.astype(float)

# =========================================================
# 2. TARGETS
# =========================================================

bins4   = [0, 3, 10, 20, y_raw.max() + 1]
labels4 = [0, 1, 2, 3]
y_bin4  = pd.cut(y_raw, bins=bins4, labels=labels4, include_lowest=True).astype(int)
y_bin2  = (y_raw > 10).astype(int)

label_names4 = {0: "Very Short (1-3w)", 1: "Short (4-10w)",
                2: "Medium (11-20w)",  3: "Long (21+w)"}

print(f"Features (full): {len(features)}  |  Features (top-40 for NB): {len(top40)}")
print(f"Binary positive rate: {y_bin2.mean():.1%}")
print("4-class distribution:", dict(zip(*np.unique(y_bin4, return_counts=True))))

# =========================================================
# 3. NEGATIVE BINOMIAL WRAPPER (statsmodels → sklearn API)
# =========================================================

class NegativeBinomialRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn wrapper around statsmodels NegativeBinomial GLM."""

    def fit(self, X, y):
        import statsmodels.api as sm
        Xc = sm.add_constant(np.nan_to_num(X, nan=0.0))
        self.result_ = sm.NegativeBinomial(y, Xc).fit(
            method="bfgs", maxiter=300, disp=False
        )
        return self

    def predict(self, X):
        import statsmodels.api as sm
        Xc = sm.add_constant(np.nan_to_num(X, nan=0.0), has_constant="add")
        return self.result_.predict(Xc)


# =========================================================
# 4. PIPELINE BUILDERS
# =========================================================

def linear_reg_pipe(model, feat_list):
    prep = ColumnTransformer([("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ]), list(range(len(feat_list))))])
    return Pipeline([("prep", prep), ("model", model)])


def tree_pipe(model, feat_list):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), list(range(len(feat_list))))
    ])
    return Pipeline([("prep", prep), ("model", model)])


# =========================================================
# 5. CV RUNNER
# =========================================================

cv = GroupKFold(n_splits=5)

def run_regression_cv(name, pipeline, X_data, y_target, raw_target=None):
    """
    Evaluates a regression model. Reports MAE/RMSE/R² on the original
    lifespan scale by inverse-transforming log predictions where needed.
    raw_target=True means the model predicts raw counts directly.
    """
    fold_mae, fold_rmse, fold_r2 = [], [], []

    for train_idx, test_idx in cv.split(X_data, groups=groups):
        X_tr, X_te = X_data[train_idx], X_data[test_idx]
        y_tr = y_target[train_idx]
        y_te_raw = (y_raw if raw_target else y_raw)[test_idx]

        pipeline.fit(X_tr, y_tr)
        pred = pipeline.predict(X_te)

        if not raw_target:
            pred = np.expm1(pred)

        pred = np.clip(pred, 0, None)

        fold_mae.append(mean_absolute_error(y_te_raw, pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_te_raw, pred)))
        fold_r2.append(r2_score(y_te_raw, pred))

    return {
        "model": name,
        "mean_mae":  np.mean(fold_mae),
        "std_mae":   np.std(fold_mae),
        "mean_rmse": np.mean(fold_rmse),
        "mean_r2":   np.mean(fold_r2),
        "std_r2":    np.std(fold_r2),
    }


def run_clf_cv(name, pipeline, X_data, y_target, binary=False):
    fold_acc, fold_f1w, fold_f1m, fold_auc = [], [], [], []

    for train_idx, test_idx in cv.split(X_data, groups=groups):
        X_tr, X_te = X_data[train_idx], X_data[test_idx]
        y_tr, y_te = y_target[train_idx], y_target[test_idx]

        pipeline.fit(X_tr, y_tr)
        pred = pipeline.predict(X_te)

        fold_acc.append(accuracy_score(y_te, pred))
        fold_f1w.append(f1_score(y_te, pred, average="weighted", zero_division=0))
        fold_f1m.append(f1_score(y_te, pred, average="macro",    zero_division=0))

        if binary:
            try:
                proba = pipeline.predict_proba(X_te)[:, 1]
                fold_auc.append(roc_auc_score(y_te, proba))
            except Exception:
                fold_auc.append(np.nan)

    result = {
        "model":       name,
        "accuracy":    np.mean(fold_acc),
        "macro_f1":    np.mean(fold_f1m),
        "weighted_f1": np.mean(fold_f1w),
    }
    if binary:
        result["roc_auc"] = np.mean(fold_auc)
    return result


# =========================================================
# 6. RUN ALL MODELS
# =========================================================

# ---------- 6A. REGRESSION (log1p target) ----------
print("\n" + "="*65)
print("REGRESSION  (predicting log1p lifespan, reported on raw scale)")
print("="*65)

reg_results = []
reg_models = [
    ("ElasticNet",
     linear_reg_pipe(ElasticNet(alpha=0.03, l1_ratio=0.5, random_state=42), features),
     X, y_log, False),
    ("Ridge",
     linear_reg_pipe(Ridge(alpha=1.0), features),
     X, y_log, False),
    ("QuantileRegressor (q=0.5)",
     linear_reg_pipe(QuantileRegressor(quantile=0.5, alpha=0.1, solver="highs"), features),
     X, y_log, False),
    ("HistGB (squared error)",
     tree_pipe(HistGradientBoostingRegressor(
         learning_rate=0.05, max_depth=6, max_iter=350,
         min_samples_leaf=20, random_state=42), features),
     X, y_log, False),
]

for name, pipe, Xd, yt, raw in reg_models:
    r = run_regression_cv(name, pipe, Xd, yt, raw_target=raw)
    reg_results.append(r)
    print(f"  {name:<35}  MAE={r['mean_mae']:.2f}  RMSE={r['mean_rmse']:.2f}  R²={r['mean_r2']:.3f} ± {r['std_r2']:.3f}")

# ---------- 6B. COUNT REGRESSION (raw target) ----------
print("\n" + "="*65)
print("COUNT REGRESSION  (raw lifespan target)")
print("="*65)

count_results = []
count_models = [
    ("PoissonRegressor",
     linear_reg_pipe(PoissonRegressor(alpha=1.0, max_iter=500), features),
     X, y_raw, True),
    ("HistGB (Poisson loss)",
     tree_pipe(HistGradientBoostingRegressor(
         loss="poisson", learning_rate=0.05, max_depth=4,
         max_iter=300, min_samples_leaf=20, random_state=42), features),
     X, y_raw, True),
    ("NegativeBinomial (top-40 features)",
     NegativeBinomialRegressor(),
     X40, y_raw, True),
]

for name, pipe, Xd, yt, raw in count_models:
    try:
        r = run_regression_cv(name, pipe, Xd, yt, raw_target=raw)
        count_results.append(r)
        print(f"  {name:<40}  MAE={r['mean_mae']:.2f}  RMSE={r['mean_rmse']:.2f}  R²={r['mean_r2']:.3f} ± {r['std_r2']:.3f}")
    except Exception as e:
        print(f"  {name:<40}  FAILED: {e}")

# ---------- 6C. 4-CLASS CLASSIFICATION ----------
print("\n" + "="*65)
print("4-CLASS CLASSIFICATION  (Very Short / Short / Medium / Long)")
print("="*65)
print("Baseline (predict majority class): accuracy ≈"
      f" {(y_bin4 == 0).mean():.1%}, macro-F1 ≈ 0.25")

clf4_results = []
clf4_models = [
    ("LogisticRegression (balanced)",
     linear_reg_pipe(LogisticRegression(
         class_weight="balanced", max_iter=1000, C=0.1, random_state=42), features)),
    ("RandomForest (balanced)",
     tree_pipe(RandomForestClassifier(
         n_estimators=400, max_depth=8, min_samples_leaf=4,
         class_weight="balanced", random_state=42, n_jobs=-1), features)),
    ("HistGradientBoosting",
     tree_pipe(HistGradientBoostingClassifier(
         learning_rate=0.05, max_depth=6, max_iter=350,
         min_samples_leaf=20, random_state=42), features)),
]

for name, pipe in clf4_models:
    r = run_clf_cv(name, pipe, X, y_bin4)
    clf4_results.append(r)
    print(f"  {name:<35}  acc={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}  weighted-F1={r['weighted_f1']:.3f}")

# ---------- 6D. BINARY CLASSIFICATION ----------
print("\n" + "="*65)
print("BINARY CLASSIFICATION  (short-lived ≤10w vs long-lived >10w)")
print("="*65)
print(f"Baseline (predict majority): accuracy ≈ {(y_bin2 == 0).mean():.1%}, AUC = 0.500")

clf2_results = []
clf2_models = [
    ("LogisticRegression (balanced)",
     linear_reg_pipe(LogisticRegression(
         class_weight="balanced", max_iter=1000, C=0.1, random_state=42), features)),
    ("RandomForest (balanced)",
     tree_pipe(RandomForestClassifier(
         n_estimators=400, max_depth=8, min_samples_leaf=4,
         class_weight="balanced", random_state=42, n_jobs=-1), features)),
    ("HistGradientBoosting",
     tree_pipe(HistGradientBoostingClassifier(
         learning_rate=0.05, max_depth=6, max_iter=350,
         min_samples_leaf=20, random_state=42), features)),
]

for name, pipe in clf2_models:
    r = run_clf_cv(name, pipe, X, y_bin2, binary=True)
    clf2_results.append(r)
    print(f"  {name:<35}  acc={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}  AUC={r.get('roc_auc', float('nan')):.3f}")

# =========================================================
# 7. SAVE SUMMARY
# =========================================================

pd.DataFrame(reg_results + count_results).to_csv(
    "data/processed_data/regression_comparison.csv", index=False
)
pd.DataFrame(clf4_results).to_csv(
    "data/processed_data/clf4_comparison.csv", index=False
)
pd.DataFrame(clf2_results).to_csv(
    "data/processed_data/clf2_comparison.csv", index=False
)
print("\nResults saved to data/processed_data/")
