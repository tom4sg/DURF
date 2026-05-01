"""
Step 1: Compare thresholds (5, 7, 10, 13 weeks) using a fixed RF to find
        which binary split is most learnable.
Step 2: Tune RandomForest on the best threshold using RandomizedSearchCV
        with GroupKFold — respects artist grouping throughout.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, make_scorer

# =========================================================
# 1. LOAD + FEATURE SCREENING (same as model_eval.py)
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

y_log  = np.log1p(feature_df["lifespan"].values.astype(float))
y_raw  = feature_df["lifespan"].values.astype(float)
groups = feature_df["artist"].values

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
X = feature_df[features].values.astype(float)

print(f"Features: {len(features)}, Songs: {len(y_raw)}\n")

# =========================================================
# 2. PIPELINE BUILDER
# =========================================================

def make_rf_pipeline(rf):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), list(range(len(features))))
    ])
    return Pipeline([("prep", prep), ("model", rf)])


cv5 = GroupKFold(n_splits=5)

# =========================================================
# 3. THRESHOLD SWEEP
# =========================================================

print("=" * 55)
print("THRESHOLD SWEEP  (fixed RF, optimising AUC)")
print("=" * 55)

fixed_rf = RandomForestClassifier(
    n_estimators=400, max_depth=8, min_samples_leaf=4,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
pipe_fixed = make_rf_pipeline(fixed_rf)

thresholds = [5, 7, 10, 13, 20]
threshold_results = []

for t in thresholds:
    y_t = (y_raw > t).astype(int)
    pos_rate = y_t.mean()

    aucs = []
    f1s  = []
    for train_idx, test_idx in cv5.split(X, groups=groups):
        pipe_fixed.fit(X[train_idx], y_t[train_idx])
        proba = pipe_fixed.predict_proba(X[test_idx])[:, 1]
        pred  = pipe_fixed.predict(X[test_idx])
        aucs.append(roc_auc_score(y_t[test_idx], proba))
        f1s.append(f1_score(y_t[test_idx], pred, zero_division=0))

    row = {
        "threshold": f"> {t}w",
        "pos_rate":  f"{pos_rate:.1%}",
        "mean_auc":  np.mean(aucs),
        "std_auc":   np.std(aucs),
        "mean_f1":   np.mean(f1s),
    }
    threshold_results.append(row)
    print(f"  > {t:2d}w  pos={pos_rate:.1%}  AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}  F1={np.mean(f1s):.3f}")

best_t = max(threshold_results, key=lambda r: r["mean_auc"])
best_threshold = thresholds[threshold_results.index(best_t)]
print(f"\n  Best threshold: > {best_threshold} weeks  (AUC={best_t['mean_auc']:.3f})")

# =========================================================
# 4. TUNE RF ON BEST THRESHOLD
# =========================================================

y_best = (y_raw > best_threshold).astype(int)

print(f"\n{'='*55}")
print(f"TUNING RandomForest  (target: > {best_threshold} weeks)")
print(f"{'='*55}")

param_dist = {
    "model__n_estimators":   randint(300, 1200),
    "model__max_depth":      [4, 6, 8, 10, 12, None],
    "model__min_samples_leaf": randint(2, 20),
    "model__max_features":   ["sqrt", "log2", 0.3, 0.5],
    "model__max_samples":    uniform(0.6, 0.4),       # bootstrap fraction
    "model__class_weight":   [
        "balanced",
        "balanced_subsample",
        {0: 1, 1: 3},
        {0: 1, 1: 4},
    ],
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
pipe_tune = make_rf_pipeline(base_rf)

search = RandomizedSearchCV(
    pipe_tune,
    param_distributions=param_dist,
    n_iter=60,
    cv=cv5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True,
)

search.fit(X, y_best, groups=groups)

print(f"\n  Best AUC (CV):  {search.best_score_:.3f}")
print(f"  Best params:")
for k, v in search.best_params_.items():
    print(f"    {k.replace('model__',''):<22} = {v}")

# Full CV with best model to get F1 + AUC together
best_pipe = search.best_estimator_

fold_auc, fold_f1, fold_acc = [], [], []
for train_idx, test_idx in cv5.split(X, groups=groups):
    best_pipe.fit(X[train_idx], y_best[train_idx])
    proba = best_pipe.predict_proba(X[test_idx])[:, 1]
    pred  = best_pipe.predict(X[test_idx])
    fold_auc.append(roc_auc_score(y_best[test_idx], proba))
    fold_f1.append(f1_score(y_best[test_idx], pred, zero_division=0))
    fold_acc.append((pred == y_best[test_idx]).mean())

print(f"\n  Tuned RF — final CV results:")
print(f"    AUC      = {np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}")
print(f"    F1       = {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")
print(f"    Accuracy = {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}")
print(f"\n  Baseline (majority class): accuracy = {(y_best==0).mean():.3f},  AUC = 0.500")

# =========================================================
# 5. SAVE
# =========================================================

results_df = pd.DataFrame(threshold_results)
results_df.to_csv("data/processed_data/threshold_sweep.csv", index=False)

best_params_df = pd.DataFrame([{
    k.replace("model__", ""): v
    for k, v in search.best_params_.items()
} | {"best_cv_auc": search.best_score_}])
best_params_df.to_csv("data/processed_data/rf_best_params.csv", index=False)

print("\nSaved threshold_sweep.csv and rf_best_params.csv")
