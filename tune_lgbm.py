"""
1. Tune LightGBM on binary target (> 7 weeks)
2. Prune features using RF importance — try top 20, 30, 40
3. Re-run tuned RF + tuned LGBM on each pruned set
4. Report best overall combination
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb

# =========================================================
# 1. LOAD + SCREEN (same as tune_rf.py)
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
y_bin  = (y_raw > 7).astype(int)
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

print(f"Songs: {len(y_raw)}  |  Features: {len(features)}")
print(f"Positive rate (>7w): {y_bin.mean():.1%}\n")

cv5 = GroupKFold(n_splits=5)


# =========================================================
# 2. HELPERS
# =========================================================

def make_rf_pipe(rf, n_feats):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), list(range(n_feats)))
    ])
    return Pipeline([("prep", prep), ("model", rf)])


def make_lgbm_pipe(clf, n_feats):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), list(range(n_feats)))
    ])
    return Pipeline([("prep", prep), ("model", clf)])


def cv_auc_f1(pipeline, Xd, y, grps):
    aucs, f1s = [], []
    for tr, te in cv5.split(Xd, groups=grps):
        pipeline.fit(Xd[tr], y[tr])
        proba = pipeline.predict_proba(Xd[te])[:, 1]
        pred  = pipeline.predict(Xd[te])
        aucs.append(roc_auc_score(y[te], proba))
        f1s.append(f1_score(y[te], pred, zero_division=0))
    return np.mean(aucs), np.std(aucs), np.mean(f1s)


# =========================================================
# 3. TUNE LIGHTGBM (full 120 features)
# =========================================================

print("=" * 60)
print("TUNING LightGBM  (target: > 7 weeks, 120 features)")
print("=" * 60)

lgbm_param_dist = {
    "model__n_estimators":      randint(200, 1200),
    "model__learning_rate":     uniform(0.01, 0.09),
    "model__max_depth":         [-1, 4, 6, 8, 10],
    "model__num_leaves":        randint(15, 100),
    "model__min_child_samples": randint(5, 40),
    "model__subsample":         uniform(0.6, 0.4),
    "model__colsample_bytree":  uniform(0.4, 0.6),
    "model__reg_alpha":         uniform(0, 2),
    "model__reg_lambda":        uniform(0, 2),
    "model__scale_pos_weight":  [1, 2, 3, 4],
}

base_lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
pipe_lgbm = make_lgbm_pipe(base_lgbm, len(features))

lgbm_search = RandomizedSearchCV(
    pipe_lgbm,
    param_distributions=lgbm_param_dist,
    n_iter=60,
    cv=cv5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True,
)
lgbm_search.fit(X, y_bin, groups=groups)

print(f"\n  Best AUC (CV): {lgbm_search.best_score_:.3f}")
print("  Best params:")
for k, v in lgbm_search.best_params_.items():
    print(f"    {k.replace('model__',''):<22} = {v}")


# =========================================================
# 4. FIT BEST RF (from tune_rf.py params) TO GET IMPORTANCES
# =========================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE PRUNING  (using RF importances)")
print("=" * 60)

best_rf_params = pd.read_csv("data/processed_data/rf_best_params.csv").iloc[0].to_dict()

tuned_rf = RandomForestClassifier(
    n_estimators=int(best_rf_params["n_estimators"]),
    max_depth=None if str(best_rf_params["max_depth"]) == "nan" else int(best_rf_params["max_depth"]),
    min_samples_leaf=int(best_rf_params["min_samples_leaf"]),
    max_features=float(best_rf_params["max_features"]) if str(best_rf_params["max_features"]).replace('.','').isdigit() else best_rf_params["max_features"],
    max_samples=float(best_rf_params["max_samples"]),
    class_weight={0: 1, 1: 4},
    random_state=42,
    n_jobs=-1,
)

imp_pipe = make_rf_pipe(tuned_rf, len(features))
imp_pipe.fit(X, y_bin)

importances = imp_pipe.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 features by RF importance:")
print(importance_df.head(20).to_string(index=False))
importance_df.to_csv("data/processed_data/rf_feature_importances.csv", index=False)


# =========================================================
# 5. COMPARE RF + LGBM ACROSS FEATURE SUBSET SIZES
# =========================================================

print("\n" + "=" * 60)
print("PRUNING COMPARISON  (RF vs LGBM at different feature counts)")
print("=" * 60)
print(f"  {'Model':<35} {'N feats':>7}  {'AUC':>7}  {'±':>5}  {'F1':>6}")
print("  " + "-" * 60)

results = []
subset_sizes = [20, 30, 40, 120]

for n in subset_sizes:
    top_feats = importance_df["feature"].head(n).tolist()
    Xn = feature_df[top_feats].values.astype(float)

    # RF
    rf_n = RandomForestClassifier(
        n_estimators=int(best_rf_params["n_estimators"]),
        max_depth=None if str(best_rf_params["max_depth"]) == "nan" else int(best_rf_params["max_depth"]),
        min_samples_leaf=int(best_rf_params["min_samples_leaf"]),
        max_features=float(best_rf_params["max_features"]) if str(best_rf_params["max_features"]).replace('.','').isdigit() else best_rf_params["max_features"],
        max_samples=float(best_rf_params["max_samples"]),
        class_weight={0: 1, 1: 4},
        random_state=42, n_jobs=-1,
    )
    rf_pipe_n = make_rf_pipe(rf_n, n)
    auc, std, f1 = cv_auc_f1(rf_pipe_n, Xn, y_bin, groups)
    print(f"  {'RF (tuned)':<35} {n:>7}  {auc:.3f}  {std:.3f}  {f1:.3f}")
    results.append({"model": "RF (tuned)", "n_features": n, "auc": auc, "std": std, "f1": f1})

    # LGBM with best params
    best_lgbm_params = {
        k.replace("model__", ""): v
        for k, v in lgbm_search.best_params_.items()
    }
    lgbm_n = lgb.LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_pipe_n = make_lgbm_pipe(lgbm_n, n)
    auc, std, f1 = cv_auc_f1(lgbm_pipe_n, Xn, y_bin, groups)
    print(f"  {'LGBM (tuned)':<35} {n:>7}  {auc:.3f}  {std:.3f}  {f1:.3f}")
    results.append({"model": "LGBM (tuned)", "n_features": n, "auc": auc, "std": std, "f1": f1})

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df["auc"].idxmax()]
print(f"\n  Best overall: {best_row['model']}  with {int(best_row['n_features'])} features")
print(f"  AUC = {best_row['auc']:.3f} ± {best_row['std']:.3f}  |  F1 = {best_row['f1']:.3f}")

results_df.to_csv("data/processed_data/lgbm_pruning_comparison.csv", index=False)
print("\nSaved lgbm_pruning_comparison.csv and rf_feature_importances.csv")
