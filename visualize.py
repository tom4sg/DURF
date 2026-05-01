"""
Presentation visualizations for the DURF project.

Produces five plots:
  1. Feature importance bar chart (top 20 by RF importance)
  2. ROC curve for the best model (RF tuned, top-20 features, >7w threshold)
  3. AUC progression chart (untuned RF → tuned RF → pruned RF 20 feats)
  4. Target distribution (lifespan histogram + >7w threshold annotation)
  5. Confusion matrix for best model (OOF predictions)

Saves to graphs/presentation/
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

os.makedirs("graphs/presentation", exist_ok=True)

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#16A34A",
    "accent":    "#DC2626",
    "neutral":   "#6B7280",
    "bg":        "#F8FAFC",
}

# =========================================================
# LOAD DATA
# =========================================================

feature_df = pd.read_csv("data/processed_data/feature_df_large_ts_features.csv")
importance_df = pd.read_csv("data/processed_data/rf_feature_importances.csv")
best_rf_params = pd.read_csv("data/processed_data/rf_best_params.csv").iloc[0].to_dict()

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
all_features = [col for col, _ in screen[:120]]
top20_features = importance_df["feature"].head(20).tolist()

X_top20 = feature_df[top20_features].values.astype(float)

cv5 = GroupKFold(n_splits=5)

def make_rf(params, n_feats):
    rf = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if str(params["max_depth"]) == "nan" else int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=(
            float(params["max_features"])
            if str(params["max_features"]).replace(".", "").isdigit()
            else params["max_features"]
        ),
        max_samples=float(params["max_samples"]),
        class_weight={0: 1, 1: 4},
        random_state=42,
        n_jobs=-1,
    )
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), list(range(n_feats)))])
    return Pipeline([("prep", prep), ("model", rf)])


# =========================================================
# PLOT 1: FEATURE IMPORTANCE BAR CHART
# =========================================================

print("Building Plot 1: Feature importance…")

top20 = importance_df.head(20).copy()

def clean_name(name):
    name = name.replace("_post2w_", " post2w ")
    name = name.replace("_365d_", " 365d ")
    name = name.replace("_180d_", " 180d ")
    name = name.replace("_90d_", " 90d ")
    name = name.replace("_30d_", " 30d ")
    name = name.replace("_release_date", " @release")
    name = name.replace("_", " ")
    name = name.strip()
    if len(name) > 45:
        name = name[:43] + "…"
    return name

labels = [clean_name(f) for f in top20["feature"]]
vals = top20["importance"].values
normalized = vals / vals.sum()

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

colors = [PALETTE["primary"] if i < 5 else PALETTE["neutral"] for i in range(len(labels))]
bars = ax.barh(range(len(labels)), normalized[::-1], color=colors[::-1],
               edgecolor="white", linewidth=0.5)

# value labels on bars
for bar, v in zip(bars, normalized[::-1]):
    ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{v:.2%}", va="center", ha="left", fontsize=8, color="#444")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels[::-1], fontsize=9)
ax.set_xlabel("Relative importance (normalized)", fontsize=11)
ax.set_title("Top 20 Features — RF Importance\n(Binary target: charted >7 weeks)",
             fontsize=13, fontweight="bold", pad=14)

ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
ax.set_axisbelow(True)
ax.set_xlim(0, normalized.max() * 1.18)

top5_patch = mpatches.Patch(color=PALETTE["primary"], label="Top 5 features")
rest_patch  = mpatches.Patch(color=PALETTE["neutral"], label="Features 6–20")
ax.legend(handles=[top5_patch, rest_patch], fontsize=9, loc="lower right", framealpha=0.8)

plt.tight_layout()
plt.subplots_adjust(left=0.38)
plt.savefig("graphs/presentation/01_feature_importance.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved 01_feature_importance.png")


# =========================================================
# PLOT 2: ROC CURVE (best model, full OOF)
# =========================================================

print("Building Plot 2: ROC curve…")

pipe_best = make_rf(best_rf_params, len(top20_features))

all_proba, all_true = [], []
for tr, te in cv5.split(X_top20, groups=groups):
    pipe_best.fit(X_top20[tr], y_bin[tr])
    proba = pipe_best.predict_proba(X_top20[te])[:, 1]
    all_proba.extend(proba)
    all_true.extend(y_bin[te])

all_proba = np.array(all_proba)
all_true  = np.array(all_true)

fpr, tpr, _ = roc_curve(all_true, all_proba)
auc = roc_auc_score(all_true, all_proba)

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5)
ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.2, linestyle="--")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])

# AUC text box in upper left — no overlap with curve
ax.text(0.05, 0.95, f"AUC = {auc:.3f}", transform=ax.transAxes,
        fontsize=13, fontweight="bold", color=PALETTE["primary"],
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["bg"], edgecolor=PALETTE["primary"], alpha=0.9))

ax.text(0.60, 0.08, "Random chance (AUC = 0.500)", transform=ax.transAxes,
        fontsize=9, color=PALETTE["neutral"], va="bottom", ha="left")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Best Model\n(Binary: charted >7 weeks  |  GroupKFold OOF)",
             fontsize=13, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.xaxis.grid(True, linestyle="--", alpha=0.3)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("graphs/presentation/02_roc_curve.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved 02_roc_curve.png")


# =========================================================
# PLOT 3: AUC PROGRESSION CHART
# =========================================================

print("Building Plot 3: AUC progression…")

# Values from prior runs (tune_rf.py + tune_lgbm.py outputs)
FIXED_AUC   = 0.700   # fixed RF, >7w threshold, threshold sweep
TUNED_AUC   = 0.712   # tuned RF, full 120 features
PRUNED20_AUC = 0.737  # tuned RF, top-20 features  ← best
LGBM_AUC    = 0.711   # tuned LGBM, 120 features

labels_prog = [
    "Fixed RF\n(120 feats)",
    "Tuned RF\n(120 feats)",
    "Tuned LGBM\n(120 feats)",
    "Tuned RF\n(top-20 feats)\n★ Best",
]
aucs_prog = [FIXED_AUC, TUNED_AUC, LGBM_AUC, PRUNED20_AUC]
colors_prog = [
    PALETTE["neutral"],
    PALETTE["secondary"],
    PALETTE["primary"],
    PALETTE["accent"],
]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

x = np.arange(len(labels_prog))
bars = ax.bar(x, aucs_prog, color=colors_prog, width=0.55,
              edgecolor="white", linewidth=0.8, zorder=3)

# value labels
for bar, v in zip(bars, aucs_prog):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# chance baseline
ax.axhline(0.5, color=PALETTE["neutral"], linestyle="--", lw=1.2, zorder=2,
           label="Random chance (0.500)")

ax.set_xticks(x)
ax.set_xticklabels(labels_prog, fontsize=9)
ax.set_ylabel("CV AUC-ROC  (GroupKFold, k=5)", fontsize=10)
ax.set_ylim(0.45, 0.80)
ax.set_title("AUC Progression: Tuning & Feature Pruning\n(Binary target: charted >7 weeks on Billboard Hot 100)",
             fontsize=12, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=1)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper left", framealpha=0.8)

plt.tight_layout()
plt.savefig("graphs/presentation/03_auc_progression.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved 03_auc_progression.png")


# =========================================================
# PLOT 4: TARGET DISTRIBUTION
# =========================================================

print("Building Plot 4: Target distribution…")

THRESHOLD = 7
majority_acc = (y_bin == 0).mean()

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

bins = np.arange(1, min(y_raw.max() + 2, 60), 1)
counts_short, _, _ = ax.hist(y_raw[y_raw <= THRESHOLD], bins=bins,
                              color=PALETTE["neutral"], alpha=0.85,
                              label=f"≤{THRESHOLD} weeks  ({(y_bin==0).mean():.0%})")
counts_long, _, _  = ax.hist(y_raw[y_raw > THRESHOLD], bins=bins,
                              color=PALETTE["primary"], alpha=0.85,
                              label=f">{THRESHOLD} weeks  ({(y_bin==1).mean():.0%})")

ax.axvline(THRESHOLD + 0.5, color=PALETTE["accent"], lw=2, linestyle="--", zorder=5)
ax.text(THRESHOLD + 1.2, ax.get_ylim()[1] * 0.88,
        f"Threshold\n>{THRESHOLD} weeks", color=PALETTE["accent"],
        fontsize=9, va="top")

# annotate median and mean
med = np.median(y_raw)
mn  = np.mean(y_raw)
ax.axvline(med, color="black", lw=1.2, linestyle=":", alpha=0.7)
ax.axvline(mn,  color="black", lw=1.2, linestyle="-.", alpha=0.7)
ax.text(med - 0.4, ax.get_ylim()[1] * 0.60, f"median\n{med:.0f}w",
        ha="right", fontsize=8, color="black", alpha=0.8)
ax.text(mn  + 0.4, ax.get_ylim()[1] * 0.45, f"mean\n{mn:.1f}w",
        ha="left",  fontsize=8, color="black", alpha=0.8)

ax.set_xlabel("Chart lifespan (weeks)", fontsize=11)
ax.set_ylabel("Number of songs", fontsize=11)
ax.set_title("Distribution of Billboard Hot 100 Chart Lifespan\n(songs capped at 59w for readability)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=10, loc="upper right", framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.set_xlim(0, 60)

note = f"n={len(y_raw)} songs  |  Majority-class baseline accuracy = {majority_acc:.1%}"
fig.text(0.5, -0.02, note, ha="center", fontsize=9, color=PALETTE["neutral"])

plt.tight_layout()
plt.savefig("graphs/presentation/04_target_distribution.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved 04_target_distribution.png")


# =========================================================
# PLOT 5: CONFUSION MATRIX (OOF predictions from best model)
# =========================================================

print("Building Plot 5: Confusion matrix…")

# collect OOF hard predictions alongside probabilities already computed
all_pred = (all_proba >= 0.5).astype(int)

cm = confusion_matrix(all_true, all_pred)
tn, fp, fn, tp = cm.ravel()
total = len(all_true)

# row-normalised (rates within each true class)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

class_labels = [f"≤{THRESHOLD}w\n(short)", f">{THRESHOLD}w\n(hit)"]
ax.set_xticks([0, 1]); ax.set_xticklabels(class_labels, fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(class_labels, fontsize=11)
ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
ax.set_ylabel("Actual", fontsize=12, labelpad=8)
ax.set_title(f"Confusion Matrix — Best Model (OOF)\nRF tuned, top-20 features, >{THRESHOLD}w threshold",
             fontsize=12, fontweight="bold", pad=14)

for i in range(2):
    for j in range(2):
        count = cm[i, j]
        rate  = cm_norm[i, j]
        color = "white" if rate > 0.55 else "black"
        ax.text(j, i, f"{count}\n({rate:.0%})",
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label="Rate within true class")

f1  = f1_score(all_true, all_pred)
acc = (all_pred == all_true).mean()
fig.text(0.5, -0.04,
         f"Accuracy={acc:.1%}  |  F1={f1:.3f}  |  AUC={auc:.3f}  "
         f"|  Majority-class baseline accuracy={majority_acc:.1%}",
         ha="center", fontsize=9, color=PALETTE["neutral"])

plt.tight_layout()
plt.savefig("graphs/presentation/05_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved 05_confusion_matrix.png")


# =========================================================
# PLOT 6: SHAP SUMMARY
# =========================================================

print("Building Plot 6: SHAP summary…")

try:
    import shap

    # fit on full data (same as importance extraction)
    pipe_shap = make_rf(best_rf_params, len(top20_features))
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), list(range(len(top20_features))))])
    prep.fit(X_top20)
    X_top20_imputed = prep.transform(X_top20)

    pipe_shap.fit(X_top20, y_bin)
    rf_model = pipe_shap.named_steps["model"]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_top20_imputed)

    # handle both old API (list of arrays) and new API (3D array)
    if isinstance(shap_values, list):
        sv = shap_values[1]          # old: list[class0, class1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]    # new: (n_samples, n_features, n_classes)
    else:
        sv = shap_values

    # print SHAP summary stats per feature
    shap_df = pd.DataFrame({
        "feature":        top20_features,
        "mean_abs_shap":  np.abs(sv).mean(axis=0),
        "mean_shap":      sv.mean(axis=0),
        "median_shap":    np.median(sv, axis=0),
        "max_shap":       sv.max(axis=0),
        "min_shap":       sv.min(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    print("\nSHAP summary (class 1 = hit >7w):")
    print(shap_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    shap_df.to_csv("data/processed_data/shap_summary.csv", index=False)

    # shap needs flat single-line names — no \n
    def shap_name(name):
        name = name.replace("_release_date", "@release")
        name = name.replace("_post2w_", " post2w ")
        name = name.replace("_365d_", " 365d ")
        name = name.replace("_180d_", " 180d ")
        name = name.replace("_90d_", " 90d ")
        name = name.replace("_30d_", " 30d ")
        name = name.replace("_full_", " full ")
        name = name.replace("_", " ")
        # truncate long names
        if len(name) > 42:
            name = name[:40] + "…"
        return name

    display_names = [shap_name(f) for f in top20_features]

    # shap creates its own figure — set size via plot_size param
    shap.summary_plot(
        sv, X_top20_imputed,
        feature_names=display_names,
        show=False,
        plot_size=(12, 8),
        color_bar=True,
    )

    fig = plt.gcf()
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.title("SHAP Feature Impact — Best Model\n(RF tuned, top-20 features, >7w threshold  |  red=high value, blue=low value)",
              fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig("graphs/presentation/06_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  Saved 06_shap_summary.png")

except ImportError:
    print("  shap not installed — run: pip install shap")


# =========================================================
# TABLE: DATA COVERAGE
# =========================================================

print("Building data coverage table…")

ig_raw = pd.read_csv("data/processed_data/12m_ig_pre_release.csv")
tt_raw = pd.read_csv("data/processed_data/12m_tt_pre_release.csv")
yt_raw = pd.read_csv("data/processed_data/12m_yt_pre_release.csv")

all_songs = set(feature_df["song_id"].unique())
n_total   = len(all_songs)

ig_songs = set(ig_raw["song_id"].unique())
tt_songs = set(tt_raw["song_id"].unique())
yt_songs = set(yt_raw["song_id"].unique())

all_three   = ig_songs & tt_songs & yt_songs & all_songs
ig_tt_only  = (ig_songs & tt_songs) - yt_songs
ig_yt_only  = (ig_songs & yt_songs) - tt_songs
tt_yt_only  = (tt_songs & yt_songs) - ig_songs
ig_only     = ig_songs - tt_songs - yt_songs
tt_only     = tt_songs - ig_songs - yt_songs
yt_only     = yt_songs - ig_songs - tt_songs
none_       = all_songs - ig_songs - tt_songs - yt_songs

rows = [
    ("Instagram",           len(ig_songs & all_songs), len(ig_songs & all_songs) / n_total),
    ("TikTok",              len(tt_songs & all_songs), len(tt_songs & all_songs) / n_total),
    ("YouTube",             len(yt_songs & all_songs), len(yt_songs & all_songs) / n_total),
    ("All three platforms", len(all_three),             len(all_three) / n_total),
    ("No social data",      len(none_),                 len(none_) / n_total),
]

cov_df = pd.DataFrame(rows, columns=["Platform", "Songs", "Coverage"])
cov_df["Coverage"] = cov_df["Coverage"].map("{:.1%}".format)
cov_df.to_csv("data/processed_data/data_coverage.csv", index=False)

print(f"\n  Data coverage  (n={n_total} songs in model)")
print(cov_df.to_string(index=False))

# also make a simple visual table
fig, ax = plt.subplots(figsize=(7, 3))
fig.patch.set_facecolor(PALETTE["bg"])
ax.axis("off")

table = ax.table(
    cellText=cov_df.values,
    colLabels=cov_df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)

# style header row
for j in range(len(cov_df.columns)):
    table[0, j].set_facecolor(PALETTE["primary"])
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title(f"Social Data Coverage  (n={n_total} songs)",
             fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("graphs/presentation/07_data_coverage.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved 07_data_coverage.png")

print("\nAll plots saved to graphs/presentation/")
