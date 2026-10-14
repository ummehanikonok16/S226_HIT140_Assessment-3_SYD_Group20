"""
HIT140 - Objective 2, Linear Regression 2.2
Predicting goals scored by a team in a single FIFA World Cup 2026 match.


Run:  python lr_2_2_analysis.py
Requires: pandas, numpy, matplotlib, scipy, statsmodels, scikit-learn

"""

import os
import sys
import glob
import subprocess


_needed = {"pandas": "pandas", "numpy": "numpy", "matplotlib": "matplotlib",
           "scipy": "scipy", "statsmodels": "statsmodels", "sklearn": "scikit-learn"}
for _mod, _pkg in _needed.items():
    try:
        __import__(_mod)
    except ImportError:
        print("Installing missing library:", _pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------
# 0. Settings  (everything is relative to the folder this script is saved in)
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Finds the CSV automatically, even if it is named ...csv.csv or ...(1).csv
_matches = sorted(glob.glob(os.path.join(HERE, "*208_team_match_dataset*.csv*")))
if not _matches:
    sys.exit("ERROR: Could not find the dataset CSV in: " + HERE +
             "\nPut HIT140_Regression_2_2_208_team_match_dataset.csv in the same folder as this script.")
CSV_PATH = _matches[0]
print("Using data file:", CSV_PATH)

FIG_DIR = os.path.join(HERE, "figures_lr22")
SEED = 42
TARGET = "goals_scored"
FEATURES = [
    "rank_diff",
    "log_value_diff",
    "prior_ppg_diff",
    "rest_days_diff",
    "team_recent_goals_avg",
    "opponent_recent_goals_conceded_avg",
    "team_recent_goals_conceded_avg",
    "opponent_recent_goals_avg",
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for st in self.streams:
            st.write(text)
    def flush(self):
        for st in self.streams:
            st.flush()

_log = open(os.path.join(HERE, "results_lr22.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log)

os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name))
    plt.close()



# 1. Load and inspect data

df = pd.read_csv(CSV_PATH)
print("=" * 70, "\n1. DATA OVERVIEW\n" + "=" * 70)
print("Shape:", df.shape)
print("Unique matches:", df["match_id"].nunique())
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nRows per stage:\n", df["stage"].value_counts())
print("\nSummary statistics:\n", df[[TARGET] + FEATURES].describe().T.round(3))

# Sanity check: each match should have exactly 2 rows
assert (df.groupby("match_id").size() == 2).all(), "Each match must have 2 rows"

X = df[FEATURES]
y = df[TARGET]


# 2. Exploratory data analysis

print("\n" + "=" * 70, "\n2. EDA\n" + "=" * 70)
print("Goals distribution:\n", y.value_counts().sort_index())
print("\nMean goals by stage:\n", df.groupby("stage")[TARGET].mean().round(3))

corr = df[[TARGET] + FEATURES].corr()
print("\nCorrelation with target:\n", corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

# 2a. Target distribution
plt.figure(figsize=(6, 4))
counts = y.value_counts().sort_index()
plt.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
plt.xlabel("Goals scored by team in match")
plt.ylabel("Number of team-matches")
plt.title("Distribution of goals scored")
save("01_goals_distribution.png")

# 2b. Correlation heatmap
plt.figure(figsize=(9, 7))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson r")
plt.xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
plt.yticks(range(len(corr)), corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation matrix")
plt.grid(False)
save("02_correlation_heatmap.png")

# 2c. Scatter plots: every predictor vs goals
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), FEATURES):
    jitter = np.random.RandomState(SEED).uniform(-0.1, 0.1, len(df))
    ax.scatter(df[col], y + jitter, alpha=0.5, s=18)
    slope, intercept = np.polyfit(df[col], y, 1)
    xs = np.linspace(df[col].min(), df[col].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="red")
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("goals_scored")
    ax.set_title(f"r = {corr.loc[col, TARGET]:.2f}", fontsize=9)
save("03_scatter_predictors.png")

# 2d. Goals by stage
stage_order = ["Group stage", "Round of 32", "Round of 16", "Quarter-finals",
               "Semi-finals", "Third-place match", "Final"]
plt.figure(figsize=(9, 4.5))
data = [df.loc[df["stage"] == s, TARGET] for s in stage_order]
plt.boxplot(data)
plt.xticks(range(1, len(stage_order) + 1), stage_order)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Goals scored")
plt.title("Goals scored by tournament stage")
save("04_goals_by_stage.png")

# 2e. Multicollinearity (VIF)
Xc = sm.add_constant(X)
vif = pd.DataFrame({
    "variable": FEATURES,
    "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(FEATURES))],
}).round(2)
print("\nVariance Inflation Factors:\n", vif.to_string(index=False))


# 3. OLS model on full data (for inference / interpretation)

print("\n" + "=" * 70, "\n3. OLS MODEL (FULL DATA)\n" + "=" * 70)
ols = sm.OLS(y, Xc).fit()
print(ols.summary())

# Robust (HC3) standard errors as a sensitivity check
ols_robust = sm.OLS(y, Xc).fit(cov_type="HC3")
print("\nCoefficients with HC3 robust standard errors:")
print(pd.DataFrame({"coef": ols_robust.params, "robust_se": ols_robust.bse,
                    "p_value": ols_robust.pvalues}).round(4))

# Standardised coefficients (to compare predictor importance)
Xz = (X - X.mean()) / X.std()
std_model = sm.OLS((y - y.mean()) / y.std(), sm.add_constant(Xz)).fit()
print("\nStandardised coefficients:\n", std_model.params.drop("const").round(3)
      .sort_values(key=abs, ascending=False))


# 4. Assumption diagnostics

print("\n" + "=" * 70, "\n4. DIAGNOSTICS\n" + "=" * 70)
resid = ols.resid
fitted = ols.fittedvalues

bp_lm, bp_p, _, _ = het_breuschpagan(resid, Xc)
sh_stat, sh_p = stats.shapiro(resid)
jb_stat, jb_p = stats.jarque_bera(resid)
print(f"Breusch-Pagan: LM = {bp_lm:.2f}, p = {bp_p:.4f}")
print(f"Shapiro-Wilk:  W = {sh_stat:.4f}, p = {sh_p:.4f}")
print(f"Jarque-Bera:   JB = {jb_stat:.2f}, p = {jb_p:.4f}")
print(f"Durbin-Watson: {sm.stats.durbin_watson(resid):.3f}")
print(f"Residual skew = {stats.skew(resid):.3f}, kurtosis = {stats.kurtosis(resid, fisher=False):.3f}")

cooks = ols.get_influence().cooks_distance[0]
print(f"Cook's D: max = {cooks.max():.3f}, points > 4/n: {(cooks > 4 / len(df)).sum()}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].scatter(fitted, resid, alpha=0.6)
axes[0, 0].axhline(0, color="red")
axes[0, 0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
sm.qqplot(resid, line="s", ax=axes[0, 1])
axes[0, 1].set_title("Normal Q-Q plot")
axes[1, 0].hist(resid, bins=20, color="steelblue", edgecolor="black")
axes[1, 0].set(xlabel="Residual", ylabel="Frequency", title="Residual histogram")
axes[1, 1].stem(cooks, markerfmt=",")
axes[1, 1].axhline(4 / len(df), color="red", ls="--", label="4/n")
axes[1, 1].set(xlabel="Observation", ylabel="Cook's distance", title="Influence")
axes[1, 1].legend()
save("05_diagnostics.png")

# Actual vs predicted
plt.figure(figsize=(5.5, 5))
plt.scatter(fitted, y, alpha=0.5)
lim = [0, max(y.max(), fitted.max()) + 0.5]
plt.plot(lim, lim, "r--")
plt.xlabel("Predicted goals")
plt.ylabel("Actual goals")
plt.title("Actual vs predicted (full data)")
save("06_actual_vs_predicted.png")

# Coefficient plot with 95% CI
ci = ols.conf_int().drop("const")
coefs = ols.params.drop("const")
plt.figure(figsize=(8, 4.5))
plt.errorbar(coefs, range(len(coefs)), xerr=[coefs - ci[0], ci[1] - coefs], fmt="o", capsize=4)
plt.axvline(0, color="red", ls="--")
plt.yticks(range(len(coefs)), coefs.index, fontsize=8)
plt.xlabel("Coefficient (95% CI)")
plt.title("Regression coefficients")
save("07_coefficients.png")


# 5. Predictive evaluation (match-level split to avoid leakage)

print("\n" + "=" * 70, "\n5. PREDICTIVE EVALUATION\n" + "=" * 70)
# Both rows of a match are mirrored, so split by match_id, not by row.
rng = np.random.RandomState(SEED)
match_ids = df["match_id"].unique()
test_ids = rng.choice(match_ids, size=int(0.2 * len(match_ids)), replace=False)
test_mask = df["match_id"].isin(test_ids)

X_tr, X_te = X[~test_mask], X[test_mask]
y_tr, y_te = y[~test_mask], y[test_mask]
print(f"Train rows: {len(X_tr)}, test rows: {len(X_te)} "
      f"({len(test_ids)} test matches)")

lr = LinearRegression().fit(X_tr, y_tr)
pred_te = lr.predict(X_te)
pred_tr = lr.predict(X_tr)
baseline = np.full(len(y_te), y_tr.mean())


def metrics(true, pred):
    return (r2_score(true, pred),
            np.sqrt(mean_squared_error(true, pred)),
            mean_absolute_error(true, pred))


res = pd.DataFrame(
    {"R2": [metrics(y_tr, pred_tr)[0], *metrics(y_te, pred_te)[:1], metrics(y_te, baseline)[0]],
     "RMSE": [metrics(y_tr, pred_tr)[1], metrics(y_te, pred_te)[1], metrics(y_te, baseline)[1]],
     "MAE": [metrics(y_tr, pred_tr)[2], metrics(y_te, pred_te)[2], metrics(y_te, baseline)[2]]},
    index=["Model - train", "Model - test", "Baseline (train mean) - test"],
).round(3)
print(res)

# Grouped 5-fold cross-validation
cv_r2 = cross_val_score(LinearRegression(), X, y, groups=df["match_id"],
                        cv=GroupKFold(n_splits=5), scoring="r2")
cv_rmse = -cross_val_score(LinearRegression(), X, y, groups=df["match_id"],
                           cv=GroupKFold(n_splits=5), scoring="neg_root_mean_squared_error")
print("\nGrouped 5-fold CV R2 per fold:", np.round(cv_r2, 3), "| mean =", round(cv_r2.mean(), 3))
print("Grouped 5-fold CV RMSE per fold:", np.round(cv_rmse, 3), "| mean =", round(cv_rmse.mean(), 3))


# 6. Reduced model (significant predictors only)

print("\n" + "=" * 70, "\n6. REDUCED MODEL\n" + "=" * 70)
reduced_feats = ["rank_diff", "log_value_diff"]
red = sm.OLS(y, sm.add_constant(df[reduced_feats])).fit()
print(red.summary().tables[0])
print(red.summary().tables[1])
print(f"\nFull model:    adj R2 = {ols.rsquared_adj:.3f}, AIC = {ols.aic:.1f}, BIC = {ols.bic:.1f}")
print(f"Reduced model: adj R2 = {red.rsquared_adj:.3f}, AIC = {red.aic:.1f}, BIC = {red.bic:.1f}")
f_stat, f_p, _ = ols.compare_f_test(red)
print(f"Partial F-test (do the 6 dropped variables add anything?): F = {f_stat:.3f}, p = {f_p:.4f}")

cv_red = cross_val_score(LinearRegression(), df[reduced_feats], y, groups=df["match_id"],
                         cv=GroupKFold(n_splits=5), scoring="r2")
print("Reduced model grouped CV R2: mean =", round(cv_red.mean(), 3))


# 7. Optional extension: Poisson regression (goals are counts)

print("\n" + "=" * 70, "\n7. POISSON REGRESSION (EXTENSION)\n" + "=" * 70)
pois = sm.GLM(y, Xc, family=sm.families.Poisson()).fit()
print(pois.summary().tables[1])
print(f"Poisson pseudo-R2 (deviance-based): {1 - pois.deviance / pois.null_deviance:.3f}")

print(f"\nDone. Figures saved in ./{FIG_DIR}/")


# 8. Sensitivity analyses

print("\n" + "=" * 70, "\n8. SENSITIVITY ANALYSES\n" + "=" * 70)

# 8a. Cluster-robust standard errors (rows of the same match are not independent)
ols_cl = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": df["match_id"]})
print("Coefficients with match-clustered standard errors:")
print(pd.DataFrame({"coef": ols_cl.params, "cluster_se": ols_cl.bse,
                    "p_value": ols_cl.pvalues}).round(4))

# 8b. Exclude rows whose recent-form variables are zero-filled (no prior match data)
has_form = (df["team_recent_goals_avg"] > 0) | (df["opponent_recent_goals_avg"] > 0)
sub = df[has_form]
ols_sub = sm.OLS(sub[TARGET], sm.add_constant(sub[FEATURES])).fit()
print(f"\nExcluding zero-filled form rows: n = {len(sub)}, R2 = {ols_sub.rsquared:.3f}, "
      f"adj R2 = {ols_sub.rsquared_adj:.3f}")
print(pd.DataFrame({"coef": ols_sub.params, "p_value": ols_sub.pvalues}).round(4))

# 8c. Add a knockout-stage indicator
df["knockout"] = (df["stage"] != "Group stage").astype(int)
ols_ko = sm.OLS(y, sm.add_constant(df[FEATURES + ["knockout"]])).fit()
print(f"\nWith knockout dummy: coef = {ols_ko.params['knockout']:.3f}, "
      f"p = {ols_ko.pvalues['knockout']:.3f}, adj R2 = {ols_ko.rsquared_adj:.3f}")


# 9. Extra numbers quoted in the report + compact 3-panel figure

print("\n" + "=" * 70, "\n9. EXTRA REPORT NUMBERS\n" + "=" * 70)
print("Match dates:", df["date"].min(), "to", df["date"].max())
print("Matches per stage:\n", df.groupby("stage")["match_id"].nunique())
print("Rows per stage and mean goals:\n", df.groupby("stage")[TARGET].agg(["count", "mean"]).round(2))
print(f"Share of rows with 0 or 1 goals: {(y <= 1).mean():.1%}")
print(f"Share of rows with 4+ goals: {(y >= 4).mean():.1%} ({(y >= 4).sum()} rows)")
print(f"Goals: mean = {y.mean():.2f}, variance = {y.var():.2f}, sd = {y.std():.2f}")
zero_team = (df["team_recent_goals_avg"] == 0)
zero_both = zero_team & (df["opponent_recent_goals_avg"] == 0)
print(f"Rows with zero-filled team form: {zero_team.sum()} ({zero_team.mean():.1%}); both teams zero: {zero_both.sum()}")
print("\nPoisson exp(coef) (multiplier on expected goals per unit):")
print(np.exp(pois.params).round(3))
print(f"50-place ranking advantage multiplier (Poisson): {np.exp(50 * pois.params['rank_diff']):.2f}")
print(f"50-place ranking advantage in goals (OLS): {50 * ols.params['rank_diff']:.2f}")
print(f"Exp(1) = {np.e:.2f} (one log-unit = squad worth about this many times the opponent's)")

# Compact 3-panel figure used in the short report
short_labels = ["rank_diff", "log_value_diff", "prior_ppg_diff", "rest_days_diff",
                "team_goals_avg", "opp_conceded_avg", "team_conceded_avg", "opp_goals_avg"]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
c = y.value_counts().sort_index()
ax[0].bar(c.index, c.values, color="steelblue", edgecolor="k")
ax[0].set(xlabel="Goals scored", ylabel="Team-matches", title="(a) Distribution of goals")
ci = ols.conf_int().drop("const")
co = ols.params.drop("const")
ax[1].errorbar(co, range(8), xerr=[co - ci[0], ci[1] - co], fmt="o", capsize=3)
ax[1].axvline(0, color="r", ls="--")
ax[1].set_yticks(range(8))
ax[1].set_yticklabels(short_labels, fontsize=7)
ax[1].invert_yaxis()
ax[1].set(xlabel="Coefficient (95% CI)", title="(b) Coefficients")
ax[2].scatter(ols.fittedvalues, ols.resid, alpha=0.5, s=14)
ax[2].axhline(0, color="r")
ax[2].set(xlabel="Fitted goals", ylabel="Residual", title="(c) Residuals vs fitted")
save("08_combined_three_panel.png")
print("Saved combined figure.")

print("\nAll results were also saved to results_lr22.txt")
sys.stdout = sys.__stdout__
_log.close()