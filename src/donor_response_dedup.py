# -*- coding: utf-8 -*-
"""
Donor response modelling on PVA data - corrected evaluation
============================================================
The supplied PVA_PARTITION file (106,546 rows) contains only 9,686 distinct
donor profiles: each was replicated ~11 times, and 99.9% of Validation rows
also appear in Training. Models scored on that split were memorising people,
not generalising. This script:
  1. removes the duplicates (one row per distinct profile)
  2. evaluates 4 models with 5-fold stratified cross-validation, with all
     imputation and scaling fitted inside each fold
  3. fits the champion on a 70/30 split for the ROC, lift and SHAP figures

Run in Colab with pva_partition.csv uploaded. Takes ~3-5 minutes.
Copy ALL printed output back, and download the PNG/CSV files it saves.
"""
# !pip install xgboost shap -q
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 42

# ------------------------------------------------------------
# 1. Load, clean, remove duplicate profiles
# ------------------------------------------------------------
df = pd.read_csv("pva_partition.csv")

def clean_currency(col):
    return pd.to_numeric(col.astype(str).str.replace('[$,]', '', regex=True), errors='coerce')

for col in ['TARGET_D_with0', 'TARGET_D', 'GiftAvgLast', 'GiftAvg36',
            'GiftAvgAll', 'GiftAvgCard36', 'DemMedHomeValue', 'DemMedIncome']:
    df[col] = clean_currency(df[col])
for col in ['StatusCat96NK', 'DemGender', 'DemHomeOwner']:
    df[col] = df[col].fillna('Unknown').astype('category').cat.codes
df['DemCluster'] = df['DemCluster'].fillna(-1)

drop_cols = ['ID', 'TARGET_D', 'TARGET_D_with0', 'Partition', 'TARGET_B']
raw_features = [c for c in df.columns if c not in drop_cols]

n_before = len(df)
df = df.drop_duplicates(subset=raw_features + ['TARGET_B']).reset_index(drop=True)
print(f"Rows before de-duplication: {n_before:,} | after: {len(df):,}")
print(f"Response rate after de-duplication: {df['TARGET_B'].mean():.1%}")

# ------------------------------------------------------------
# 2. Engineered RFM features (all use pre-campaign history only)
# ------------------------------------------------------------
df['CardPromRatio']    = df['PromCntCardAll'] / (df['PromCntAll'] + 1)
df['CardResponseRate'] = df['GiftCntCardAll'] / (df['PromCntCardAll'] + 1)
df['FreqRecencyScore'] = df['GiftCnt36'] / (df['GiftTimeLast'] + 1)
df['GiftTrend']        = df['GiftAvg36'] / (df['GiftAvgAll'] + 1)
df['DonationRate']     = df['GiftCntAll'] / (df['PromCntAll'] + 1)
df['DonorLifespan']    = df['GiftTimeFirst'] - df['GiftTimeLast']

features = raw_features + ['CardPromRatio', 'CardResponseRate', 'FreqRecencyScore',
                           'GiftTrend', 'DonationRate', 'DonorLifespan']
X, y = df[features], df['TARGET_B']
print(f"Predictors: {len(features)}")

# ------------------------------------------------------------
# 3. Models (imputation/scaling inside the pipeline = no fold leakage)
# ------------------------------------------------------------
models = {
    'Random Forest': make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features='sqrt',
                               random_state=SEED, n_jobs=-1)),
    'Gradient Boosting (XGBoost)': make_pipeline(
        SimpleImputer(strategy='median'),
        xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                          random_state=SEED, n_jobs=-1, verbosity=0)),
    'Neural Network (MLP)': make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, early_stopping=True,
                      max_iter=500, random_state=SEED)),
    'SVM (RBF)': make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        SVC(kernel='rbf', C=1, probability=True, random_state=SEED)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
print("\n5-fold cross-validated performance (de-duplicated data)")
print("-" * 60)
for name, model in models.items():
    s = cross_validate(model, X, y, cv=cv, scoring=['roc_auc', 'accuracy', 'f1'], n_jobs=1)
    rows.append({'Model': name,
                 'AUC_mean': s['test_roc_auc'].mean(), 'AUC_sd': s['test_roc_auc'].std(),
                 'Accuracy': s['test_accuracy'].mean(), 'F1': s['test_f1'].mean()})
    print(f"{name:30s} AUC {s['test_roc_auc'].mean():.4f} ± {s['test_roc_auc'].std():.4f}"
          f" | Acc {s['test_accuracy'].mean():.4f} | F1 {s['test_f1'].mean():.4f}")

results = pd.DataFrame(rows).sort_values('AUC_mean', ascending=False)
results.round(4).to_csv('model_comparison_cv.csv', index=False)
champion_name = results.iloc[0]['Model']
print(f"\nChampion by mean CV AUC: {champion_name}")

# ------------------------------------------------------------
# 4. Hold-out view of all models: ROC, KS, lift
# ------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
probs = {}
fig, ax = plt.subplots(figsize=(8, 6))
print("\nHold-out (30%) performance")
print("-" * 60)
for name, model in models.items():
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    probs[name] = p
    fpr, tpr, thr = roc_curve(y_te, p)
    ks = np.max(tpr - fpr)
    auc = roc_auc_score(y_te, p)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
    print(f"{name:30s} AUC {auc:.4f} | KS {ks:.4f} | KS cutoff {thr[np.argmax(tpr - fpr)]:.3f}")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('ROC curves on hold-out set (de-duplicated data)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('roc_curves.png', dpi=150); plt.show()

p = probs[champion_name]
order = np.argsort(p)[::-1]
ys = y_te.values[order]
base = ys.mean()
lift_rows = []
print(f"\nTargeting table - {champion_name} (base response rate {base:.1%})")
print(f"{'Depth':<8}{'Contacted':>10}{'Responders':>12}{'Precision':>11}{'Lift':>7}{'Capture':>9}")
for pct in [10, 20, 30, 50]:
    n = int(len(ys) * pct / 100)
    hits = ys[:n].sum()
    prec = hits / n
    lift_rows.append({'Depth': f'Top {pct}%', 'Contacted': n, 'Responders': int(hits),
                      'Precision': prec, 'Lift': prec / base, 'Capture': hits / ys.sum()})
    print(f"Top {pct:>2}%{n:>10,}{hits:>12,}{prec:>10.1%}{prec / base:>7.2f}{hits / ys.sum():>9.1%}")
pd.DataFrame(lift_rows).round(4).to_csv('targeting_table.csv', index=False)

# ------------------------------------------------------------
# 5. Interpretability for the tree-based champion (or RF if not tree-based)
# ------------------------------------------------------------
tree_name = champion_name if champion_name in ('Random Forest', 'Gradient Boosting (XGBoost)') else 'Random Forest'
pipe = models[tree_name]
imputer, est = pipe[0], pipe[-1]
X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=features)
sample = X_te_imp.sample(n=min(1000, len(X_te_imp)), random_state=SEED)
sv = shap.TreeExplainer(est).shap_values(sample)
if isinstance(sv, list):
    sv = sv[1]
elif getattr(sv, 'ndim', 2) == 3:
    sv = sv[:, :, 1]

shap.summary_plot(sv, sample, max_display=12, show=False)
plt.title(f'SHAP summary - {tree_name}'); plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight'); plt.show()

imp = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)
imp.round(4).to_csv('shap_importance.csv', header=['mean_abs_shap'])
print(f"\nTop 10 predictors by mean |SHAP| ({tree_name}):")
print(imp.head(10).round(4).to_string())

print("\nSaved: model_comparison_cv.csv, targeting_table.csv, shap_importance.csv,"
      " roc_curves.png, shap_summary.png")
