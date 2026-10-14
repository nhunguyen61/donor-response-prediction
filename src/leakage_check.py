# -*- coding: utf-8 -*-
"""
Leakage check for the PVA donor-response models.
Run in Colab with pva_partition.csv uploaded (same as the main notebook).
Takes about 2-3 minutes. Copy ALL printed output back.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("pva_partition.csv")

# ---- Same cleaning as the main notebook ----
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

print("=" * 60)
print("1. DATASET SHAPE")
print("=" * 60)
print("Rows:", len(df), "| Unique IDs:", df['ID'].nunique())
print("Partition counts:", df['Partition'].value_counts().to_dict())
print("Raw predictor columns:", len(raw_features))
print(raw_features)

# ---- 2. Duplicate rows ----
print("\n" + "=" * 60)
print("2. DUPLICATE PREDICTOR ROWS (ignoring ID)")
print("=" * 60)
key = df[raw_features].astype(str).agg('|'.join, axis=1)
df['_key'] = key
print("Distinct predictor profiles:", key.nunique(), "of", len(df), "rows")
print(f"Rows that duplicate another row: {key.duplicated(keep=False).mean():.1%}")

train_keys = set(df.loc[df['Partition'] == 'Training', '_key'])
val = df[df['Partition'] == 'Validation']
in_train = val['_key'].isin(train_keys)
print(f"Validation rows whose exact profile also appears in Training: {in_train.mean():.1%}")

# Do duplicates share the same target?
g = df.groupby('_key')['TARGET_B'].agg(['nunique', 'size'])
g = g[g['size'] > 1]
print(f"Duplicate groups with ONE consistent target: {(g['nunique'] == 1).mean():.1%}")

# ---- 3. Re-test the champion model ----
print("\n" + "=" * 60)
print("3. RANDOM FOREST AUC: ORIGINAL vs CLEAN VALIDATION")
print("=" * 60)
eng = pd.DataFrame(index=df.index)
eng['CardPromRatio'] = df['PromCntCardAll'] / (df['PromCntAll'] + 1)
eng['CardResponseRate'] = df['GiftCntCardAll'] / (df['PromCntCardAll'] + 1)
eng['FreqRecencyScore'] = df['GiftCnt36'] / (df['GiftTimeLast'] + 1)
eng['GiftTrend'] = df['GiftAvg36'] / (df['GiftAvgAll'] + 1)
eng['DonationRate'] = df['GiftCntAll'] / (df['PromCntAll'] + 1)
eng['DonorLifespan'] = df['GiftTimeFirst'] - df['GiftTimeLast']
X_all = pd.concat([df[raw_features], eng], axis=1)
features = list(X_all.columns)

is_tr = df['Partition'] == 'Training'
is_va = df['Partition'] == 'Validation'
medians = X_all[is_tr].median()            # train-only medians (fixes a small leak)
X_all = X_all.fillna(medians)

X_tr, y_tr = X_all[is_tr], df.loc[is_tr, 'TARGET_B']
rf = RandomForestClassifier(n_estimators=200, max_depth=12, max_features='sqrt',
                            random_state=42, n_jobs=-1).fit(X_tr, y_tr)

va_idx = df.index[is_va]
clean_idx = va_idx[~in_train.values]
p_all = rf.predict_proba(X_all.loc[va_idx])[:, 1]
print(f"AUC on full validation set:              {roc_auc_score(df.loc[va_idx, 'TARGET_B'], p_all):.4f}")
if len(clean_idx) > 100 and df.loc[clean_idx, 'TARGET_B'].nunique() == 2:
    p_clean = rf.predict_proba(X_all.loc[clean_idx])[:, 1]
    print(f"AUC on validation rows NOT seen in train: {roc_auc_score(df.loc[clean_idx, 'TARGET_B'], p_clean):.4f}"
          f"  (n = {len(clean_idx):,})")
else:
    print("Too few unseen validation rows to score.")

# ---- 4. Drop-one-feature test ----
print("\n" + "=" * 60)
print("4. AUC WHEN EACH TOP FEATURE IS REMOVED (full validation)")
print("=" * 60)
top = pd.Series(rf.feature_importances_, index=features).nlargest(5).index
y_va = df.loc[va_idx, 'TARGET_B']
for f in top:
    cols = [c for c in features if c != f]
    m = RandomForestClassifier(n_estimators=100, max_depth=12, max_features='sqrt',
                               random_state=42, n_jobs=-1).fit(X_tr[cols], y_tr)
    auc = roc_auc_score(y_va, m.predict_proba(X_all.loc[va_idx, cols])[:, 1])
    print(f"without {f:20s}: AUC {auc:.4f}")
