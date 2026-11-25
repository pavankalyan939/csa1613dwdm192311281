"""Q7 - Loan-default pipeline: ORIGINAL (flawed) vs REDESIGNED.
Dataset: synthetic bank-loan data (50,000 rows, ~3% defaults) generated below so the
script is self-contained.  To use a public dataset (e.g. UCI German Credit), replace
make_data() with a loader that returns a DataFrame with columns:
income, age, loan_amount, debt_ratio, credit_years, late_payments, occupation, default (0/1).
"""
import numpy as np, pandas as pd, json
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
rng = np.random.default_rng(42)

def make_data(n=50000):
    occ = rng.choice(['Salaried','Self-employed','Professional','Manual','Other'], n, p=[.35,.2,.15,.2,.1])
    mult = pd.Series(occ).map({'Salaried':1,'Self-employed':1.1,'Professional':1.8,'Manual':.7,'Other':.6}).values
    income = rng.lognormal(10.6, .75, n) * mult
    age = np.clip(rng.normal(40, 11, n), 21, 70)
    loan = income * rng.uniform(.2, 1.5, n)
    debt = rng.beta(2, 5, n)
    cy = np.clip(rng.normal(8, 4, n), 0, 30)
    late = rng.poisson(.5, n)
    eff = pd.Series(occ).map({'Salaried':0,'Self-employed':.6,'Professional':-.6,'Manual':.5,'Other':.3}).values
    z = .9*late + 2.2*debt - .7*np.log(income/40000) + eff - .03*(age-40) - .04*(cy-8)
    lo, hi = -15, 5
    for _ in range(60):                       # calibrate intercept to ~3% defaults
        b = (lo+hi)/2
        p = 1/(1+np.exp(-(z+b)))
        lo, hi = (b, hi) if p.mean() < .03 else (lo, b)
    y = (rng.random(n) < 1/(1+np.exp(-(z+b)))).astype(int)
    df = pd.DataFrame(dict(income=income, age=age, loan_amount=loan, debt_ratio=debt,
                           credit_years=cy, late_payments=late, occupation=occ, default=y))
    df.loc[rng.random(n) < .08, 'income'] = np.nan        # 8% missing income
    df.loc[rng.random(n) < .18, 'occupation'] = np.nan    # 18% missing occupation
    return df

df = make_data()
num = ['age','loan_amount','debt_ratio','credit_years','late_payments']
y = df['default'].values
tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=.30, stratify=y, random_state=1)
test_y = y[te_idx]
res = {'rows': len(df), 'default_rate_all': float(y.mean()), 'test_rows': len(te_idx),
       'test_defaults': int(test_y.sum()), 'baseline_always_no_accuracy': float(1-test_y.mean())}

def metrics(yt, yp, proba):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return dict(accuracy=accuracy_score(yt, yp), precision=precision_score(yt, yp, zero_division=0),
                recall=recall_score(yt, yp), f1=f1_score(yt, yp), roc_auc=roc_auc_score(yt, proba),
                pr_auc=average_precision_score(yt, proba), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

# ---------------- ORIGINAL (flawed) pipeline ----------------
o = df.copy()
true_income_sd = o['income'].std()                                   # SD of observed incomes
o['income'] = o['income'].fillna(o['income'].mean())                 # 1 global-mean imputation (all data)
res['orig_income_sd_after_mean_imputation'] = float(o['income'].std())
res['orig_income_sd_observed'] = float(true_income_sd)
o = o.drop(columns=['occupation'])                                   # 2 drop occupation
edges = np.linspace(o['income'].min(), o['income'].max(), 6)         # 3 equal-width, 5 bins
o['inc_bin'] = np.clip(np.digitize(o['income'], edges[1:-1]), 0, 4)
res['orig_income_bin_share'] = [float((o['inc_bin']==k).mean()) for k in range(5)]
Xo = pd.concat([pd.DataFrame(MinMaxScaler().fit_transform(o[num]), columns=num),       # 4 scaled on ALL data (leak)
                pd.get_dummies(o['inc_bin'], prefix='b').astype(float)], axis=1).values
rs = np.random.default_rng(7)
samp = rs.choice(tr_idx, size=int(.10*len(df)), replace=False)      # 5 random 10% sample
res['orig_train_rows'] = len(samp); res['orig_train_default_rate'] = float(y[samp].mean())
m = LogisticRegression(max_iter=2000).fit(Xo[samp], y[samp])
res['original'] = metrics(test_y, m.predict(Xo[te_idx]), m.predict_proba(Xo[te_idx])[:,1])

# ---------------- REDESIGNED pipeline (all fitting on TRAIN only) ----------------
tr = df.iloc[tr_idx].copy(); te = df.iloc[te_idx].copy()
for d in (tr, te):
    d['inc_missing'] = d['income'].isna().astype(float)
    d['occupation'] = d['occupation'].fillna('Unknown')                 # keep occupation (Unknown category)
grp_med = tr.groupby('occupation')['income'].median()                    # group-wise median, train only
for d in (tr, te):
    d['income'] = d['income'].fillna(d['occupation'].map(grp_med))
    d['log_inc'] = np.log1p(d['income'])                                  # reduce right skew
res['redesign_income_skew_raw'] = float(skew(tr['income'])); res['redesign_income_skew_log'] = float(skew(tr['log_inc']))
res['redesign_income_sd_after_group_median'] = float(tr['income'].std())
kb = KBinsDiscretizer(5, encode='onehot-dense', strategy='quantile').fit(tr[['income']])   # equal-frequency, train cuts
res['redesign_income_bin_share_train'] = [float(v) for v in kb.transform(tr[['income']]).mean(0)]
res['redesign_bin_edges'] = [float(v) for v in kb.bin_edges_[0]]
cols = num + ['log_inc']
corr = tr[cols].corr().abs(); drop = [c for c in cols if (corr[c].drop(c) > .9).any()][:0]
res['corr_filter_dropped'] = drop
sc = StandardScaler().fit(tr[cols])
def build(d):
    return np.hstack([sc.transform(d[cols]), kb.transform(d[['income']]), d[['inc_missing']].values,
                      pd.get_dummies(d['occupation']).reindex(columns=sorted(tr['occupation'].unique()), fill_value=0).astype(float).values])
Xtr, Xte = build(tr), build(te)
ytr = tr['default'].values
pos = np.where(ytr==1)[0]; neg = np.where(ytr==0)[0]
rs2 = np.random.default_rng(11)
neg_s = rs2.choice(neg, size=3*len(pos), replace=False)                  # stratified under-sampling 1:3 (train only)
idx = np.concatenate([pos, neg_s])
res['redesign_train_rows'] = len(idx); res['redesign_train_default_rate'] = float(ytr[idx].mean())
m2 = LogisticRegression(max_iter=2000).fit(Xtr[idx], ytr[idx])
res['redesigned'] = metrics(test_y, m2.predict(Xte), m2.predict_proba(Xte)[:,1])
json.dump(res, open('out/q7_results.json','w'), indent=1)
print(json.dumps(res, indent=1))

# ---------- Step 8: decision threshold chosen on a validation part of TRAIN (never on test) ----------
from sklearn.model_selection import train_test_split as tts
a, v = tts(np.arange(len(ytr)), test_size=.25, stratify=ytr, random_state=3)
pa = a[ytr[a]==1]; na = a[ytr[a]==0]
ia = np.concatenate([pa, np.random.default_rng(5).choice(na, size=3*len(pa), replace=False)])
m3 = LogisticRegression(max_iter=2000).fit(Xtr[ia], ytr[ia])
pv = m3.predict_proba(Xtr[v])[:,1]
thr = max(t for t in np.linspace(.05,.95,91) if recall_score(ytr[v], pv>=t) >= .50)   # highest threshold with recall>=50%
pt = m3.predict_proba(Xte)[:,1]
res['threshold_chosen_on_validation'] = float(thr)
res['redesigned_tuned'] = metrics(test_y, (pt>=thr).astype(int), pt)
json.dump(res, open('out/q7_results.json','w'), indent=1)
print('thr', thr, json.dumps(res['redesigned_tuned'], indent=1))
