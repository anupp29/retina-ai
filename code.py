# ============================================================
# RETINA AI - Student Dropout Risk Prediction
# End-to-End: Feature Engineering + GBM Ensemble + Multimodal DL
# ============================================================
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
TARGET = 'dropout_risk'
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ============================================================
# 1. LOAD
# ============================================================
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
att   = pd.read_csv("Attendance_series.csv")
notes = pd.read_csv("Counsellor_notes.csv")

# ============================================================
# 2. ATTENDANCE AGGREGATE FEATURES (vectorized)
# ============================================================
att['t'] = (att['semester'] - 1) * 8 + att['week']

overall = att.groupby('student_id')['attendance_pct'].agg(
    att_mean='mean', att_std='std', att_min='min', att_max='max').reset_index()

sem_pivot = att.pivot_table(index='student_id', columns='semester', values='attendance_pct', aggfunc='mean')
sem_pivot.columns = [f'att_sem{c}_mean' for c in sem_pivot.columns]
sem_pivot = sem_pivot.reset_index()

subj_pivot = att.pivot_table(index='student_id', columns='subject', values='attendance_pct', aggfunc='mean')
subj_pivot.columns = [f'att_{c}_mean' for c in subj_pivot.columns]
subj_pivot = subj_pivot.reset_index()

agg = att.groupby('student_id').agg(n=('t','count'), sum_t=('t','sum'), sum_y=('attendance_pct','sum'))
att['ty'] = att['t'] * att['attendance_pct']
att['tt'] = att['t'] ** 2
agg2 = att.groupby('student_id').agg(sum_ty=('ty','sum'), sum_tt=('tt','sum'))
agg = agg.join(agg2)
agg['att_slope'] = (agg['n']*agg['sum_ty'] - agg['sum_t']*agg['sum_y']) / (agg['n']*agg['sum_tt'] - agg['sum_t']**2)
slope = agg[['att_slope']].reset_index()

low_att = att.groupby('student_id')['attendance_pct'].apply(lambda x: (x < 0.5).sum()).reset_index(name='att_low_weeks')

att_feats = (overall.merge(sem_pivot, on='student_id')
                     .merge(subj_pivot, on='student_id')
                     .merge(slope, on='student_id')
                     .merge(low_att, on='student_id'))
att_feats['att_recent_vs_early'] = att_feats['att_sem3_mean'] - att_feats['att_sem1_mean']
att_feats['att_sem3_2_diff']     = att_feats['att_sem3_mean'] - att_feats['att_sem2_mean']

# ============================================================
# 3. COUNSELLOR NOTES FEATURES
# ============================================================
le_note = LabelEncoder()
notes['note_id'] = le_note.fit_transform(notes['counsellor_note'])

parts = notes['counsellor_note'].str.rstrip('.').str.split('. ', n=1, expand=True, regex=False)
notes['situation'] = parts[0]
le_sit = LabelEncoder()
notes['situation_id'] = le_sit.fit_transform(notes['situation'])

risk_kws = 'demotivat|financial|health|emergency|severe|multiple backlogs|part-time hours|unresponsive'
warn_kws = 'stress|struggl|focus|attendance|absence|family|monitor'
good_kws = 'well|good|no major issues|no further action|active'
notes['note_high_risk_kw'] = notes['counsellor_note'].str.lower().str.contains(risk_kws).astype(int)
notes['note_med_risk_kw']  = notes['counsellor_note'].str.lower().str.contains(warn_kws).astype(int)
notes['note_low_risk_kw']  = notes['counsellor_note'].str.lower().str.contains(good_kws).astype(int)

# ============================================================
# 4. TABULAR DERIVED FEATURES
# ============================================================
for df in [train, test]:
    cg = ['cgpa_sem1','cgpa_sem2','cgpa_sem3','cgpa_sem4']
    df['cgpa_mean']  = df[cg].mean(axis=1)
    df['cgpa_std']   = df[cg].std(axis=1)
    df['cgpa_min']   = df[cg].min(axis=1)
    df['cgpa_trend'] = df['cgpa_sem4'] - df['cgpa_sem1']
    df['cgpa_d21'] = df['cgpa_sem2'] - df['cgpa_sem1']
    df['cgpa_d32'] = df['cgpa_sem3'] - df['cgpa_sem2']
    df['cgpa_d43'] = df['cgpa_sem4'] - df['cgpa_sem3']
    df['total_backlogs'] = df['backlogs_sem1'] + df['backlogs_sem2'] + df['backlogs_sem3']
    df['backlog_trend']  = df['backlogs_sem3'] - df['backlogs_sem1']
    df['bl_d21'] = df['backlogs_sem2'] - df['backlogs_sem1']
    df['bl_d32'] = df['backlogs_sem3'] - df['backlogs_sem2']
    df['commute_missing']    = df['commute_time_mins'].isnull().astype(int)
    df['parent_edu_missing'] = df['parent_education'].isnull().astype(int)

med = train['commute_time_mins'].median()
for df in [train, test]:
    df['commute_time_mins'] = df['commute_time_mins'].fillna(med)
    df['parent_education']  = df['parent_education'].fillna('Unknown')

# ============================================================
# 5. MERGE EVERYTHING
# ============================================================
notes_cols = ['student_id','note_id','situation_id','note_high_risk_kw','note_med_risk_kw','note_low_risk_kw']
train = train.merge(att_feats, on='student_id', how='left').merge(notes[notes_cols], on='student_id', how='left')
test  = test.merge(att_feats, on='student_id', how='left').merge(notes[notes_cols], on='student_id', how='left')

for df in [train, test]:
    df['att_d21'] = df['att_sem2_mean'] - df['att_sem1_mean']
    df['att_d32'] = df['att_sem3_mean'] - df['att_sem2_mean']
    df['cgpa_div_backlog'] = df['cgpa_mean'] / (1 + df['total_backlogs'])
    df['cgpa_x_backlog']   = df['cgpa_mean'] * (1 + df['total_backlogs'])

# ============================================================
# 6. ENCODE LOW-CARDINALITY CATEGORICALS
# ============================================================
cat_cols = ['branch','gender','hostel_status','family_income','parent_education']
for c in cat_cols:
    le = LabelEncoder()
    full = pd.concat([train[c], test[c]], axis=0).astype(str)
    le.fit(full)
    train[c+'_enc'] = le.transform(train[c].astype(str))
    test[c+'_enc']  = le.transform(test[c].astype(str))

# ============================================================
# 7. OOF TARGET ENCODING for note_id & situation_id
# ============================================================
def kfold_te_multiclass(train_df, test_df, col, target_col, n_classes=3, n_splits=5, smoothing=10, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(train_df), n_classes))
    priors = train_df[target_col].value_counts(normalize=True).reindex(range(n_classes)).values
    priors_s = pd.Series(priors, index=range(n_classes))
    for tr_idx, val_idx in skf.split(train_df, train_df[target_col]):
        tr = train_df.iloc[tr_idx]
        counts = tr.groupby(col)[target_col].value_counts().unstack(fill_value=0).reindex(columns=range(n_classes), fill_value=0)
        totals = counts.sum(axis=1)
        enc = counts.add(smoothing * priors, axis=1).div(totals + smoothing, axis=0)
        val_vals = train_df.iloc[val_idx][col]
        e = enc.reindex(val_vals).fillna(priors_s)
        oof[val_idx] = e.values
    counts_full = train_df.groupby(col)[target_col].value_counts().unstack(fill_value=0).reindex(columns=range(n_classes), fill_value=0)
    totals_full = counts_full.sum(axis=1)
    enc_full = counts_full.add(smoothing * priors, axis=1).div(totals_full + smoothing, axis=0)
    e_test = enc_full.reindex(test_df[col]).fillna(priors_s)
    return oof, e_test.values

oof_te, test_te = kfold_te_multiclass(train, test, 'note_id', TARGET, smoothing=10)
for i in range(3):
    train[f'note_te_p{i}'] = oof_te[:, i]
    test[f'note_te_p{i}']  = test_te[:, i]

oof_te_sit, test_te_sit = kfold_te_multiclass(train, test, 'situation_id', TARGET, smoothing=15)
for i in range(3):
    train[f'sit_te_p{i}'] = oof_te_sit[:, i]
    test[f'sit_te_p{i}']  = test_te_sit[:, i]

for df in [train, test]:
    df['note_risk_spread'] = df['note_te_p2'] - df['note_te_p0']

# ============================================================
# 8. FINAL FEATURE LIST
# ============================================================
FEATURES = [c for c in train.columns if c not in ['student_id', TARGET, 'counsellor_note'] + cat_cols]
CAT_FEATURES = ['branch_enc','gender_enc','hostel_status_enc','family_income_enc','parent_education_enc','note_id','situation_id']

X = train[FEATURES].copy()
y = train[TARGET].copy()
X_test = test[FEATURES].copy()
cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES]
print("Features:", len(FEATURES), "| Train:", X.shape, "| Test:", X_test.shape)

# ============================================================
# 9. GBM ENSEMBLE: 5-FOLD CV (LightGBM + XGBoost + CatBoost + ExtraTrees)
# ============================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(X), 3)); pred_lgb = np.zeros((len(X_test), 3))
oof_xgb = np.zeros((len(X), 3)); pred_xgb = np.zeros((len(X_test), 3))
oof_cb  = np.zeros((len(X), 3)); pred_cb  = np.zeros((len(X_test), 3))
oof_et  = np.zeros((len(X), 3)); pred_et  = np.zeros((len(X_test), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    sw_tr = compute_sample_weight('balanced', y_tr)
    cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tr)

    lgb_model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, n_estimators=2000, learning_rate=0.03,
        num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1)
    lgb_model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)],
                   eval_metric='multi_logloss', categorical_feature=CAT_FEATURES,
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)
    pred_lgb += lgb_model.predict_proba(X_test) / N_SPLITS

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, n_estimators=2000, learning_rate=0.03,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, eval_metric='mlogloss', early_stopping_rounds=100)
    xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)
    pred_xgb += xgb_model.predict_proba(X_test) / N_SPLITS

    cb_model = CatBoostClassifier(
        loss_function='MultiClass', iterations=2000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, class_weights=cw.tolist(), cat_features=cat_idx,
        random_seed=SEED, verbose=False, early_stopping_rounds=100)
    cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_cb[val_idx] = cb_model.predict_proba(X_val)
    pred_cb += cb_model.predict_proba(X_test) / N_SPLITS

    et_model = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3,
                                     class_weight='balanced', n_jobs=-1, random_state=SEED)
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict_proba(X_val)
    pred_et += et_model.predict_proba(X_test) / N_SPLITS

    fold_f1 = f1_score(y_val, np.argmax((oof_lgb[val_idx]+oof_xgb[val_idx]+oof_cb[val_idx]+oof_et[val_idx])/4, axis=1), average='macro')
    print(f"Fold {fold+1} GBM ensemble macro-F1: {fold_f1:.4f}")

oof_ens4  = (oof_lgb + oof_xgb + oof_cb + oof_et) / 4
test_ens4 = (pred_lgb + pred_xgb + pred_cb + pred_et) / 4
print("GBM 4-model OOF macro-F1 (raw argmax):", f1_score(y, np.argmax(oof_ens4, axis=1), average='macro'))

# ============================================================
# 10. MULTIMODAL DEEP LEARNING (Tabular MLP + Attendance LSTM + Notes Text LSTM)
# ============================================================
all_students = pd.concat([train['student_id'], test['student_id']]).reset_index(drop=True)
n_train = len(train)

# --- 10a. Attendance sequence tensor (N, 24, 3) ---
subjects = sorted(att['subject'].unique())
piv = att.pivot_table(index='student_id', columns=['t','subject'], values='attendance_pct')
full_cols = pd.MultiIndex.from_product([range(1,25), subjects], names=['t','subject'])
piv = piv.reindex(index=all_students, columns=full_cols)
row_mean = piv.mean(axis=1)
piv = piv.apply(lambda col: col.fillna(row_mean))
piv = piv.fillna(piv.values.mean())
ATT_SEQ = piv.values.reshape(len(all_students), 24, len(subjects)).astype(np.float32)
ATT_SEQ_train, ATT_SEQ_test = ATT_SEQ[:n_train], ATT_SEQ[n_train:]

# --- 10b. Counsellor note text -> token sequences ---
notes_s = notes.set_index('student_id').reindex(all_students)['counsellor_note']
tok_series = notes_s.str.lower().str.replace('.', ' ', regex=False).str.replace(',', ' ', regex=False).str.split()
vocab = sorted(set(w for toks in tok_series for w in toks))
word2idx = {w: i+1 for i, w in enumerate(vocab)}
MAX_LEN, VOCAB_SIZE = 10, len(vocab)
def tok_to_seq(toks):
    seq = [word2idx[w] for w in toks][:MAX_LEN]
    return seq + [0] * (MAX_LEN - len(seq))
TEXT_SEQ = np.array([tok_to_seq(t) for t in tok_series], dtype=np.int64)
TEXT_SEQ_train, TEXT_SEQ_test = TEXT_SEQ[:n_train], TEXT_SEQ[n_train:]

# --- 10c. Tabular numeric (scaled) + categorical (for embeddings) ---
EMBED_CATS = CAT_FEATURES
NUMERIC_FEATS = [c for c in FEATURES if c not in EMBED_CATS]

scaler = StandardScaler()
NUM_train = scaler.fit_transform(X[NUMERIC_FEATS].values.astype(np.float32))
NUM_test  = scaler.transform(X_test[NUMERIC_FEATS].values.astype(np.float32))
CAT_train = X[EMBED_CATS].values.astype(np.int64)
CAT_test  = X_test[EMBED_CATS].values.astype(np.int64)
CAT_CARDINALITIES = [int(X[c].max()) + 2 for c in EMBED_CATS]
y_arr = y.values.astype(np.int64)

# --- 10d. Dataset + Model ---
class StudentDataset(Dataset):
    def __init__(self, num_f, cat_f, att_seq, text_seq, labels=None):
        self.num_f, self.cat_f = torch.tensor(num_f, dtype=torch.float32), torch.tensor(cat_f, dtype=torch.long)
        self.att_seq, self.text_seq = torch.tensor(att_seq, dtype=torch.float32), torch.tensor(text_seq, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    def __len__(self): return len(self.num_f)
    def __getitem__(self, i):
        if self.labels is not None:
            return self.num_f[i], self.cat_f[i], self.att_seq[i], self.text_seq[i], self.labels[i]
        return self.num_f[i], self.cat_f[i], self.att_seq[i], self.text_seq[i]

class MultiModalNet(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, vocab_size, n_subjects=3, n_classes=3):
        super().__init__()
        emb_dims = [min(24, (c+1)//2) for c in cat_cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cat_cardinalities, emb_dims)])
        total_emb = sum(emb_dims)
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_numeric + total_emb, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64))
        self.att_lstm = nn.LSTM(input_size=n_subjects, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.text_emb = nn.Embedding(vocab_size + 1, 16, padding_idx=0)
        self.text_lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        combined_dim = 64 + 32*2 + 16*2
        self.head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))
    def forward(self, x_num, x_cat, x_att, x_text):
        embs = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        tab = self.tab_mlp(torch.cat([x_num, embs], dim=1))
        _, (h_att, _) = self.att_lstm(x_att)
        att_out = torch.cat([h_att[0], h_att[1]], dim=1)
        e = self.text_emb(x_text)
        _, (h_txt, _) = self.text_lstm(e)
        txt_out = torch.cat([h_txt[0], h_txt[1]], dim=1)
        return self.head(torch.cat([tab, att_out, txt_out], dim=1))

# --- 10e. 5-fold training with early stopping on macro-F1 ---
oof_nn = np.zeros((len(X), 3))
pred_nn = np.zeros((len(X_test), 3))
EPOCHS, PATIENCE, BATCH_SIZE = 60, 8, 128

for fold, (tr_idx, val_idx) in enumerate(skf.split(NUM_train, y_arr)):
    train_ds = StudentDataset(NUM_train[tr_idx], CAT_train[tr_idx], ATT_SEQ_train[tr_idx], TEXT_SEQ_train[tr_idx], y_arr[tr_idx])
    val_ds   = StudentDataset(NUM_train[val_idx], CAT_train[val_idx], ATT_SEQ_train[val_idx], TEXT_SEQ_train[val_idx], y_arr[val_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = MultiModalNet(NUM_train.shape[1], CAT_CARDINALITIES, VOCAB_SIZE).to(device)
    cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_arr[tr_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1, best_state, patience_ctr = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for num_f, cat_f, att_f, txt_f, lbl in train_dl:
            num_f, cat_f, att_f, txt_f, lbl = [t.to(device) for t in (num_f, cat_f, att_f, txt_f, lbl)]
            optimizer.zero_grad()
            loss = criterion(model(num_f, cat_f, att_f, txt_f), lbl)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for num_f, cat_f, att_f, txt_f, lbl in val_dl:
                num_f, cat_f, att_f, txt_f = [t.to(device) for t in (num_f, cat_f, att_f, txt_f)]
                out = model(num_f, cat_f, att_f, txt_f)
                val_preds.append(out.argmax(1).cpu().numpy())
                val_true.append(lbl.numpy())
        val_preds, val_true = np.concatenate(val_preds), np.concatenate(val_true)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1, best_state, patience_ctr = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    print(f"Fold {fold+1} NN best val macro-F1: {best_f1:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_dl_full = DataLoader(StudentDataset(NUM_train[val_idx], CAT_train[val_idx], ATT_SEQ_train[val_idx], TEXT_SEQ_train[val_idx]), batch_size=512, shuffle=False)
        probs = []
        for num_f, cat_f, att_f, txt_f in val_dl_full:
            num_f, cat_f, att_f, txt_f = [t.to(device) for t in (num_f, cat_f, att_f, txt_f)]
            probs.append(torch.softmax(model(num_f, cat_f, att_f, txt_f), dim=1).cpu().numpy())
        oof_nn[val_idx] = np.concatenate(probs)

        test_dl = DataLoader(StudentDataset(NUM_test, CAT_test, ATT_SEQ_test, TEXT_SEQ_test), batch_size=512, shuffle=False)
        probs = []
        for num_f, cat_f, att_f, txt_f in test_dl:
            num_f, cat_f, att_f, txt_f = [t.to(device) for t in (num_f, cat_f, att_f, txt_f)]
            probs.append(torch.softmax(model(num_f, cat_f, att_f, txt_f), dim=1).cpu().numpy())
        pred_nn += np.concatenate(probs) / N_SPLITS

print("NN OOF macro-F1 (raw argmax):", f1_score(y, np.argmax(oof_nn, axis=1), average='macro'))

# ============================================================
# 11. BLEND GBM + DL, WEIGHT TUNE, CONFUSION MATRIX, SUBMISSION
# ============================================================
best_score, best_beta, best_w = -1, 0.5, (1.0,1.0,1.0)
for beta in np.arange(0.0, 1.01, 0.1):
    blend = beta*oof_ens4 + (1-beta)*oof_nn
    for w1 in np.arange(0.4, 1.81, 0.02):
        for w2 in np.arange(0.3, 1.61, 0.02):
            w = np.array([1.0, w1, w2])
            score = f1_score(y, np.argmax(blend*w, axis=1), average='macro')
            if score > best_score:
                best_score, best_beta, best_w = score, beta, w

print(f"\nFINAL blended tuned macro-F1: {best_score:.4f} | beta(GBM weight)={best_beta:.2f} | class weights={best_w}")

final_oof  = best_beta*oof_ens4 + (1-best_beta)*oof_nn
final_test = best_beta*test_ens4 + (1-best_beta)*pred_nn

oof_preds_final = np.argmax(final_oof*best_w, axis=1)
print(classification_report(y, oof_preds_final, target_names=['Low','Medium','High']))

cm = confusion_matrix(y, oof_preds_final)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low','Medium','High'], yticklabels=['Low','Medium','High'])
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title(f'Final Blended OOF Confusion Matrix (Macro-F1={best_score:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=150)
plt.show()

test_preds = np.argmax(final_test*best_w, axis=1)
submission = pd.DataFrame({'student_id': test['student_id'], 'dropout_risk': test_preds})
submission.to_csv('submission_final.csv', index=False)
print(submission['dropout_risk'].value_counts(normalize=True))
print("Saved submission_final.csv")