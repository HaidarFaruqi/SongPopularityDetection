# 🎓 Strategi Improvisasi untuk Mahasiswa

## 🎯 Konteks: Mahasiswa Mengikuti Kompetisi ML

Anda sudah punya baseline dengan **OOF RMSE ~15.5-16.0** dari Siklus 5. Sekarang pertanyaannya: **"Apa yang harus dilakukan selanjutnya untuk naik ke ranking atas?"**

---

## 📊 Current State vs Target

| Metric | Current (Siklus 5) | Good Score | Excellent Score |
|--------|-------------------|------------|-----------------|
| **OOF RMSE** | ~15.5-16.0 | ~15.0-15.3 | < 15.0 |
| **Features** | ~67 features | 70-100 features | 60-80 (optimal) |
| **CV Stability** | Std ~0.03-0.05 | Std < 0.03 | Std < 0.02 |

**Target Improvement:** 0.3-0.5 RMSE untuk masuk top 10%

---

## 🔥 Prioritas Improvisasi (Quick Wins First!)

### Priority 1: Quick Wins (1-2 hari) ⚡
**Goal:** 0.1-0.2 RMSE improvement dengan effort minimal

### Priority 2: Medium Effort (3-5 hari) 🎯
**Goal:** 0.2-0.3 RMSE improvement dengan feature engineering

### Priority 3: Advanced (1-2 minggu) 🚀
**Goal:** 0.3-0.5+ RMSE improvement dengan ensemble & tuning

---

## ⚡ PRIORITY 1: Quick Wins (Mulai Dari Sini!)

### 1.1 Hyperparameter Tuning Manual (Paling Mudah!)

**Logika Mahasiswa:**
> "Saat ini pakai default parameters. Kalau tuning dikit, pasti bisa improve!"

**Action Plan:**

```python
# EXPERIMENT 1: Coba learning rate lebih kecil
lgbm_params = {
    'n_estimators': 3000,        # Tambah dari 2000
    'learning_rate': 0.005,      # Kurangi dari 0.01 (slower = more precise)
    'num_leaves': 31,
    # ... rest sama
}
```

**Expected:** +0.05-0.1 RMSE improvement

```python
# EXPERIMENT 2: Coba tree lebih complex
lgbm_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'num_leaves': 63,            # Tambah dari 31 (more complex)
    'max_depth': 8,              # Dari -1 (unlimited) ke 8
    # ... rest sama
}
```

**Expected:** +0.03-0.08 RMSE improvement

**Cara Test:**
1. Copy notebook ke notebook baru
2. Ubah parameter di Cell 3 (lgbm_params)
3. Run Cell 9 (training)
4. Compare OOF RMSE dengan baseline
5. Catat hasil di Excel/spreadsheet

**Tips Mahasiswa:**
- Test 1 parameter sekaligus (jangan langsung banyak!)
- Catat semua hasil di spreadsheet
- Gunakan Google Colab gratis (jangan waste laptop battery)

---

### 1.2 Feature Selection (Remove Noise!)

**Logika Mahasiswa:**
> "Cell 12 shows top features. Mungkin banyak features yang ga penting malah bikin noise!"

**Action Plan:**

```python
# Setelah training, analyze feature importance
# Copy dari Cell 12 output

# Lihat features dengan importance < 1% dari total
low_importance = feature_importance[
    (feature_importance['mean'] / feature_importance['mean'].sum() * 100) < 1.0
]

print(f"Low importance features ({len(low_importance)}):")
print(low_importance.index.tolist())
```

**Remove Low-Importance Features:**

Di Cell 8, tambahkan exclude:

```python
# Setelah base_features = [...]
# Add manual exclusions based on feature importance analysis
manual_exclude = [
    'lyrics_feature_15',  # Example: low importance
    'lyrics_feature_17',
    'has_special_edition',  # Example: very rare event
    # ... add more based on analysis
]

base_features = [f for f in base_features if f not in manual_exclude]
```

**Expected:** +0.05-0.15 RMSE improvement (kadang malah worse - test!)

**Tips Mahasiswa:**
- Jangan remove semua low-importance sekaligus
- Remove 5-10 features dulu, test
- Kadang remove features bisa worse, itu normal!

---

### 1.3 Simple Stacking/Averaging

**Logika Mahasiswa:**
> "Saat ini cuma pakai 1 model final. Kalau average 5 fold models, harusnya lebih robust!"

**Action Plan:**

Di Cell 10, ganti final prediction:

```python
# OLD WAY (train 1 final model)
# final_model.fit(X_train_final, y_train_final)
# predictions = final_model.predict(X_test)

# NEW WAY (average 5 fold models)
print("\n🔄 Using fold models for prediction (more robust)...")

# Prepare test with fold-specific encoding (5 times)
all_fold_predictions = []

for fold_idx, fold_model in enumerate(trained_models, 1):
    # Encode test using full train as reference
    df_test_fold = create_fold_features(df_test, df_train_full, 'popularity')
    X_test_fold = df_test_fold[all_features_used]

    # Predict
    fold_pred = fold_model.predict(X_test_fold)
    all_fold_predictions.append(fold_pred)
    print(f"  Fold {fold_idx}: mean={fold_pred.mean():.2f}, std={fold_pred.std():.2f}")

# Average predictions
predictions = np.mean(all_fold_predictions, axis=0)
predictions = np.clip(predictions, 0, 100)

print(f"\n✅ Averaged predictions from {len(trained_models)} models")
print(f"   Mean: {predictions.mean():.2f}")
print(f"   Std:  {predictions.std():.2f}")
```

**Expected:** +0.03-0.10 RMSE improvement

**Tips Mahasiswa:**
- Ini "ensemble" paling sederhana
- Hampir selalu improve sedikit
- Low risk, easy win!

---

## 🎯 PRIORITY 2: Medium Effort (Feature Engineering)

### 2.1 Artist-Genre Interaction Features

**Logika Mahasiswa:**
> "Mungkin artist tertentu lebih populer di genre tertentu. Interaksi artist-genre belum di-capture!"

**Action Plan:**

Tambahkan di `create_fold_features()` (Cell 2):

```python
# Add to end of create_fold_features() function:

# Artist-Genre Interaction (NEW!)
if all(col in df_fold.columns for col in ['artists', 'track_genre']):
    # Create artist-genre combination
    df_fold['artist_genre'] = df_fold['artists'] + '_' + df_fold['track_genre']

    # Encode using reference data
    artist_genre_mean = df_reference.groupby(
        df_reference['artists'] + '_' + df_reference['track_genre']
    )[target_col].mean()

    df_fold['artist_genre_avg_pop'] = df_fold['artist_genre'].map(artist_genre_mean).fillna(global_mean)

return df_fold
```

**Expected:** +0.05-0.15 RMSE improvement

---

### 2.2 Temporal Features per Genre

**Logika Mahasiswa:**
> "Mungkin lagu rock dari tahun 1980 beda popularitasnya dengan lagu pop dari 1980!"

**Action Plan:**

Tambahkan di `create_fold_features()`:

```python
# Genre-Year Interaction (NEW!)
if all(col in df_fold.columns for col in ['track_genre', 'release_year']):
    # Is this genre getting more/less popular over time?
    genre_year_trends = df_reference.groupby(
        ['track_genre', (df_reference['release_year'] // 10) * 10]
    )[target_col].mean().to_dict()

    df_fold['decade_temp'] = (df_fold['release_year'] // 10) * 10
    df_fold['genre_decade_avg'] = df_fold.apply(
        lambda row: genre_year_trends.get((row['track_genre'], row['decade_temp']), global_mean),
        axis=1
    )
    df_fold.drop('decade_temp', axis=1, inplace=True)

return df_fold
```

**Expected:** +0.05-0.10 RMSE improvement

---

### 2.3 Better Lyrics Processing

**Logika Mahasiswa:**
> "Saat ini cuma pakai TF-IDF + SVD. Mungkin bisa extract sentiment atau topic!"

**Action Plan:**

**Option A: Sentiment Analysis (Simple)**

```python
# Install (if needed)
# !pip install textblob

from textblob import TextBlob

def add_lyrics_sentiment(df):
    """Add sentiment features from lyrics"""
    if 'lyrics' not in df.columns:
        return df

    print("Extracting lyrics sentiment...")

    sentiments = []
    for lyrics in df['lyrics'].fillna(''):
        if len(lyrics) > 0:
            blob = TextBlob(lyrics)
            sentiment = blob.sentiment.polarity  # -1 to 1
        else:
            sentiment = 0
        sentiments.append(sentiment)

    df['lyrics_sentiment'] = sentiments
    return df

# Add to Cell 7, after SVD:
df_train = add_lyrics_sentiment(df_train)
df_test = add_lyrics_sentiment(df_test)
```

**Option B: Topic Keywords (Manual)**

```python
def add_lyrics_topics(df):
    """Add boolean features for common topics"""
    if 'lyrics' not in df.columns:
        return df

    # Define topic keywords
    topics = {
        'love_theme': ['love', 'heart', 'kiss', 'baby', 'darling'],
        'party_theme': ['party', 'dance', 'tonight', 'night', 'club'],
        'sad_theme': ['cry', 'tears', 'pain', 'hurt', 'alone'],
        'rebel_theme': ['fight', 'rebel', 'break', 'free', 'wild']
    }

    for topic_name, keywords in topics.items():
        df[topic_name] = df['lyrics'].fillna('').str.lower().apply(
            lambda x: int(any(kw in x for kw in keywords))
        )

    return df

# Add to Cell 7:
df_train = add_lyrics_topics(df_train)
df_test = add_lyrics_topics(df_test)
```

**Expected:** +0.05-0.12 RMSE improvement

---

### 2.4 Better Handling of Rare Artists/Genres

**Logika Mahasiswa:**
> "Artist dengan 1-2 songs aja mungkin unreliable statisticsnya. Perlu smoothing!"

**Action Plan:**

Update `create_fold_features()` dengan smoothing:

```python
# Add after artist statistics calculation:

# Smoothing untuk rare artists (Bayesian smoothing)
artist_counts_ref = df_reference.groupby('artists').size()

# Only trust statistics if artist has enough songs
MIN_SONGS_THRESHOLD = 5

df_fold['artist_song_count'] = df_fold['artists'].map(artist_counts_ref).fillna(1)

# If artist has < 5 songs, smooth towards global mean
df_fold['artist_avg_pop_smoothed'] = df_fold.apply(
    lambda row: (
        row['artist_avg_pop'] * min(row['artist_song_count'], MIN_SONGS_THRESHOLD) +
        global_mean * (MIN_SONGS_THRESHOLD - min(row['artist_song_count'], MIN_SONGS_THRESHOLD))
    ) / MIN_SONGS_THRESHOLD,
    axis=1
)

# Same for genre (though genres usually have many songs)
genre_counts_ref = df_reference.groupby('track_genre').size()
df_fold['genre_song_count'] = df_fold['track_genre'].map(genre_counts_ref).fillna(1)

df_fold['genre_avg_pop_smoothed'] = df_fold.apply(
    lambda row: (
        row['genre_avg_pop'] * min(row['genre_song_count'], MIN_SONGS_THRESHOLD) +
        global_mean * (MIN_SONGS_THRESHOLD - min(row['genre_song_count'], MIN_SONGS_THRESHOLD))
    ) / MIN_SONGS_THRESHOLD,
    axis=1
)
```

**Expected:** +0.03-0.08 RMSE improvement

---

## 🚀 PRIORITY 3: Advanced (Kalau Masih Ada Waktu)

### 3.1 Automated Hyperparameter Tuning dengan Optuna

**Logika Mahasiswa:**
> "Manual tuning lama banget. Pakai Optuna biar otomatis!"

**Action Plan:**

```python
# Install
!pip install optuna

import optuna

def objective(trial):
    """Optuna objective function"""

    # Define search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Quick 3-fold CV for speed
    results = train_with_proper_cv(
        df_train,
        base_features,
        target_col='popularity',
        cv_folds=3  # Faster
    )

    return results['overall_rmse']

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # 50 trials = ~10-20 jam di Colab

print(f"Best RMSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

**Expected:** +0.10-0.25 RMSE improvement

**Warning:** Ini LAMA! 50 trials × 3-fold CV = ~10-20 jam. Run overnight!

---

### 3.2 Multi-Model Ensemble

**Logika Mahasiswa:**
> "Kalau LightGBM aja, kenapa ga gabung dengan XGBoost atau CatBoost?"

**Action Plan:**

```python
# Install
!pip install xgboost catboost

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Train 3 different models
models_ensemble = []

# Model 1: LightGBM (existing)
lgbm = LGBMRegressor(**lgbm_params)
lgbm.fit(X_train_final, y_train_final)
models_ensemble.append(('lgbm', lgbm))

# Model 2: XGBoost
xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train_final, y_train_final)
models_ensemble.append(('xgb', xgb))

# Model 3: CatBoost
cat = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.01,
    depth=8,
    random_state=42,
    verbose=False
)
cat.fit(X_train_final, y_train_final)
models_ensemble.append(('cat', cat))

# Predict with all models
predictions_ensemble = []
for name, model in models_ensemble:
    pred = model.predict(X_test)
    predictions_ensemble.append(pred)
    print(f"{name}: mean={pred.mean():.2f}, std={pred.std():.2f}")

# Average
predictions = np.mean(predictions_ensemble, axis=0)
predictions = np.clip(predictions, 0, 100)

print(f"\nEnsemble: mean={predictions.mean():.2f}, std={predictions.std():.2f}")
```

**Expected:** +0.10-0.20 RMSE improvement

**Tips:** Weighted average kadang lebih baik:
```python
# Weighted average based on CV performance
weights = [0.5, 0.3, 0.2]  # LightGBM 50%, XGB 30%, Cat 20%
predictions = np.average(predictions_ensemble, axis=0, weights=weights)
```

---

### 3.3 Neural Network Features (Advanced!)

**Logika Mahasiswa:**
> "Lyrics punya sequence structure. Mungkin LSTM/BERT bisa extract better features!"

**Action Plan:**

```python
# Install
!pip install transformers torch

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_bert_embeddings(texts, max_length=128):
    """Get BERT embeddings for texts"""
    embeddings = []

    for text in texts:
        # Tokenize
        inputs = tokenizer(
            text[:512],  # BERT max 512 tokens
            return_tensors='pt',
            max_length=max_length,
            padding='max_length',
            truncation=True
        )

        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] token

        embeddings.append(embedding[0])

    return np.array(embeddings)

# Process lyrics (SLOW! ~1-2 hours)
print("Extracting BERT embeddings (this will take a while)...")
train_bert = get_bert_embeddings(df_train['lyrics'].fillna('').tolist())
test_bert = get_bert_embeddings(df_test['lyrics'].fillna('').tolist())

# Add to dataframe
for i in range(train_bert.shape[1]):
    df_train[f'bert_feature_{i}'] = train_bert[:, i]
    df_test[f'bert_feature_{i}'] = test_bert[:, i]

print(f"Added {train_bert.shape[1]} BERT features")
```

**Expected:** +0.15-0.30 RMSE improvement (if lyrics are important!)

**Warning:**
- VERY SLOW (1-2 hours)
- Needs GPU (use Colab Pro or free GPU)
- High risk (might not improve much)

---

## 📊 Strategi Testing Sistematis

### Spreadsheet Tracking

Buat Excel/Google Sheets:

| Experiment | Description | OOF RMSE | CV Std | LB Score | Time | Notes |
|------------|-------------|----------|--------|----------|------|-------|
| Baseline | Siklus 5 original | 15.82 | 0.034 | - | 20m | - |
| Exp 1.1a | lr=0.005, n_est=3000 | 15.76 | 0.032 | - | 35m | ✅ Small improve |
| Exp 1.1b | num_leaves=63 | 15.88 | 0.041 | - | 25m | ❌ Worse, overfitting |
| Exp 1.2 | Remove 10 low features | 15.79 | 0.033 | - | 18m | ~ Neutral |
| Exp 1.3 | Fold averaging | 15.73 | 0.032 | 15.68 | 20m | ✅✅ Good! |
| ... | ... | ... | ... | ... | ... | ... |

**Tips:**
- Test 1 thing at a time
- Keep baseline sebagai reference
- Track everything (even failures!)
- LB Score = leaderboard score (after submit)

---

## 🎯 Recommended Path untuk Mahasiswa

### Week 1: Quick Wins
- **Day 1-2:** Hyperparameter tuning manual (Exp 1.1)
- **Day 3:** Feature selection (Exp 1.2)
- **Day 4:** Fold averaging (Exp 1.3)
- **Day 5:** Submit best, analyze LB feedback

**Expected Progress:** Baseline 15.82 → 15.65-15.70

### Week 2: Medium Effort
- **Day 6-7:** Artist-genre interaction (Exp 2.1)
- **Day 8-9:** Temporal per genre (Exp 2.2)
- **Day 10:** Lyrics sentiment (Exp 2.3)
- **Day 11:** Rare artist smoothing (Exp 2.4)
- **Day 12:** Combine best features, submit

**Expected Progress:** 15.70 → 15.45-15.55

### Week 3: Advanced (If needed)
- **Day 13-15:** Optuna tuning (overnight runs)
- **Day 16-17:** Multi-model ensemble
- **Day 18:** BERT embeddings (if desperate)
- **Day 19-20:** Final ensemble, submit

**Expected Progress:** 15.55 → 15.20-15.35

---

## 💡 Pro Tips untuk Mahasiswa

### 1. Time Management
```
⏰ Prioritas:
1. Quick wins first (low-hanging fruit)
2. Medium effort jika masih banyak waktu
3. Advanced HANYA jika kompetisi deadline masih lama (>2 minggu)
```

### 2. Compute Resources
```
💻 Gunakan:
- Google Colab FREE untuk experiment
- Colab PRO ($10/month) jika butuh GPU untuk BERT
- Kaggle Kernels (30h/week free GPU)

🚫 Jangan:
- Run di laptop pribadi (makan battery, lambat)
- Run semua experiment sekaligus (waste resources)
```

### 3. Validation Strategy
```
✅ Always:
- Trust CV score lebih dari LB score (LB bisa overfitting)
- Check CV std (< 0.03 is good, < 0.02 is excellent)
- Compare OOF vs LB score (should be close)

❌ Avoid:
- Tuning based on LB feedback aja (overfitting!)
- Submit terlalu sering (limited submissions)
```

### 4. Collaboration (Jika Boleh)
```
👥 Jika kompetisi allow teams:
- Bagi tugas: A fokus feature engineering, B fokus tuning
- Share hasil experiment di spreadsheet bersama
- Ensemble model dari anggota team berbeda
```

---

## 🎓 Learning Path

### Pemula (First Competition)
**Focus:** Priority 1 only
- Pahami baseline dulu
- Coba quick wins
- Track experiment systematically
- **Goal:** Top 30-50%

### Intermediate (2-3 Competitions)
**Focus:** Priority 1 + Priority 2
- Feature engineering lebih aggressive
- Manual tuning lebih detail
- Ensemble sederhana
- **Goal:** Top 10-20%

### Advanced (5+ Competitions)
**Focus:** All priorities
- Optuna tuning
- Multi-model ensemble
- Advanced NLP (BERT, etc)
- **Goal:** Top 5%

---

## 📚 Resources untuk Belajar

### Feature Engineering
- [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [Feature Engineering for Machine Learning (Book)](https://www.amazon.com/Feature-Engineering-Machine-Learning-Principles/dp/1491953241)

### Hyperparameter Tuning
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hyperparameter Tuning Guide](https://www.kaggle.com/code/prashant111/a-guide-on-hyperparameter-tuning)

### Ensemble Methods
- [Ensemble Learning Guide](https://www.kaggle.com/code/arthurtok/introduction-to-ensembling-stacking-in-python)
- [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)

---

## ✅ Checklist Sebelum Submit Final

```
Final Submission Checklist:
□ OOF RMSE < LB target (jika tau target)
□ CV std < 0.03 (model stable)
□ Tested on local CV (no data leakage)
□ Predictions in valid range [0, 100]
□ No NaN in predictions
□ Track_id alignment correct (very important!)
□ Saved best model & code (reproducible)
□ Documented what works & what doesn't
```

---

## 🎉 Penutup

**Remember:**
1. **Start simple** - Quick wins dulu baru advanced
2. **Track everything** - Failed experiments = learning
3. **Trust CV > LB** - Don't overfit to leaderboard
4. **Time management** - Deadline > perfect score
5. **Have fun!** - ML competition = learning opportunity

**Good luck! 🚀**

---

**Last Updated:** 2025-11-07
**Author:** Claude (AI Assistant)
**For:** Machine Learning Competition Strategy
