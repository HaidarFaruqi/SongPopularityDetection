# 📊 COMPREHENSIVE REVIEW: FEATURES & DATA PROCESSING
## Song Popularity Detection Model

Tanggal Review: 2025-11-07

---

## 🎯 EXECUTIVE SUMMARY

**Status Keseluruhan:** ⚠️ GOOD dengan beberapa area yang perlu improvement

**Poin Utama:**
- ✅ Feature engineering sudah cukup komprehensif
- ⚠️ Ada potensi **DATA LEAKAGE** yang harus diperbaiki
- ⚠️ Beberapa preprocessing steps perlu diperbaiki
- 💡 Ada peluang untuk feature tambahan yang powerful

---

## 🔍 ANALISIS DETAIL

### 1. REVIEW FEATURES (Kategori & Kualitas)

#### ✅ **FEATURES YANG SUDAH BAIK:**

##### A. Artist Features
```
- artist_avg_pop       ✅ Target encoding - powerful feature
- artist_song_count    ✅ Artist popularity indicator
```
**Status:** GOOD
**Catatan:** Implementasi sudah benar dengan fallback ke global mean

##### B. Audio Features (Raw)
```
- energy, danceability, valence, loudness, tempo
- acousticness, speechiness, instrumentalness, liveness
- key, mode, time_signature
```
**Status:** GOOD - Semua fitur audio Spotify API standard

##### C. Audio Features (Engineered)
```
- energy_x_dance       ✅ Interaction feature
- duration_min         ✅ Better scale daripada ms
- key_mode            ✅ Kombinasi categorical
- tempo_category      ✅ Binning continuous variable
```
**Status:** GOOD

##### D. Temporal Features
```
- years_since_release  ✅ Relevance decay
- decade              ✅ Era grouping
- is_classic          ✅ Binary flag
- is_recent_hit       ✅ Binary flag
```
**Status:** GOOD

##### E. Track Name Features
```
- track_name_length        ✅ Title complexity
- track_name_word_count    ✅ Title structure
```
**Status:** ACCEPTABLE - Basic tapi helpful

##### F. Interaction Features
```
- artist_x_dance      ✅ Artist + audio interaction
- artist_x_energy     ✅ Artist + audio interaction
```
**Status:** GOOD

##### G. Lyrics Features (NLP)
```
- lyrics_feature_0 to lyrics_feature_19  ✅ TF-IDF + SVD
```
**Status:** GOOD - Explained variance reported

---

### 2. ⚠️ POTENSI MASALAH KRITIS

#### 🚨 **MASALAH #1: POTENTIAL DATA LEAKAGE (CRITICAL)**

**Lokasi:** `engineer_features()` method

**Masalah:**
```python
# CURRENT CODE (BERPOTENSI LEAK):
artist_popularity_map = self.train_df.groupby('artists')['popularity'].mean()

# Mapping ke test menggunakan statistik dari FULL training data
self.test_df['artist_avg_pop'] = self.test_df['artists'].map(artist_popularity_map)
```

**Mengapa ini masalah?**
- ❌ Saat cross-validation, kita menggunakan artist_avg_pop yang dihitung dari **FULL training set**
- ❌ Ini termasuk data dari validation fold yang seharusnya "unseen"
- ❌ Model bisa "melihat" target information dari validation set secara tidak langsung

**SOLUSI:**
```python
# CORRECT WAY: Hitung artist encoding INSIDE CV loop
# Atau gunakan target encoding yang proper dengan regularization

from category_encoders import TargetEncoder

# Inside train_models atau prepare_features
target_encoder = TargetEncoder(cols=['artists'], smoothing=1.0)

# Fit hanya pada training fold, transform pada validation fold
# Ini harus dilakukan INSIDE CV loop
```

**Impact:** Bisa menyebabkan overfitting dan CV score yang terlalu optimistic

**Priority:** 🔴 HIGH - Harus diperbaiki

---

#### ⚠️ **MASALAH #2: IMPUTATION STRATEGY**

**Lokasi:** `prepare_features()` method

**Masalah:**
```python
self.imputer = SimpleImputer(strategy='median')
```

**Issue:**
- ⚠️ Median imputation untuk SEMUA features (termasuk categorical yang sudah diencode)
- ⚠️ Tidak ada handling khusus untuk different feature types

**Analisis:**
- Median untuk numerical features: ✅ OK
- Median untuk categorical encoded: ⚠️ QUESTIONABLE
  - Contoh: `genre_encoded` [0,1,2,3,4] → median mungkin 2.0
  - Apakah nilai 2.0 (bukan integer) masuk akal untuk categorical?

**SOLUSI:**
```python
# Better approach:
from sklearn.compose import ColumnTransformer

numeric_features = [f for f in self.features if f not in self.categorical_features_encoded]

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_features),
    ('cat', SimpleImputer(strategy='most_frequent'), self.categorical_features_encoded)
])
```

**Priority:** 🟡 MEDIUM

---

#### ⚠️ **MASALAH #3: MISSING STANDARD SCALING**

**Lokasi:** Model training

**Masalah:**
- ❌ Tidak ada standard scaling untuk numerical features
- LightGBM tree-based model memang tidak butuh scaling
- TAPI jika nanti ingin ensemble dengan linear models (Ridge, Lasso), akan bermasalah

**Catatan:**
- Untuk LightGBM only: ✅ OK (tidak perlu scaling)
- Untuk future ensemble: ⚠️ Perlu scaling

**Priority:** 🟢 LOW (OK untuk sekarang)

---

#### ⚠️ **MASALAH #4: LYRICS PROCESSING**

**Lokasi:** `process_lyrics()` method

**Potensi Issue:**
```python
tfidf = TfidfVectorizer(
    max_features=500,
    min_df=5,              # ← Mungkin terlalu strict untuk dataset kecil
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'   # ← Asumsi semua lyrics bahasa Inggris
)
```

**Pertanyaan:**
- Apakah ada lyrics dalam bahasa lain? (Spanish, Korean, dll)
- Jika min_df=5, berapa banyak kata yang di-filter out?

**SOLUSI:**
```python
# Check distribusi bahasa dulu
# Adjust min_df berdasarkan dataset size
# Pertimbangkan multilingual stop words
```

**Priority:** 🟡 MEDIUM

---

### 3. 📉 FEATURES YANG MISSING / BISA DITAMBAHKAN

#### 🔴 **HIGH PRIORITY - Features yang Powerful:**

1. **Genre Statistics**
   ```python
   # Genre-level aggregates
   genre_avg_pop = train.groupby('track_genre')['popularity'].mean()
   genre_std_pop = train.groupby('track_genre')['popularity'].std()
   genre_song_count = train.groupby('track_genre').size()
   ```
   **Why:** Genre adalah strong predictor, statistik genre bisa sangat informatif

2. **Artist Variance/Std**
   ```python
   artist_std_pop = train.groupby('artists')['popularity'].std()
   artist_min_pop = train.groupby('artists')['popularity'].min()
   artist_max_pop = train.groupby('artists')['popularity'].max()
   ```
   **Why:** Consistency indicator - apakah artist consistently popular?

3. **Release Year Trend**
   ```python
   # Momentum: rata-rata popularity untuk songs dari same year
   year_avg_pop = train.groupby('release_year')['popularity'].mean()
   ```
   **Why:** Beberapa tahun mungkin punya trend tertentu

4. **Audio Feature Ratios**
   ```python
   # Ratios yang meaningful
   'energy_to_valence_ratio': energy / (valence + 1e-6)
   'acoustic_to_energy_ratio': acousticness / (energy + 1e-6)
   ```
   **Why:** Relative measures bisa lebih informatif

#### 🟡 **MEDIUM PRIORITY - Nice to Have:**

5. **Track Name Features (Advanced)**
   ```python
   # Has featuring artist?
   has_feat = track_name.str.contains('feat|ft\.|featuring', case=False)

   # Has remix/remaster indicators
   is_remix = track_name.str.contains('remix|remaster|version', case=False)

   # Title sentiment (if possible)
   title_sentiment_score
   ```

6. **Duration Bins**
   ```python
   duration_category = pd.cut(duration_ms, bins=[0, 180000, 240000, 300000, np.inf],
                              labels=['short', 'medium', 'long', 'very_long'])
   ```

7. **Audio Feature Aggregates**
   ```python
   # Average of all audio features
   audio_features_mean = ['energy', 'danceability', ...].mean()
   audio_features_std = ['energy', 'danceability', ...].std()
   ```

#### 🟢 **LOW PRIORITY - Experimental:**

8. **Lyrics Advanced Features**
   ```python
   # Lyrics length
   lyrics_word_count = lyrics.str.split().str.len()

   # Lyrics complexity (unique words / total words)
   lyrics_complexity

   # Sentiment score
   lyrics_sentiment
   ```

9. **Time-based Features**
   ```python
   # Quarter of release
   release_quarter = pd.to_datetime(release_date).dt.quarter

   # Season
   release_season
   ```

---

### 4. 🔧 PREPROCESSING ISSUES

#### Issue #1: Label Encoding untuk Categorical
```python
# CURRENT:
le = LabelEncoder()
df['genre_encoded'] = le.transform(df['genre'])
```

**Problem:**
- LabelEncoder assigns arbitrary integers: rock=0, pop=1, jazz=2
- Ini implies ordinal relationship yang tidak ada
- LightGBM akan handle ini OK, tapi tidak optimal

**BETTER:**
```python
# LightGBM native categorical support
df['track_genre'] = df['track_genre'].astype('category')
# Lalu pass categorical_feature parameter ke LightGBM
```

**Priority:** 🟡 MEDIUM

#### Issue #2: Test Set Encoding
```python
# CURRENT CODE:
combined_series = pd.concat([train_df[col], test_df[col]])
le.fit(combined_series)
```

**Status:** ✅ CORRECT
Ini sudah benar - fit on combined untuk avoid unknown categories di test

---

### 5. 🤖 MODEL CONFIGURATION REVIEW

#### Current Config:
```python
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

**Analysis:**
- ✅ n_estimators=1000: OK, tapi perlu early stopping
- ✅ learning_rate=0.01: Conservative, good
- ⚠️ num_leaves=31 + max_depth=6: Bisa konfliktual
  - num_leaves=31 implies depth ~5
  - max_depth=6 explicitly limits it
- ❌ **MISSING**: early_stopping_rounds
- ❌ **MISSING**: categorical_feature parameter
- ❌ **MISSING**: reg_alpha, reg_lambda (regularization)

**RECOMMENDED:**
```python
LGBMRegressor(
    n_estimators=2000,           # Increase karena ada early stopping
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,                # Let num_leaves control depth
    min_child_samples=20,        # Prevent overfitting
    subsample=0.8,               # Bagging
    colsample_bytree=0.8,        # Feature sampling
    reg_alpha=0.1,               # L1 regularization
    reg_lambda=0.1,              # L2 regularization
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Add early stopping in fit:
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=100,
          verbose=False)
```

---

### 6. 📊 CROSS-VALIDATION STRATEGY

#### Current:
```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

**Status:** ✅ ACCEPTABLE

**Considerations:**
- Untuk time-series data dengan release_year: consider **TimeSeriesSplit**
  - Train on older songs, validate on newer songs
- Untuk imbalanced target: consider **StratifiedKFold** (bin popularity first)

**Current approach OK jika:**
- Data cukup random distributed
- Tidak ada temporal dependency yang strong

---

## 🎯 REKOMENDASI PRIORITAS

### 🔴 **URGENT (Harus Diperbaiki):**

1. **Fix Target Encoding Data Leakage**
   - Implement proper target encoding with CV-aware approach
   - Atau gunakan library seperti `category_encoders`
   - **Impact:** Bisa improve CV reliability significantly

2. **Add Early Stopping to LightGBM**
   - Prevent overfitting
   - Optimize training time
   - **Impact:** Better generalization

### 🟡 **HIGH PRIORITY (Sangat Direkomendasikan):**

3. **Add Genre-level Features**
   - Genre mean, std, count
   - **Impact:** Likely +0.1-0.2 RMSE improvement

4. **Add Artist Variance Features**
   - Artist std, min, max popularity
   - **Impact:** Better artist profiling

5. **Improve Model Hyperparameters**
   - Add regularization
   - Add subsample, colsample
   - **Impact:** Better generalization

### 🟢 **MEDIUM PRIORITY (Nice to Have):**

6. **Fix Imputation Strategy**
   - Different strategies for numeric vs categorical
   - **Impact:** Marginal improvement

7. **Add Advanced Track Name Features**
   - has_feat, is_remix, etc.
   - **Impact:** Small but helpful

8. **Add Audio Feature Ratios**
   - energy/valence, acoustic/energy ratios
   - **Impact:** Could capture interactions better

---

## 📈 EXPECTED IMPACT

Jika semua rekomendasi HIGH PRIORITY diimplementasi:

**Current RMSE:** ~16.0-16.5
**Expected RMSE:** ~15.5-16.0 (improvement 0.3-0.5)

**Breakdown:**
- Fix data leakage: ±0.0 to -0.1 (CV jadi lebih realistic, score bisa naik atau turun)
- Genre features: -0.1 to -0.2
- Artist variance: -0.05 to -0.1
- Better hyperparameters: -0.05 to -0.1
- Other features: -0.05 to -0.1

---

## ✅ CHECKLIST UNTUK IMPROVEMENT

### Data Leakage
- [ ] Implement proper target encoding (TargetEncoder with CV)
- [ ] Verify no other leakage sources

### Feature Engineering
- [ ] Add genre statistics (mean, std, count)
- [ ] Add artist variance features
- [ ] Add audio feature ratios
- [ ] Add advanced track name features

### Model Configuration
- [ ] Add early_stopping_rounds
- [ ] Add regularization (reg_alpha, reg_lambda)
- [ ] Add subsample and colsample_bytree
- [ ] Experiment with num_leaves

### Preprocessing
- [ ] Separate imputation for numeric vs categorical
- [ ] Consider different CV strategies (TimeSeriesSplit?)

### Validation
- [ ] Verify CV scores are realistic
- [ ] Check for overfitting (train vs val gap)
- [ ] Analyze feature importance changes

---

## 💡 ADVANCED IDEAS (OPTIONAL)

1. **Two-Stage Modeling**
   - Stage 1: Predict popularity range (low/mid/high)
   - Stage 2: Regression within each range
   - Different models for different ranges

2. **Ensemble Approach**
   - LightGBM + XGBoost + CatBoost
   - Ridge/Lasso for linear baseline
   - Stacking ensemble

3. **Feature Selection**
   - Remove low-importance features
   - Reduce collinearity
   - Potentially improve generalization

4. **Hyperparameter Tuning**
   - Optuna/GridSearch for optimal hyperparameters
   - Could squeeze extra 0.05-0.1 RMSE

---

## 🎓 KESIMPULAN

**Overall Assessment:** Code sudah SOLID tapi ada beberapa critical issues yang perlu diperbaiki.

**Kekuatan:**
- ✅ Feature engineering sudah comprehensive
- ✅ Code structure bagus dan well-documented
- ✅ Pipeline complete dari data → model → submission

**Kelemahan:**
- ⚠️ Data leakage di target encoding
- ⚠️ Model configuration belum optimal
- ⚠️ Missing beberapa powerful features

**Next Step:**
1. Fix data leakage (CRITICAL)
2. Add genre & artist variance features
3. Improve model hyperparameters
4. Re-run dan compare results

---

**Prepared by:** Claude AI
**Review Date:** 2025-11-07
**Model Version:** Siklus 4 Enhanced
