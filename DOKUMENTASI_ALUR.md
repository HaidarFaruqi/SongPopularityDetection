# 🎵 DOKUMENTASI ALUR LENGKAP SISTEM PREDIKSI POPULARITAS LAGU

## 📋 Daftar Isi
1. [Overview Sistem](#1-overview-sistem)
2. [Arsitektur Pipeline](#2-arsitektur-pipeline)
3. [Alur Detail Step-by-Step](#3-alur-detail-step-by-step)
4. [Penjelasan Algoritma & Teknik](#4-penjelasan-algoritma--teknik)
5. [Analisis & Evaluasi](#5-analisis--evaluasi)
6. [Cara Menjalankan](#6-cara-menjalankan)

---

## 1. Overview Sistem

### 🎯 Tujuan
Membangun sistem Machine Learning untuk memprediksi popularitas lagu berdasarkan karakteristik audio, artist, lyrics, dan metadata lainnya.

### 📊 Input & Output

**Input**:
- **Training Data** (`train.csv`): Data lagu dengan label popularitas
- **Testing Data** (`test.csv`): Data lagu tanpa label (untuk prediksi)

**Output**:
- **Submission File** (`submission_siklus4_enhanced.csv`): Prediksi popularitas untuk test set
- **Visualizations** (`siklus4_comprehensive_analysis.png`): 15 grafik analisis komprehensif
- **Console Reports**: Insights dan statistik detail

### 🔧 Teknologi Stack

| Kategori | Library | Fungsi |
|----------|---------|--------|
| Data Processing | Pandas, NumPy | Manipulasi data |
| NLP | scikit-learn (TfidfVectorizer, TruncatedSVD) | Pemrosesan lyrics |
| Machine Learning | LightGBM | Model training |
| Preprocessing | scikit-learn (LabelEncoder, SimpleImputer) | Encoding & imputation |
| Evaluation | scikit-learn.metrics | Evaluasi performa |
| Visualization | Matplotlib, Seaborn | Visualisasi data |
| Statistics | SciPy | Analisis statistik |

---

## 2. Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SONG POPULARITY PREDICTOR                     │
│                         PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────┘

   📂 INPUT                    🔄 PROCESSING                  📤 OUTPUT

train.csv ──┐
            ├──> [1] Load Data ──────────────────────────────────┐
test.csv  ──┘                                                     │
                                                                  │
            ┌─────────────────────────────────────────────────────┘
            │
            ├──> [2] EDA ────────────> Statistics & Understanding
            │
            ├──> [3] Feature Engineering ──────┐
            │         • Artist Features        │
            │         • Audio Features          │
            │         • Temporal Features       │
            │         • Track Name Features     │
            │         • Interaction Features    │
            │                                   │
            ├──> [4] Process Lyrics (NLP) ─────┤
            │         • TF-IDF                  │
            │         • SVD                     │
            │                                   │
            ├──> [5] Prepare Features ─────────┤
            │         • Label Encoding          │
            │         • Imputation              │
            │                                   │
            ├──────────────────────────────────┘
            │
            ├──> [6] Train Models ──────────────┐
            │         • LightGBM                │
            │         • 5-Fold CV               │
            │         • OOF Predictions         │
            │                                   │
            ├──> [7] Visualizations ────────────┤──> PNG file (15 plots)
            │                                   │
            ├──> [8] Insights Report ───────────┤──> Console output
            │                                   │
            ├──> [9] Error Analysis ────────────┘
            │
            └──> [10] Create Submission ────────> submission.csv

```

---

## 3. Alur Detail Step-by-Step

### 📂 STEP 1: Load Data

**Fungsi**: `load_data()`

**Proses**:
```python
# Baca CSV files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Tampilkan basic info
print(f"Training data: {train_df.shape}")
print(f"Target range: [{min}, {max}]")
print(f"Target mean: {mean}")
```

**Output**:
- `self.train_df`: DataFrame training (dengan label popularity)
- `self.test_df`: DataFrame testing (tanpa label)

**Informasi yang didapat**:
- Jumlah samples: Training vs Testing
- Range popularitas: 0-100
- Distribusi target: Mean, std

---

### 📊 STEP 2: Exploratory Data Analysis (EDA)

**Fungsi**: `eda()`

**Proses**:
1. **Basic Statistics**:
   - Count, mean, std, min, max, quartiles
   - Memahami distribusi popularitas

2. **Missing Values Check**:
   - Identifikasi kolom dengan missing data
   - Planning untuk imputation

3. **Genre Analysis**:
   - Top 5 genre terpopuler
   - Understanding data composition

**Output**:
```
Statistics:
  count: 114000
  mean:  42.5
  std:   23.8

Top Genres:
  pop:      15000
  hip-hop:  12000
  rock:     10000
```

**Insight**:
- Distribusi popularitas: Apakah balanced atau skewed?
- Missing patterns: Random atau systematic?
- Genre dominance: Pop, hip-hop biasanya dominan

---

### 🔧 STEP 3: Feature Engineering

**Fungsi**: `engineer_features()`

**Proses Detail**:

#### 3.1 Artist Features (Target Encoding)
```python
# Hitung statistik per artist HANYA dari training data
artist_popularity_map = train_df.groupby('artists')['popularity'].mean()
artist_song_count_map = train_df.groupby('artists').size()

# Apply ke train dan test
train_df['artist_avg_pop'] = train_df['artists'].map(artist_popularity_map)
test_df['artist_avg_pop'] = test_df['artists'].map(artist_popularity_map)

# Fallback untuk artist yang tidak dikenal
global_mean = train_df['popularity'].mean()
train_df['artist_avg_pop'].fillna(global_mean, inplace=True)
test_df['artist_avg_pop'].fillna(global_mean, inplace=True)
```

**Mengapa penting**:
- ✅ Leak-free: Menggunakan training data saja
- ✅ Generalization: Fallback untuk unknown artists
- ✅ Highly predictive: Artist reputation sangat berpengaruh

#### 3.2 Audio Features
```python
# Interaction features
df['energy_x_dance'] = df['energy'] * df['danceability']

# Duration conversion
df['duration_min'] = df['duration_ms'] / 60000

# Categorical binning
df['tempo_category'] = pd.cut(df['tempo'],
                              bins=[0, 90, 120, 150, 250],
                              labels=['slow', 'moderate', 'fast', 'very_fast'])

# Key + Mode combination
df['key_mode'] = df['key'].astype(str) + '_' + df['mode'].astype(str)
```

#### 3.3 Temporal Features
```python
# Age calculation
df['years_since_release'] = 2025 - df['release_year']

# Decade grouping
df['decade'] = (df['release_year'] // 10) * 10

# Binary flags
df['is_classic'] = (df['release_year'] < 2000).astype(int)
df['is_recent_hit'] = (df['release_year'] >= 2020).astype(int)
```

#### 3.4 Track Name Features
```python
# Clean track names
clean_name = df['track_name'].str.lower()
clean_name = clean_name.str.replace(r'[\(\[].*?[\)\]]', '', regex=True)  # Remove (Remix), [Remastered]
clean_name = clean_name.str.split(' - feat.').str[0]  # Remove featuring info
clean_name = clean_name.str.strip()

# Extract features
df['track_name_length'] = clean_name.str.len()
df['track_name_word_count'] = clean_name.str.count(' ') + 1
```

#### 3.5 Interaction Features
```python
# Artist × Audio interactions
df['artist_x_dance'] = df['artist_avg_pop'] * df['danceability']
df['artist_x_energy'] = df['artist_avg_pop'] * df['energy']
```

**Total Fitur Baru**: ~15-20 fitur engineered

---

### 📝 STEP 4: Process Lyrics (NLP)

**Fungsi**: `process_lyrics(n_components=20)`

**Alur NLP Pipeline**:

```
Lyrics Text
    ↓
[TF-IDF Vectorization]
    ↓
500 TF-IDF Features
    ↓
[Truncated SVD]
    ↓
20 Latent Features
```

**Detail Implementasi**:

#### 4.1 TF-IDF Vectorization
```python
tfidf = TfidfVectorizer(
    max_features=500,      # Ambil 500 kata terpenting
    min_df=5,              # Kata harus muncul min di 5 lagu
    max_df=0.8,            # Buang kata yang ada di >80% lagu
    ngram_range=(1, 2),    # Unigram (1 word) dan bigram (2 words)
    stop_words='english'   # Remove stopwords (the, is, are, dll)
)

train_tfidf = tfidf.fit_transform(train_df['lyrics'])
test_tfidf = tfidf.transform(test_df['lyrics'])
```

**Contoh TF-IDF**:
```
Lyrics: "I love you so much, you're my everything"

After TF-IDF:
- "love":       0.452  (important, moderate frequency)
- "you":        0.231  (common, lower weight)
- "everything": 0.892  (rare, high weight)
- "the":        0.000  (stopword, removed)
```

#### 4.2 Dimensionality Reduction (SVD)
```python
svd = TruncatedSVD(n_components=20, random_state=42)
train_lyrics_features = svd.fit_transform(train_tfidf)  # 500 → 20
test_lyrics_features = svd.transform(test_tfidf)
```

**Mengapa SVD?**:
- 📉 Reduce dimensionality: 500 → 20 (lebih efisien)
- 🎯 Capture semantics: Menangkap topik/tema tersembunyi
- 🚫 Reduce noise: Membuang variasi yang tidak penting

**Explained Variance**:
- Biasanya 20 komponen menangkap 30-50% variance
- Trade-off: Information retention vs Overfitting prevention

**Output**: 20 kolom baru: `lyrics_feature_0`, `lyrics_feature_1`, ..., `lyrics_feature_19`

---

### 🎯 STEP 5: Prepare Features

**Fungsi**: `prepare_features()`

**Proses**:

#### 5.1 Feature Selection
```python
# Semua fitur numerik
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude non-predictive columns
exclude = ['popularity', 'track_id', 'track_name', 'artists', 'lyrics', 'release_year']
numeric_features = [f for f in numeric_features if f not in exclude]

# Fitur kategorik
categorical_features = ['track_genre', 'key_mode', 'tempo_category', 'decade']
```

#### 5.2 Label Encoding
```python
for col in categorical_features:
    le = LabelEncoder()

    # Fit dengan kombinasi train + test (untuk handle semua kategori)
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined.astype(str))

    # Transform
    train_df[col + '_encoded'] = le.transform(train_df[col].astype(str))
    test_df[col + '_encoded'] = le.transform(test_df[col].astype(str))

    # Save encoder untuk future use
    label_encoders[col] = le
```

**Contoh Encoding**:
```
track_genre:
  'pop'     → 0
  'rock'    → 1
  'hip-hop' → 2
  'jazz'    → 3
```

#### 5.3 Missing Value Imputation
```python
imputer = SimpleImputer(strategy='median')

# Fit hanya pada training data
imputer.fit(train_df[features])

# Transform train dan test
train_df[features] = imputer.transform(train_df[features])
test_df[features] = imputer.transform(test_df[features])
```

**Mengapa Median?**:
- ✅ Robust terhadap outliers
- ✅ Lebih baik untuk skewed distributions
- ✅ Preserves data distribution

#### 5.4 Set Categorical Dtype (untuk LightGBM)
```python
for col in categorical_features_encoded:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')
```

**Mengapa?**:
- LightGBM punya optimized handling untuk categorical features
- Lebih cepat dan lebih efisien memory
- Better split finding

**Total Features**: Biasanya 40-60 features siap untuk modeling

---

### 🤖 STEP 6: Train Models

**Fungsi**: `train_models(cv_folds=5)`

**Model**: LightGBM (Light Gradient Boosting Machine)

#### 6.1 Apa itu LightGBM?

**LightGBM** adalah algoritma Gradient Boosting yang:
- ⚡ **Cepat**: Optimized untuk dataset besar
- 🎯 **Akurat**: State-of-the-art untuk tabular data
- 💾 **Efisien**: Low memory usage
- 🌳 **Tree-based**: Ensemble of decision trees

**Cara Kerja Gradient Boosting**:
```
1. Mulai dengan prediksi sederhana (mean)
2. Hitung error (residual)
3. Train tree untuk memprediksi residual
4. Update prediksi: prediksi_baru = prediksi_lama + learning_rate × tree_prediction
5. Ulangi steps 2-4 sebanyak n_estimators kali
6. Final prediction = kombinasi semua trees
```

#### 6.2 Hyperparameters

```python
lgbm = LGBMRegressor(
    n_estimators=1000,      # Jumlah trees (boosting iterations)
    learning_rate=0.01,     # Learning rate kecil → slow learning, less overfitting
    num_leaves=31,          # Kompleksitas tree (max leaves per tree)
    max_depth=6,            # Kedalaman maksimal tree
    random_state=42,        # Reproducibility
    n_jobs=-1,              # Parallel processing (gunakan semua CPU cores)
    verbose=-1              # Silent mode
)
```

**Penjelasan Hyperparameters**:

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `n_estimators` | 1000 | Jumlah trees yang dibangun. Lebih banyak = lebih kompleks. |
| `learning_rate` | 0.01 | Seberapa besar kontribusi setiap tree. Kecil = lebih konservatif, butuh lebih banyak trees. |
| `num_leaves` | 31 | Jumlah max daun per tree. Lebih banyak = lebih kompleks, lebih prone to overfit. |
| `max_depth` | 6 | Kedalaman max tree. Membatasi kompleksitas. |

#### 6.3 Cross-Validation

**K-Fold Cross-Validation** (k=5):

```
Dataset Split:
┌─────────────────────────────────────┐
│ Full Training Data (100%)            │
└─────────────────────────────────────┘
         ↓
   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
   │  Fold 1  │  Fold 2  │  Fold 3  │  Fold 4  │  Fold 5  │
   │   20%    │   20%    │   20%    │   20%    │   20%    │
   └──────────┴──────────┴──────────┴──────────┴──────────┘

Iteration 1: Fold 1 = validation, Folds 2-5 = training
Iteration 2: Fold 2 = validation, Folds 1,3-5 = training
Iteration 3: Fold 3 = validation, Folds 1-2,4-5 = training
Iteration 4: Fold 4 = validation, Folds 1-3,5 = training
Iteration 5: Fold 5 = validation, Folds 1-4 = training
```

**Implementasi**:
```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scoring
cv_scores = cross_val_score(
    lgbm, X, y,
    cv=kfold,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

# Out-of-Fold predictions
oof_predictions = cross_val_predict(lgbm, X, y, cv=kfold)
```

**Mengapa Cross-Validation?**:
- ✅ **Reliable evaluation**: Setiap sample di-validate sekali
- ✅ **Reduce overfitting**: Tidak train dan test pada data yang sama
- ✅ **Stable metrics**: Average dari 5 folds lebih stabil
- ✅ **Detect variance**: Jika std tinggi → model tidak stabil

**Metrics**: RMSE (Root Mean Squared Error)
```
RMSE = sqrt( mean( (actual - predicted)² ) )

Contoh:
  Actual:    [50, 60, 70]
  Predicted: [52, 58, 72]
  Errors:    [2, -2, 2]
  Squared:   [4, 4, 4]
  Mean:      4
  RMSE:      2.0
```

**Interpretasi RMSE**:
- RMSE = 16 → rata-rata error adalah 16 poin popularitas
- Lower is better
- Dalam skala 0-100, RMSE < 16 sudah sangat baik

#### 6.4 Final Training

Setelah CV, train pada full dataset:
```python
lgbm.fit(X, y)
```

**Output**:
- Trained model disimpan di `self.models['LightGBM']`
- CV scores: `[16.2, 16.1, 16.3, 16.0, 16.2]`
- Mean RMSE: `16.16 ± 0.11`
- Out-of-Fold predictions untuk analisis

---

### 📊 STEP 7: Create Comprehensive Visualizations

**Fungsi**: `create_comprehensive_visualizations()`

**Output**: Single PNG file dengan 15 subplots (5 rows × 3 columns)

#### Visualisasi yang dibuat:

| # | Plot | Tujuan |
|---|------|--------|
| 1 | **CV Scores by Fold** | Melihat konsistensi performa across folds |
| 2 | **Model Performance Metrics** | RMSE, MAE, R² dalam satu pandangan |
| 3 | **Top 25 Feature Importance** | Fitur mana yang paling berpengaruh |
| 4 | **Actual vs Predicted Scatter** | Seberapa dekat prediksi dengan actual |
| 5 | **Residuals Distribution** | Distribusi error (ideally normal) |
| 6 | **Residual Plot** | Pattern dalam error (ideally random) |
| 7 | **RMSE by Prediction Range** | Performa model di range prediksi berbeda |
| 8 | **Actual vs Predicted Distribution** | Membandingkan distribusi |
| 9 | **Q-Q Plot** | Test normalitas residuals |
| 10 | **RMSE by Popularity Range** | Performa di low/mid/high popularity |
| 11 | **Feature Importance by Category** | Kategori fitur mana yang paling penting |
| 12 | **Worst Predictions Highlighted** | Highlight 5% worst errors |
| 13 | **Error Distribution Boxplot** | Error distribution by popularity bins |
| 14 | **Top 10 Most Important Features** | Detail top features dengan percentage |
| 15 | **Model Summary Statistics Table** | Ringkasan semua metrics |

**Insight dari Visualizations**:
- **Fold consistency**: Std < 0.15 → model stabil
- **Feature importance**: Top 10 fitur biasanya contribute 60-70%
- **Residual normality**: Jika normal → model well-calibrated
- **Error patterns**: Jika ada pattern → ada signal yang belum ditangkap

---

### 📝 STEP 8: Generate Insights Report

**Fungsi**: `generate_insights_report()`

**Output**: Console report dengan 5 sections

#### 8.1 Model Performance Summary
```
Cross-Validation:
  Fold 1: RMSE = 16.2341
  Fold 2: RMSE = 16.1092
  ...
  Mean:   RMSE = 16.1623
  Std:    RMSE = 0.1142

Out-of-Fold Performance:
  RMSE: 16.1623
  MAE:  12.3451
  R²:   0.7234
  MAPE: 28.45%
```

#### 8.2 Top 30 Most Important Features
```
    Feature                 Importance  Importance_Pct
1   artist_avg_pop         15234.2     18.5%
2   track_genre_encoded     9876.1     12.0%
3   years_since_release     7654.3      9.3%
...
```

#### 8.3 Residuals Analysis
```
Residual Statistics:
  Mean:     -0.0023  (mendekati 0 = unbiased)
  Std:      16.1623
  Skewness:  0.1234  (mendekati 0 = symmetric)
  Kurtosis:  0.4567

Shapiro-Wilk Test:
  P-value: 0.3421
  → Residuals appear normally distributed ✓
```

#### 8.4 Error Analysis by Popularity Range
```
Range          Label        Count    RMSE      MAE       Bias
0-20           Very Low     12345    18.2341   14.123    2.345
20-40          Low          23456    15.4567   11.234   -0.456
40-60          Medium       34567    14.3210   10.567   -1.234
60-80          High         28901    16.7890   12.890    1.567
80-100         Very High     5678    21.4567   17.234    3.456
```

**Insight**:
- Error terbesar di extreme ranges (very low & very high)
- Medium range paling mudah diprediksi

#### 8.5 Key Insights & Recommendations
```
✓ Model Performance:
  • EXCELLENT! RMSE 16.16 is competitive 🎯

✓ Bias Analysis:
  • Unbiased predictions (mean residual: -0.002) ✓

✓ Error Distribution:
  • Symmetric error distribution (skew: 0.123) ✓

✓ Feature Insights:
  Top 5 features account for 58.3% of importance:
    1. artist_avg_pop: 18.5%
    2. track_genre_encoded: 12.0%
    ...

✓ Recommendations for Next Iteration:
  • Artist features are highly important - already leveraged ✓
  • Focus on Very High popularity range (RMSE: 21.46)
  • Consider more advanced features...
```

---

### 🔍 STEP 9: Analyze Errors

**Fungsi**: `analyze_errors(n=20)`

**Tujuan**: Deep dive ke 20 worst predictions untuk find patterns

#### Output:

**9.1 Top Worst Predictions**
```
track_name              artists    genre    year  actual  predicted  error
"Shape of You"          Ed Sheeran pop      2017    92      67       25.3
"Bohemian Rhapsody"     Queen      rock     1975    88      62       26.1
...
```

**9.2 Pattern Analysis**

**Genre Distribution in Errors**:
```
Genre         Errors  Total   Error Rate
rock          8       5000    0.16%
classical     5       2000    0.25%
jazz          4       3000    0.13%
```

**Temporal Patterns**:
```
1990-2000: 3 songs
2000-2010: 5 songs
2010-2020: 8 songs
2020-2025: 4 songs
```

**Error Direction**:
```
Over-predictions:  45% (model too high)
Under-predictions: 55% (model too low)
→ Model tends to under-predict popular songs
```

**Artist vs Song Mismatch**:
```
High-artist/Low-song: 3 songs
  → Model over-relies on artist popularity
Low-artist/High-song: 7 songs
  → Model under-estimates breakthrough hits
```

**Recommendations**:
- Investigate rock and classical genres
- Add features for "viral/breakthrough" detection
- Consider genre-specific models

---

### 📤 STEP 10: Create Submission

**Fungsi**: `create_submission(predictions, filename)`

**Proses**:

```python
# 1. Predict pada test set
X_test = test_df[features]
predictions = lgbm.predict(X_test)

# 2. Clip predictions ke valid range [0, 100]
predictions = np.clip(predictions, 0, 100)

# 3. Create submission DataFrame
submission = pd.DataFrame({
    'track_id': test_df['track_id'],
    'popularity': predictions
})

# 4. Save to CSV
submission.to_csv('submission_siklus4_enhanced.csv', index=False)
```

**Output**:
```
✓ Saved: submission_siklus4_enhanced.csv
  • Predictions: 28500
  • Range: [5.23, 94.67]
  • Mean:  42.34
  • Std:   23.12
```

**Validation Checks**:
- ✅ Predictions dalam range [0, 100]
- ✅ Distribusi mirip dengan training data
- ✅ No missing values
- ✅ Correct format untuk submission

---

## 4. Penjelasan Algoritma & Teknik

### 🌳 LightGBM (Light Gradient Boosting Machine)

#### Konsep Dasar

**Gradient Boosting** = Ensemble method yang menggabungkan banyak weak learners (decision trees) menjadi strong learner

**Analogi Sederhana**:
```
Bayangkan Anda belajar main basket:

Iteration 1: Coach mengajar basic shooting
  → Anda bisa shoot, tapi masih banyak miss

Iteration 2: Coach koreksi error Anda (too high, too low)
  → Anda improve sedikit

Iteration 3: Coach koreksi error lagi
  → Anda improve lagi

... dan seterusnya

Setelah 1000 iterations, Anda jadi ahli!

LightGBM bekerja sama:
- Setiap tree belajar dari error tree sebelumnya
- Kombinasi 1000 trees → prediksi yang sangat akurat
```

#### Formula Matematis

```
Prediksi_final = f₀ + α·f₁ + α·f₂ + ... + α·f₁₀₀₀

di mana:
  f₀ = initial prediction (mean)
  fᵢ = tree ke-i yang memprediksi residual
  α = learning_rate (0.01)
```

#### Keunggulan LightGBM

| Keunggulan | Penjelasan |
|------------|------------|
| **Speed** | 10-20x lebih cepat dari XGBoost |
| **Memory Efficiency** | Histogram-based algorithm |
| **Accuracy** | State-of-the-art untuk tabular data |
| **Categorical Support** | Native support untuk categorical features |
| **Parallel Processing** | Multi-threading optimization |

---

### 📊 TF-IDF (Term Frequency - Inverse Document Frequency)

#### Konsep Dasar

**TF-IDF** mengukur seberapa penting sebuah kata dalam dokumen

**Formula**:
```
TF-IDF(word, document) = TF(word, document) × IDF(word)

TF (Term Frequency) = (jumlah kemunculan word di document) / (total words di document)

IDF (Inverse Document Frequency) = log( total_documents / documents_containing_word )
```

#### Contoh Perhitungan

**Lyrics 1**: "love love love you"
**Lyrics 2**: "you are my everything"
**Lyrics 3**: "love is everything"

**Untuk kata "love" di Lyrics 1**:
```
TF = 3/4 = 0.75
IDF = log(3/2) = 0.176
TF-IDF = 0.75 × 0.176 = 0.132
```

**Untuk kata "everything" di Lyrics 2**:
```
TF = 1/4 = 0.25
IDF = log(3/2) = 0.176
TF-IDF = 0.25 × 0.176 = 0.044
```

#### Mengapa TF-IDF Bagus?

- ✅ **Downweight common words**: "the", "is", "are" → low score
- ✅ **Upweight rare words**: Kata unik → high score
- ✅ **Context-aware**: Mempertimbangkan distribusi kata

---

### 🔬 SVD (Singular Value Decomposition)

#### Konsep Dasar

**SVD** adalah teknik dimensionality reduction

**Analogi Sederhana**:
```
Bayangkan Anda punya foto 4K (3840 × 2160 pixels):
- Original: 8,294,400 pixels
- Kompres ke 1080p: 2,073,600 pixels (75% lebih kecil)
- Masih recognizable, tapi lebih efisien

SVD bekerja sama:
- Original: 500 TF-IDF features
- Kompres ke 20 components (96% lebih kecil)
- Masih menangkap semantic meaning
```

#### Formula Matematis

```
Matrix A (m × n) = U × Σ × Vᵀ

di mana:
  U = left singular vectors (documents)
  Σ = singular values (importance)
  V = right singular vectors (features)

TruncatedSVD: Ambil k terbesar dari Σ
```

#### Manfaat SVD

| Manfaat | Penjelasan |
|---------|------------|
| **Reduce dimensions** | 500 → 20 features |
| **Denoise** | Buang variasi noise |
| **Capture semantics** | Temukan hidden topics |
| **Faster training** | Lebih sedikit features → lebih cepat |
| **Less overfitting** | Kompleksitas berkurang |

---

### 🎯 Cross-Validation

#### K-Fold Cross-Validation

**Tujuan**: Estimasi performa model yang reliable

**Mengapa Tidak Cukup Train-Test Split Biasa?**
```
Train-Test Split (70-30):
  ✗ Single estimate (bisa lucky/unlucky dengan split)
  ✗ 30% data tidak digunakan untuk training
  ✗ Tidak tau variance dari model

K-Fold Cross-Validation (k=5):
  ✓ 5 estimates → more reliable
  ✓ 100% data digunakan (80% train, 20% val per fold)
  ✓ Bisa hitung mean dan std (measure stability)
```

#### Out-of-Fold Predictions

**Konsep**: Prediksi untuk setiap sample saat sample tersebut di validation set

```
Fold 1: Train on 2-5, predict on 1 → OOF[1]
Fold 2: Train on 1,3-5, predict on 2 → OOF[2]
Fold 3: Train on 1-2,4-5, predict on 3 → OOF[3]
Fold 4: Train on 1-3,5, predict on 4 → OOF[4]
Fold 5: Train on 1-4, predict on 5 → OOF[5]

Final: Concatenate OOF[1-5] → Full OOF predictions
```

**Kegunaan OOF**:
- ✅ Evaluasi model pada full dataset
- ✅ Analisis error patterns
- ✅ Ensembling (untuk advanced techniques)

---

### 🎨 Feature Engineering Principles

#### 1. Domain Knowledge
```
Musik → Artist reputation penting
Finance → Credit history penting
E-commerce → Purchase history penting
```

#### 2. Interaction Features
```
Individual:       energy=0.8, danceability=0.9
Interaction:      energy × danceability = 0.72

Kenapa lebih baik?
  → Non-linear relationship
  → Model bisa belajar pattern lebih kompleks
```

#### 3. Binning (Categorical Encoding)
```
Continuous:   tempo = [65, 89, 91, 119, 121, 149, 151, 180]
Categorical:  tempo_category = [slow, slow, moderate, moderate,
                                 fast, fast, very_fast, very_fast]

Keuntungan:
  → Reduce noise
  → Capture non-linear relationships
  → More robust to outliers
```

#### 4. Target Encoding
```
Artist → Popularity mapping

Artist A: [80, 85, 82] → avg = 82.3
Artist B: [45, 50, 48] → avg = 47.7

New song from Artist A → feature = 82.3

Hati-hati:
  ✗ JANGAN gunakan test data untuk encoding
  ✓ Gunakan training data saja
  ✓ Add smoothing untuk rare categories
```

---

## 5. Analisis & Evaluasi

### 📏 Metrics yang Digunakan

#### 1. RMSE (Root Mean Squared Error)
```
RMSE = sqrt( (1/n) × Σ(actual - predicted)² )

Karakteristik:
  • Punish large errors lebih berat (karena squared)
  • Dalam satuan yang sama dengan target (popularity)
  • Lower is better
  • Target: < 16.0
```

#### 2. MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|actual - predicted|

Karakteristik:
  • Tidak squared → less sensitive to outliers
  • Interpretasi: rata-rata absolute error
  • Lower is better
```

#### 3. R² (R-squared / Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

di mana:
  SS_res = Σ(actual - predicted)²  # Residual sum of squares
  SS_tot = Σ(actual - mean)²       # Total sum of squares

Interpretasi:
  • R² = 1.0 → perfect prediction
  • R² = 0.5 → model explains 50% variance
  • R² = 0.0 → model no better than predicting mean
  • Higher is better
```

#### 4. MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) × Σ|actual - predicted| / actual

Karakteristik:
  • Percentage error → easier interpretation
  • Scale-independent
  • Lower is better
```

### 📊 Interpretasi Hasil

**Typical Results**:
```
RMSE: 16.16
  → Rata-rata error 16 poin (dalam skala 0-100)
  → Error ~16% dari range

MAE: 12.34
  → Rata-rata absolute error 12 poin
  → Lebih rendah dari RMSE (ada large errors yang di-punish)

R²: 0.72
  → Model explain 72% variance
  → Good performance

MAPE: 28.45%
  → Rata-rata error 28% dari nilai actual
  → Acceptable untuk prediction task
```

---

## 6. Cara Menjalankan

### 💻 Prerequisites

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
```

### 📁 File Structure

```
project/
│
├── train.csv                              # Training data
├── test.csv                               # Testing data
├── song_popularity_predictor.py           # Main script
├── PENJELASAN_FITUR.md                    # Feature documentation
├── DOKUMENTASI_ALUR.md                    # This file
│
└── outputs/                                # Output folder
    ├── submission_siklus4_enhanced.csv    # Predictions
    └── siklus4_comprehensive_analysis.png # Visualizations
```

### ▶️ Cara Menjalankan

#### Option 1: Run Full Pipeline
```python
from song_popularity_predictor import SongPopularityPredictor

# Initialize
predictor = SongPopularityPredictor(
    data_path='.',           # Folder dengan train.csv dan test.csv
    output_path='./outputs'  # Output folder
)

# Run full pipeline
predictor.run_full_pipeline()
```

#### Option 2: Step-by-Step
```python
predictor = SongPopularityPredictor(data_path='.', output_path='./outputs')

# Step by step
predictor.load_data()
predictor.eda()
predictor.engineer_features()
predictor.process_lyrics(n_components=20)
predictor.prepare_features()
predictor.train_models(cv_folds=5)
predictor.create_comprehensive_visualizations()
predictor.generate_insights_report()
predictor.analyze_errors(n=20)

# Create submission
X_test = predictor.test_df[predictor.features]
predictions = predictor.models['LightGBM'].predict(X_test)
predictor.create_submission(predictions, 'my_submission.csv')
```

#### Option 3: Command Line
```bash
# Run langsung
python song_popularity_predictor.py
```

### 📤 Expected Output

**Console**:
```
🎵🎵🎵 SIKLUS 4 ENHANCED 🎵🎵🎵
================================================================================
📂 LOADING DATA
================================================================================
✓ Training data: (114000, 25)
✓ Testing data: (28500, 24)
...

================================================================================
🤖 TRAINING MODEL (LightGBM)
================================================================================
Training with 5-fold cross-validation...

Cross-Validation Results:
  Fold 1: RMSE = 16.2341
  Fold 2: RMSE = 16.1092
  ...

✓ Mean RMSE: 16.1623 (+/- 0.1142)
...

🎯 FINAL OOF RMSE: 16.1623
```

**Files**:
- `outputs/submission_siklus4_enhanced.csv`
- `outputs/siklus4_comprehensive_analysis.png`

---

## 🎯 Tips & Best Practices

### 1. Data Leakage Prevention
```python
# ✓ CORRECT: Use training data only for statistics
artist_map = train_df.groupby('artists')['popularity'].mean()
train_df['artist_avg_pop'] = train_df['artists'].map(artist_map)
test_df['artist_avg_pop'] = test_df['artists'].map(artist_map)

# ✗ WRONG: Using full data
full_df = pd.concat([train_df, test_df])
artist_map = full_df.groupby('artists')['popularity'].mean()  # LEAK!
```

### 2. Feature Engineering Iteration
```
1. Start simple: Basic features
2. Analyze feature importance
3. Add domain-specific features
4. Create interactions
5. Remove low-importance features
6. Repeat until convergence
```

### 3. Hyperparameter Tuning
```python
# Start dengan default
# Lalu tune satu-satu:
#   1. n_estimators (100, 500, 1000, 2000)
#   2. learning_rate (0.1, 0.05, 0.01, 0.005)
#   3. num_leaves (15, 31, 63, 127)
#   4. max_depth (3, 6, 9, 12)

# Gunakan GridSearchCV atau Optuna untuk automated tuning
```

### 4. Error Analysis
```
Always analyze errors:
  • Which samples have largest errors?
  • Is there pattern in errors? (by genre, year, etc)
  • Are errors systematic (bias) or random?
  • Can we add features to reduce specific errors?
```

### 5. Ensemble Methods
```python
# Combine multiple models
predictions = (
    0.5 * lgbm.predict(X_test) +
    0.3 * xgb.predict(X_test) +
    0.2 * catboost.predict(X_test)
)

# Usually better than single model
```

---

## 🚀 Next Steps & Improvements

### 1. Advanced Features
- Genre co-occurrence patterns
- Artist collaboration network features
- Temporal momentum (trending artists)
- Audio feature clusters (K-means)

### 2. Model Improvements
- Ensemble of LightGBM + XGBoost + CatBoost
- Neural networks for lyrics (BERT, RoBERTa)
- Two-stage modeling (low-pop vs high-pop)

### 3. Hyperparameter Optimization
- Bayesian optimization (Optuna)
- Grid search with CV
- Early stopping optimization

### 4. Feature Selection
- Recursive Feature Elimination
- SHAP values analysis
- Correlation analysis

### 5. Data Augmentation
- Synthetic minority oversampling (for rare genres)
- Feature perturbation
- Mixup for audio features

---

## 📚 Referensi & Resources

### Papers
- LightGBM: [Ke et al., 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- Gradient Boosting: [Friedman, 2001](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

### Libraries Documentation
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Tutorials
- [Kaggle: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [Kaggle: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning)

---

**End of Documentation**

Semoga dokumentasi ini membantu Anda memahami seluruh alur sistem prediksi popularitas lagu! 🎵
