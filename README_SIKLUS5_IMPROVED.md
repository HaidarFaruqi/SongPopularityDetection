# 🎵 Song Popularity Detection - SIKLUS 5 IMPROVED

## 🚀 Major Improvements dari Siklus 4

### ✅ CRITICAL FIXES

#### 1. **Fixed Data Leakage** 🔴 CRITICAL
**Problem:** Artist & genre target encoding menggunakan full training data, termasuk validation fold
**Solution:** Target encoding dilakukan DALAM CV loop, hanya menggunakan training fold data
**Impact:** CV scores jadi lebih reliable, tidak over-optimistic

#### 2. **Early Stopping & Regularization**
**Added:**
- Early stopping (100 rounds)
- L1 regularization (reg_alpha=0.1)
- L2 regularization (reg_lambda=0.1)
- Row sampling (subsample=0.8)
- Column sampling (colsample_bytree=0.8)

**Impact:** Better generalization, prevent overfitting

---

### 🆕 NEW FEATURES ADDED

#### 1. **Genre Statistics** (5 features)
```python
- genre_avg_pop      # Average popularity untuk genre tersebut
- genre_std_pop      # Std deviation
- genre_min_pop      # Min popularity
- genre_max_pop      # Max popularity
- genre_song_count   # Jumlah lagu dalam genre
```
**Expected Impact:** -0.1 to -0.2 RMSE

#### 2. **Artist Variance Features** (5 features)
```python
- artist_avg_pop     # Average (sudah ada di v4, di-improve)
- artist_std_pop     # NEW: Std deviation (consistency indicator)
- artist_min_pop     # NEW: Min popularity
- artist_max_pop     # NEW: Max popularity
- artist_song_count  # Count
```
**Expected Impact:** -0.05 to -0.1 RMSE

#### 3. **Audio Feature Ratios** (8+ features)
```python
- energy_valence_ratio
- energy_acoustic_ratio
- dance_acoustic_ratio
- speech_music_ratio
- loudness_energy_alignment
- audio_feature_std
- audio_feature_mean
```
**Expected Impact:** -0.03 to -0.05 RMSE

#### 4. **Advanced Track Name Features** (5 new features)
```python
- has_featuring           # Featuring artist?
- is_remix               # Remix/remaster?
- has_parenthesis        # Has extra info in parenthesis?
- has_special_edition    # Deluxe/special edition?
- title_word_diversity   # Unique words / total words ratio
```
**Expected Impact:** -0.02 to -0.03 RMSE

#### 5. **Enhanced Temporal Features**
```python
- era                    # More granular than decade (pre_70, 70s, 80s, etc)
- decade_avg_pop         # Average popularity untuk decade tersebut
```

---

### 📊 COMPREHENSIVE EDA & INSIGHTS

#### Before Processing:
1. **Target Distribution Analysis**
   - Distribution by popularity bins
   - Imbalance detection
   - Skewness & kurtosis

2. **Genre Analysis**
   - Genre count & popularity stats
   - Top genres by popularity
   - Genre diversity

3. **Artist Analysis**
   - Total unique artists
   - Single-song vs prolific artists
   - Artist-popularity relationships

4. **Audio Features Correlation**
   - Individual correlations with target
   - Strength classification (Strong/Moderate/Weak)

5. **Temporal Analysis**
   - Popularity by decade
   - Time trends
   - Year range coverage

6. **Data Quality Checks**
   - Duplicate detection
   - Outlier detection
   - Missing values analysis

7. **Multicollinearity Check**
   - High correlation pairs
   - Recommendations for dimensionality reduction

8. **Feature Engineering Opportunities**
   - Automated detection of opportunities

---

## 📁 File Structure (Improved)

```
.
├── main_improved_pipeline.py          # 🔥 Main script - Run this!
├── enhanced_features.py               # Enhanced feature engineering
├── proper_cv_training.py              # Proper CV training (no leakage)
├── song_popularity_improved.py        # Class-based implementation
│
├── TPW_AhThatsHot.ipynb              # Original notebook (Siklus 4)
├── TPW_AhThatsHot_Improved.ipynb     # New improved notebook
│
├── REVIEW_FEATURE_AND_DATA.md        # Comprehensive review
├── IMPLEMENTATION_FIXES.md            # Implementation guide
├── data_validation.py                 # Data validation script
│
└── outputs/
    └── submission_siklus5_improved.csv
```

---

## 🏃 Quick Start

### Option 1: Run Main Script (Recommended)

```bash
python main_improved_pipeline.py
```

**Output:**
- Comprehensive EDA
- Feature engineering with insights
- Proper CV training (no leakage!)
- Test predictions
- `submission_siklus5_improved.csv`

### Option 2: Step-by-Step

```python
from enhanced_features import engineer_features_enhanced
from proper_cv_training import train_model_with_proper_cv

# Load data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Feature engineering
df_train, df_test, mappings = engineer_features_enhanced(df_train, df_test)

# Train with proper CV
results = train_model_with_proper_cv(df_train, features, cv_folds=5)

# Done!
```

---

## 📈 Expected Performance

### Baseline (Siklus 4):
```
OOF RMSE: ~16.0-16.5
Issues: Data leakage, missing features, no regularization
```

### Improved (Siklus 5):
```
Expected OOF RMSE: ~15.5-16.0
Improvement: 0.3-0.5 RMSE ⬇️

Breakdown:
- Fix data leakage:      ±0.0 to -0.1  (More realistic CV)
- Genre features:        -0.1 to -0.2  🔥
- Artist variance:       -0.05 to -0.1
- Audio ratios:          -0.03 to -0.05
- Better hyperparams:    -0.05 to -0.1
- Other features:        -0.05 to -0.1
```

---

## 🔧 Key Technical Details

### 1. Proper Target Encoding (No Leakage)

**❌ WRONG (Siklus 4):**
```python
# Menggunakan full training data
artist_map = train_df.groupby('artists')['popularity'].mean()

# Di-apply ke semua folds → LEAKAGE!
train_df['artist_avg_pop'] = train_df['artists'].map(artist_map)
```

**✅ CORRECT (Siklus 5):**
```python
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    # Hitung hanya dari training fold
    df_train_fold = train_df.iloc[train_idx]
    artist_map = df_train_fold.groupby('artists')['popularity'].mean()

    # Apply ke validation fold
    df_val_fold['artist_avg_pop'] = df_val_fold['artists'].map(artist_map)
```

### 2. Improved Model Configuration

```python
LGBMRegressor(
    n_estimators=2000,           # Increased for early stopping
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,               # NEW: Row sampling
    colsample_bytree=0.8,        # NEW: Column sampling
    reg_alpha=0.1,               # NEW: L1 regularization
    reg_lambda=0.1,              # NEW: L2 regularization
)

# With early stopping
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=100)
```

### 3. Feature Engineering Pipeline

```
Raw Data
    ↓
Base Audio Features (energy_x_dance, duration_min, etc.)
    ↓
Audio Ratios (energy_valence_ratio, etc.)
    ↓
Temporal Features (era, years_since_release, etc.)
    ↓
Track Name Features (has_featuring, is_remix, etc.)
    ↓
[IN CV LOOP] Genre Statistics
    ↓
[IN CV LOOP] Artist Statistics
    ↓
Lyrics Features (TF-IDF + SVD)
    ↓
Ready for Modeling!
```

---

## 📊 Feature Categories & Count

| Category | Features | Total |
|----------|----------|-------|
| **Original Audio** | energy, danceability, valence, etc. | ~10 |
| **Audio Derived** | energy_x_dance, duration_min, etc. | ~4 |
| **Audio Ratios** | energy_valence_ratio, etc. | ~8 |
| **Temporal** | years_since_release, decade, era, etc. | ~6 |
| **Track Name** | length, word_count, has_featuring, etc. | ~9 |
| **Genre Stats** | genre_avg_pop, genre_std_pop, etc. | ~5 |
| **Artist Stats** | artist_avg_pop, artist_std_pop, etc. | ~5 |
| **Lyrics** | lyrics_feature_0 to lyrics_feature_19 | ~20 |
| **Total** | | **~67** |

---

## 🎯 Validation Strategy

### Cross-Validation:
- **Method:** 5-Fold KFold
- **Shuffle:** Yes (random_state=42)
- **Target Encoding:** Per-fold (no leakage!)
- **Early Stopping:** Yes (100 rounds)

### Metrics Tracked:
- **RMSE** (primary metric)
- **MAE** (robustness check)
- **R²** (explained variance)

### Feature Importance:
- Tracked per-fold
- Aggregated (mean ± std)
- Top features analysis

---

## 🔍 Insights Generated

### Data Loading Phase:
- Dataset shapes
- Target statistics (mean, std, skew, kurtosis)
- Missing values summary

### EDA Phase:
- Target distribution by bins
- Genre analysis (top genres, popularity by genre)
- Artist analysis (prolific artists, single-song artists)
- Audio correlations with target
- Temporal trends
- Data quality issues
- Multicollinearity detection
- Feature engineering opportunities

### Training Phase:
- Per-fold RMSE, MAE, R²
- CV statistics (mean, std, min, max)
- Overall OOF performance
- Best iteration per fold
- Feature importance rankings

---

## 📝 Usage Examples

### Example 1: Run Full Pipeline
```python
from main_improved_pipeline import run_improved_pipeline

results, submission = run_improved_pipeline(
    data_path='/content',
    output_path='./outputs'
)

print(f"OOF RMSE: {results['overall_rmse']:.4f}")
```

### Example 2: Just Feature Engineering
```python
from enhanced_features import engineer_features_enhanced

df_train, df_test, mappings = engineer_features_enhanced(df_train, df_test)

# mappings contains genre_stats & artist_stats untuk reference
```

### Example 3: Train Custom Model
```python
from proper_cv_training import train_model_with_proper_cv

results = train_model_with_proper_cv(
    df_train,
    features=my_features,
    target_col='popularity',
    cv_folds=5
)

# results contains models, predictions, scores, feature importance
```

---

## 🐛 Debugging & Troubleshooting

### Issue 1: "KeyError: 'genre_avg_pop'"
**Solution:** Don't manually add genre/artist features to train set. Mereka akan dibuat otomatis dalam CV loop.

### Issue 2: CV scores sangat berbeda antar folds
**Possible causes:**
- Data imbalance (consider stratified CV)
- Temporal dependency (consider TimeSeriesSplit)
- Outliers (check data quality)

### Issue 3: OOF RMSE berbeda dari CV mean RMSE
**Normal!** OOF dihitung dari concatenated predictions across all folds. Bisa sedikit berbeda dari mean CV score.

---

## 🎓 Key Learnings

### 1. Data Leakage is Serious
- Even "simple" target encoding bisa cause leakage
- Always encode features INSIDE CV loop
- Validate dengan comparing train vs val performance

### 2. Feature Engineering > Hyperparameter Tuning
- Good features: -0.3 RMSE improvement
- Hyperparameter tuning: -0.05 to -0.1 RMSE
- Focus on features first!

### 3. Regularization is Important
- L1 + L2 regularization
- Row & column sampling
- Early stopping
- All contribute to better generalization

### 4. Comprehensive EDA Pays Off
- Identify data quality issues early
- Discover feature engineering opportunities
- Understand target distribution
- Detect multicollinearity

---

## 📚 References & Resources

### Papers:
- LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- Target Encoding Done the Right Way

### Libraries:
- LightGBM: https://lightgbm.readthedocs.io/
- scikit-learn: https://scikit-learn.org/
- pandas: https://pandas.pydata.org/

### Internal Docs:
- `REVIEW_FEATURE_AND_DATA.md` - Detailed review
- `IMPLEMENTATION_FIXES.md` - Implementation guide
- `data_validation.py` - Validation script

---

## 🤝 Contributing

Untuk improvements lebih lanjut:
1. Check `REVIEW_FEATURE_AND_DATA.md` untuk ideas
2. Implement di modular files
3. Update documentation
4. Run validation & compare results

---

## 📊 Changelog

### Siklus 5 (Current) - IMPROVED
- ✅ Fixed data leakage
- ✅ Added 23+ new features
- ✅ Improved model configuration
- ✅ Comprehensive EDA & insights
- ✅ Modular code structure

### Siklus 4 (Baseline)
- Feature engineering (basic)
- LightGBM model
- 15 visualizations
- ⚠️ Data leakage issue

---

## 📧 Contact & Support

Untuk questions atau issues:
- Check documentation first
- Review `IMPLEMENTATION_FIXES.md` untuk common issues
- Run `data_validation.py` untuk verify data

---

**Last Updated:** 2025-11-07
**Version:** Siklus 5 - Improved
**Expected Performance:** 15.5-16.0 RMSE
**Status:** ✅ Production Ready
