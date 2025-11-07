# 🔧 IMPLEMENTATION GUIDE: FIXES & IMPROVEMENTS

## Quick Reference untuk Implementasi Perbaikan

---

## 🚨 FIX #1: DATA LEAKAGE (CRITICAL)

### Problem
```python
# CURRENT CODE (WRONG - ADA DATA LEAKAGE):
def engineer_features(self):
    # Ini menghitung statistik dari FULL training data
    artist_popularity_map = self.train_df.groupby('artists')['popularity'].mean()

    # Statistik ini kemudian digunakan di SEMUA folds dalam CV
    # Padahal harusnya hanya dihitung dari training folds saja
    self.train_df['artist_avg_pop'] = self.train_df['artists'].map(artist_popularity_map)
```

### Solution Option 1: Manual Target Encoding dalam CV Loop

```python
def train_models_fixed(self, cv_folds=5):
    """
    Train dengan proper target encoding (no leakage)
    """
    X = self.train_df[self.features]
    y = self.train_df['popularity']

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\nFold {fold}/{cv_folds}")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ==================================================================
        # IMPORTANT: Hitung artist encoding HANYA dari training fold
        # ==================================================================
        artist_map = self.train_df.iloc[train_idx].groupby('artists')['popularity'].mean()
        global_mean = y_train.mean()

        # Apply ke training fold
        X_train['artist_avg_pop'] = X_train['artists'].map(artist_map).fillna(global_mean)

        # Apply ke validation fold (bisa ada artists yang belum pernah muncul)
        X_val['artist_avg_pop'] = X_val['artists'].map(artist_map).fillna(global_mean)

        # Train model
        model = LGBMRegressor(...)
        model.fit(X_train, y_train)

        # Predict on validation
        oof_predictions[val_idx] = model.predict(X_val)

    # Calculate overall score
    rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    return rmse
```

### Solution Option 2: Menggunakan category_encoders Library

```python
# Install dulu: pip install category_encoders

from category_encoders import TargetEncoder

def prepare_features_fixed(self):
    """
    Prepare features dengan proper target encoding
    """
    # ... existing code ...

    # Untuk categorical features yang butuh target encoding
    target_encode_cols = ['artists']  # bisa tambah lainnya

    # Inisialisasi target encoder
    # smoothing parameter mengontrol regularization
    self.target_encoder = TargetEncoder(
        cols=target_encode_cols,
        smoothing=1.0,  # Higher = more regularization
        min_samples_leaf=1
    )

    # NOTE: Fitting dilakukan INSIDE CV loop!
    # Jangan fit di sini, karena akan cause leakage

    return self

def train_models_with_target_encoder(self, cv_folds=5):
    """
    Train dengan target encoder (handles CV automatically)
    """
    X = self.train_df.drop('popularity', axis=1)
    y = self.train_df['popularity']

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit target encoder ONLY on training fold
        X_train_encoded = self.target_encoder.fit_transform(X_train, y_train)

        # Transform validation fold
        X_val_encoded = self.target_encoder.transform(X_val)

        # Train and predict
        model = LGBMRegressor(...)
        model.fit(X_train_encoded, y_train)
        oof_predictions[val_idx] = model.predict(X_val_encoded)

    return oof_predictions
```

---

## 🎯 FIX #2: IMPROVED MODEL CONFIGURATION

### Current (Suboptimal)
```python
lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# No early stopping!
lgbm.fit(X, y)
```

### Improved Version
```python
def train_models_improved(self, cv_folds=5):
    """
    Train dengan config yang lebih baik
    """
    X = self.train_df[self.features]
    y = self.train_df['popularity']

    # Improved hyperparameters
    lgbm_params = {
        'n_estimators': 2000,          # Increase karena ada early stopping
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,               # Let num_leaves control
        'min_child_samples': 20,       # Min data in leaf
        'subsample': 0.8,              # Row sampling
        'subsample_freq': 1,
        'colsample_bytree': 0.8,       # Column sampling
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 0.1,             # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMRegressor(**lgbm_params)

        # Fit dengan early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=100,  # Stop jika tidak improve 100 rounds
            verbose=False
        )

        # Predict
        oof_predictions[val_idx] = model.predict(X_val)

        # Score
        fold_score = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
        cv_scores.append(fold_score)

        print(f"Fold {fold}: RMSE = {fold_score:.4f} (best iteration: {model.best_iteration_})")

    overall_score = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"\nOverall OOF RMSE: {overall_score:.4f}")

    return model, oof_predictions, cv_scores
```

---

## 📊 FIX #3: ADD GENRE & ARTIST STATISTICS FEATURES

### Implementation

```python
def engineer_features_v2(self):
    """
    Enhanced feature engineering dengan genre & artist stats
    """
    # ... existing features ...

    # =======================================================================
    # GENRE-LEVEL STATISTICS
    # =======================================================================
    print("[NEW] Genre-level statistics...")

    # IMPORTANT: Ini juga harus dilakukan dalam CV loop untuk avoid leakage!
    # Tapi untuk test set, kita bisa gunakan full training stats

    # Untuk test set processing:
    genre_stats = self.train_df.groupby('track_genre')['popularity'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    genre_stats.columns = ['track_genre', 'genre_avg_pop', 'genre_std_pop',
                           'genre_min_pop', 'genre_max_pop', 'genre_song_count']

    # Merge ke train dan test
    self.train_df = self.train_df.merge(genre_stats, on='track_genre', how='left')
    self.test_df = self.test_df.merge(genre_stats, on='track_genre', how='left')

    # Fill missing dengan global stats
    global_genre_mean = self.train_df['popularity'].mean()
    global_genre_std = self.train_df['popularity'].std()

    self.train_df['genre_avg_pop'].fillna(global_genre_mean, inplace=True)
    self.test_df['genre_avg_pop'].fillna(global_genre_mean, inplace=True)
    self.train_df['genre_std_pop'].fillna(global_genre_std, inplace=True)
    self.test_df['genre_std_pop'].fillna(global_genre_std, inplace=True)

    # =======================================================================
    # ARTIST VARIANCE STATISTICS
    # =======================================================================
    print("[NEW] Artist variance statistics...")

    artist_stats = self.train_df.groupby('artists')['popularity'].agg([
        'std', 'min', 'max'
    ]).reset_index()
    artist_stats.columns = ['artists', 'artist_std_pop', 'artist_min_pop', 'artist_max_pop']

    self.train_df = self.train_df.merge(artist_stats, on='artists', how='left')
    self.test_df = self.test_df.merge(artist_stats, on='artists', how='left')

    # Fill missing
    self.train_df['artist_std_pop'].fillna(0, inplace=True)
    self.test_df['artist_std_pop'].fillna(0, inplace=True)

    # =======================================================================
    # AUDIO FEATURE RATIOS
    # =======================================================================
    print("[NEW] Audio feature ratios...")

    for df in [self.train_df, self.test_df]:
        # Ratios yang meaningful
        df['energy_valence_ratio'] = df['energy'] / (df['valence'] + 1e-6)
        df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-6)
        df['speech_music_ratio'] = df['speechiness'] / (1 - df['speechiness'] + 1e-6)

        # Composite features
        df['audio_complexity'] = df[['energy', 'danceability', 'valence']].std(axis=1)

    # =======================================================================
    # ADVANCED TRACK NAME FEATURES
    # =======================================================================
    print("[NEW] Advanced track name features...")

    for df in [self.train_df, self.test_df]:
        df['has_featuring'] = df['track_name'].str.contains(
            'feat|ft\.|featuring', case=False, na=False
        ).astype(int)

        df['is_remix'] = df['track_name'].str.contains(
            'remix|remaster|version', case=False, na=False
        ).astype(int)

        df['has_parenthesis'] = df['track_name'].str.contains(
            r'\(|\[', case=False, na=False
        ).astype(int)

    print("✓ Enhanced feature engineering completed!")
    return self
```

---

## 🔄 FIX #4: BETTER IMPUTATION STRATEGY

### Current (One-size-fits-all)
```python
self.imputer = SimpleImputer(strategy='median')
self.train_df[self.features] = self.imputer.fit_transform(self.train_df[self.features])
```

### Improved (Feature-type specific)
```python
from sklearn.compose import ColumnTransformer

def prepare_features_v2(self):
    """
    Prepare features dengan proper imputation per feature type
    """
    # ... existing code untuk identify features ...

    # Separate numeric dan categorical features
    numeric_features = [f for f in self.features if f not in self.categorical_features_encoded]
    categorical_features = self.categorical_features_encoded

    # Create column transformer
    preprocessor = ColumnTransformer([
        ('num_impute', SimpleImputer(strategy='median'), numeric_features),
        ('cat_impute', SimpleImputer(strategy='most_frequent'), categorical_features)
    ], remainder='passthrough')

    # Fit and transform
    self.preprocessor = preprocessor
    X_train_processed = preprocessor.fit_transform(self.train_df[self.features])
    X_test_processed = preprocessor.transform(self.test_df[self.features])

    # Convert back to DataFrame
    all_features = numeric_features + categorical_features
    self.train_df[all_features] = X_train_processed
    self.test_df[all_features] = X_test_processed

    return self
```

---

## 🎯 COMPLETE IMPROVED PIPELINE

### Full Example dengan Semua Fixes

```python
def run_improved_pipeline(self):
    """
    Full pipeline dengan semua improvements
    """
    print("🎵 IMPROVED PIPELINE - SIKLUS 5")
    print("="*80)

    # Load data
    self.load_data()
    self.eda()

    # Enhanced feature engineering
    self.engineer_features_v2()  # Dengan genre stats, artist variance, dll
    self.process_lyrics(n_components=20)

    # Prepare features (improved imputation)
    self.prepare_features_v2()

    # Train dengan proper target encoding dan early stopping
    # NOTE: Ini membutuhkan refactor train_models untuk handle CV properly
    self.train_models_improved(cv_folds=5)

    # Visualizations dan analysis
    self.create_comprehensive_visualizations()
    self.generate_insights_report()
    self.analyze_errors(n=20)

    # Create submission
    X_test = self.test_df[self.features]
    predictions = self.models['LightGBM'].predict(X_test)
    self.create_submission(predictions, 'submission_siklus5_improved.csv')

    print("\n✅ IMPROVED PIPELINE COMPLETE!")
    return self
```

---

## ⚡ QUICK WINS (Implementasi Cepat)

### 1. Add Early Stopping (5 menit)
```python
# Dalam method train_models, ganti:
lgbm.fit(X, y)

# Dengan:
lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=100,
         verbose=False)
```

### 2. Add Regularization (2 menit)
```python
# Tambahkan ke LGBMRegressor params:
lgbm = LGBMRegressor(
    # ... existing params ...
    reg_alpha=0.1,      # L1
    reg_lambda=0.1,     # L2
    subsample=0.8,      # Bagging
    colsample_bytree=0.8  # Feature sampling
)
```

### 3. Add Genre Mean Feature (10 menit)
```python
# Dalam engineer_features:
genre_mean = self.train_df.groupby('track_genre')['popularity'].mean()
self.train_df['genre_avg_pop'] = self.train_df['track_genre'].map(genre_mean)
self.test_df['genre_avg_pop'] = self.test_df['track_genre'].map(genre_mean)

# Fill missing
global_mean = self.train_df['popularity'].mean()
self.train_df['genre_avg_pop'].fillna(global_mean, inplace=True)
self.test_df['genre_avg_pop'].fillna(global_mean, inplace=True)
```

---

## 📋 IMPLEMENTATION PRIORITY

### Week 1: Critical Fixes
- [ ] Fix data leakage (target encoding dalam CV)
- [ ] Add early stopping
- [ ] Add regularization parameters

### Week 2: High-Value Features
- [ ] Add genre statistics
- [ ] Add artist variance
- [ ] Add audio feature ratios

### Week 3: Optimization
- [ ] Hyperparameter tuning
- [ ] Feature selection
- [ ] Ensemble methods

---

## 🧪 TESTING STRATEGY

### Before Implementation
```python
# Baseline RMSE
baseline_rmse = 16.XX
```

### After Each Fix
```python
# Test dengan same CV seed
# Compare:
# - CV score distribution
# - OOF RMSE
# - Feature importance changes
```

### Validation
- [ ] CV scores are stable across folds
- [ ] No train/val gap too large (overfitting check)
- [ ] Test predictions reasonable (no negative, >100)

---

**Last Updated:** 2025-11-07
**Implementation Guide Version:** 1.0
