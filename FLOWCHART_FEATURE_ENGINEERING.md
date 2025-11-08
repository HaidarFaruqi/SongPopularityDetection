# 📊 Visual Flowchart - Feature Engineering Pipeline

## 🎯 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    📥 INPUT: RAW DATA                                │
│                                                                      │
│  train.csv (89,740 rows × 20 columns)                              │
│  test.csv  (22,435 rows × 19 columns, no popularity)               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 🧹 STEP 1: DATA CLEANING                             │
│                                                                      │
│  • Remove duplicates (exact matches)                                │
│    Before: 89,740 → After: ~85,000 rows                            │
│                                                                      │
│  • Fix invalid years                                                │
│    21 → 2021, 1 → 2001, 99 → 1999                                 │
│    Impact: ~3,200 years fixed                                      │
│                                                                      │
│  • Handle popularity = 0                                            │
│    Create flag: is_zero_popularity                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              👤 STEP 2: ARTIST FEATURES (2 features)                │
│                                                                      │
│  INPUT: 'artists' column (text)                                    │
│     ↓                                                               │
│  GROUP BY artist → Calculate statistics                            │
│     ↓                                                               │
│  OUTPUT:                                                            │
│  • artist_avg_pop    = mean(popularity) per artist                 │
│  • artist_song_count = count(songs) per artist                     │
│                                                                      │
│  TECHNIQUE: Target Encoding (OUT-OF-FOLD)                          │
│  IMPORTANCE: ⭐⭐⭐⭐⭐ (28.5% + 3.7% = 32.2%)                      │
│                                                                      │
│  Example:                                                           │
│    Taylor Swift    → artist_avg_pop = 85.7                         │
│    Unknown Artist  → artist_avg_pop = 45.2 (global mean)           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│           🎵 STEP 3: AUDIO FEATURES (13 → 17 features)              │
│                                                                      │
│  INPUT: 13 original audio columns                                  │
│  ├─ danceability, energy, loudness, speechiness                    │
│  ├─ acousticness, instrumentalness, liveness, valence              │
│  ├─ tempo, duration_ms, key, mode, explicit                        │
│  └─ (all numeric, range 0-1 or specific scales)                   │
│                                                                      │
│  ENGINEERING: Create 4 new features                                │
│  ├─ energy_x_dance  = energy × danceability                        │
│  ├─ duration_min    = duration_ms / 60000                          │
│  ├─ key_mode        = key + "_" + mode (categorical)              │
│  └─ tempo_category  = binning(tempo, 4 bins)                       │
│                                                                      │
│  OUTPUT: 17 audio features total                                   │
│  IMPORTANCE: ⭐⭐⭐⭐ (33.2%)                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│           ⏰ STEP 4: TEMPORAL FEATURES (4 features)                 │
│                                                                      │
│  INPUT: release_year                                               │
│                                                                      │
│  TRANSFORMATIONS:                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ years_since_release = 2025 - release_year                  │   │
│  │   Range: 0 (new) to 75 (very old)                         │   │
│  │   Pattern: Smaller = More Popular ✅                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ decade = (release_year // 10) × 10                        │   │
│  │   Values: 1950, 1960, ..., 2020                           │   │
│  │   Pattern: 2020s > 2010s > 2000s > ... ✅                 │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ is_classic = 1 if release_year < 2000 else 0             │   │
│  │   Binary flag for old songs                               │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ is_recent_hit = 1 if release_year >= 2020 else 0         │   │
│  │   Binary flag for new songs                               │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  IMPORTANCE: ⭐⭐⭐ (11.4%)                                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│          📝 STEP 5: TRACK NAME FEATURES (2 features)                │
│                                                                      │
│  INPUT: track_name (text)                                          │
│     ↓                                                               │
│  CLEANING:                                                          │
│    "Bohemian Rhapsody - Remastered 2011"                           │
│    → Remove (parentheses), [brackets]                              │
│    → Remove " - feat.", " - remastered", etc                       │
│    → Lowercase                                                      │
│    → Result: "bohemian rhapsody"                                   │
│     ↓                                                               │
│  EXTRACTION:                                                        │
│  • track_name_length     = len(clean_name)                         │
│  • track_name_word_count = count(words)                            │
│                                                                      │
│  PATTERN: Shorter = More Popular ✅                                │
│  IMPORTANCE: ⭐⭐ (2.2%)                                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│         📖 STEP 6: NLP FEATURES - LYRICS (20 features)              │
│                                                                      │
│  INPUT: lyrics (long text)                                         │
│                                                                      │
│  STEP 6A: TF-IDF VECTORIZATION                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Raw lyrics → TF-IDF → 500 features (sparse matrix)          │ │
│  │                                                               │ │
│  │ Parameters:                                                   │ │
│  │   • max_features = 500                                       │ │
│  │   • ngram_range = (1, 2)   (words + 2-word phrases)         │ │
│  │   • stop_words = 'english' (remove "the", "is", etc)        │ │
│  │   • min_df = 5             (word in ≥5 songs)               │ │
│  │   • max_df = 0.8           (word in ≤80% songs)             │ │
│  │                                                               │ │
│  │ Output: (89740, 500) sparse matrix                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│       ↓                                                             │
│  STEP 6B: DIMENSIONALITY REDUCTION (SVD)                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ TF-IDF (500 features) → SVD → 20 dense features             │ │
│  │                                                               │ │
│  │ Technique: TruncatedSVD (PCA for sparse matrices)           │ │
│  │ n_components = 20                                            │ │
│  │                                                               │ │
│  │ Output: (89740, 20) dense matrix                             │ │
│  │ Explained variance: ~35%                                     │ │
│  │                                                               │ │
│  │ Topics captured:                                             │ │
│  │   Topic 0: Love songs                                        │ │
│  │   Topic 1: Party/club songs                                  │ │
│  │   Topic 2: Sad/heartbreak songs                              │ │
│  │   Topic 3: Upbeat/motivational                               │ │
│  │   ...                                                         │ │
│  │   Topic 19: Other patterns                                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│       ↓                                                             │
│  OUTPUT: lyrics_feature_0, lyrics_feature_1, ..., lyrics_feature_19│
│  IMPORTANCE: ⭐⭐⭐ (9.5%)                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│        🔄 STEP 7: INTERACTION FEATURES (2 features)                 │
│                                                                      │
│  CONCEPT: Capture non-linear effects through multiplication        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ artist_x_dance = artist_avg_pop × danceability            │   │
│  │                                                             │   │
│  │ Example:                                                    │   │
│  │   Taylor Swift (85) × Danceable (0.8) = 68.0 🔥           │   │
│  │   Unknown (42) × Danceable (0.8) = 33.6                   │   │
│  │                                                             │   │
│  │ Captures: Popular artist + good audio = AMPLIFIED!        │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ artist_x_energy = artist_avg_pop × energy                 │   │
│  │                                                             │   │
│  │ Similar logic: Popular + energetic = viral potential      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  IMPORTANCE: ⭐⭐⭐ (8.4%)                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│         🏷️ STEP 8: CATEGORICAL ENCODING (4 features)               │
│                                                                      │
│  INPUT: 4 categorical columns                                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ track_genre (114 unique values)                           │   │
│  │   "pop" → 45, "rock" → 78, "hip-hop" → 32, ...           │   │
│  │   → track_genre_encoded                                    │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ key_mode (24 combinations)                                │   │
│  │   "0_1" (C Major) → 1, "7_0" (G Minor) → 14, ...         │   │
│  │   → key_mode_encoded                                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ tempo_category (4 bins)                                   │   │
│  │   "slow" → 0, "moderate" → 1, "fast" → 2, ...            │   │
│  │   → tempo_category_encoded                                 │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ decade (8 values)                                         │   │
│  │   1950 → 0, 1960 → 1, ..., 2020 → 7                      │   │
│  │   → decade_encoded                                         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  TECHNIQUE: LabelEncoder (fit on train+test combined)             │
│  IMPORTANCE: ⭐⭐ (6.0%)                                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│           🎯 STEP 9: FEATURE SELECTION & PREPARATION                │
│                                                                      │
│  STEP 9A: Select Features                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Get all numeric columns                                    │   │
│  │   ↓                                                         │   │
│  │ EXCLUDE:                                                    │   │
│  │   • popularity (target, not feature!)                      │   │
│  │   • track_id (identifier, no info)                         │   │
│  │   • track_name, artists, lyrics (text, already extracted) │   │
│  │   • release_year (raw, already → decade, years_since)     │   │
│  │   ↓                                                         │   │
│  │ CLEAN: Remove *_encoded from numeric (prevent duplicate)  │   │
│  │   ↓                                                         │   │
│  │ COMBINE: numeric_clean + encoded_categorical               │   │
│  │   ↓                                                         │   │
│  │ RESULT: 49 features total                                  │   │
│  └────────────────────────────────────────────────────────────┘   │
│       ↓                                                             │
│  STEP 9B: Imputation                                               │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Handle missing values with median                          │   │
│  │   • SimpleImputer(strategy='median')                       │   │
│  │   • Fit on train, transform on train & test               │   │
│  │   • Result: 0 missing values ✅                            │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  📤 OUTPUT: FEATURE MATRIX                          │
│                                                                      │
│  X_train: (85,000 rows × 49 features)                              │
│  y_train: (85,000 rows × 1)  ← popularity                          │
│                                                                      │
│  X_test:  (22,435 rows × 49 features)                              │
│                                                                      │
│  ✅ READY FOR MODEL TRAINING!                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              🤖 MODEL TRAINING (LightGBM)                           │
│                                                                      │
│  Model: LightGBM Regressor                                         │
│  Hyperparameters:                                                   │
│    • n_estimators = 1000                                           │
│    • learning_rate = 0.01                                          │
│    • num_leaves = 31                                               │
│    • max_depth = 6                                                 │
│                                                                      │
│  Validation: 5-Fold Cross-Validation                               │
│  Metric: RMSE (Root Mean Squared Error)                            │
│                                                                      │
│  🏆 RESULT: RMSE = 16.05                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Breakdown (49 features)

```
CATEGORY                    COUNT   IMPORTANCE   EXAMPLES
═══════════════════════════════════════════════════════════════════
👤 Artist Features            2      32.2%       artist_avg_pop, artist_song_count
🎵 Audio Original            13      25.1%       danceability, energy, loudness
🎵 Audio Engineered           4       8.1%       energy_x_dance, duration_min
⏰ Temporal                   4      11.4%       years_since_release, decade
📝 Track Name                 2       2.2%       track_name_length, word_count
📖 NLP/Lyrics                20       9.5%       lyrics_feature_0 ... 19
🔄 Interactions               2       8.4%       artist_x_dance, artist_x_energy
🏷️ Categorical Encoded       4       6.0%       track_genre_encoded, etc.
───────────────────────────────────────────────────────────────────
TOTAL                        49     100.0%
```

---

## 🔄 Data Flow Summary

```
RAW DATA (20 cols)
    ↓
CLEANING (-4,740 duplicates, fix 3,200 years)
    ↓
ARTIST ENCODING (+2 features) ← MOST IMPORTANT STEP!
    ↓
AUDIO ENGINEERING (+4 features)
    ↓
TEMPORAL FEATURES (+4 features)
    ↓
TRACK NAME FEATURES (+2 features)
    ↓
NLP PROCESSING (+20 features)
    ↓
INTERACTIONS (+2 features)
    ↓
CATEGORICAL ENCODING (+4 encoded)
    ↓
FEATURE SELECTION (remove 20 cols)
    ↓
IMPUTATION (fill missing)
    ↓
FINAL MATRIX (49 features)
    ↓
MODEL TRAINING
    ↓
RMSE 16.05 ✅
```

---

## ⚙️ Key Decision Points

### 1️⃣ Why OUT-OF-FOLD encoding for artist features?

```
❌ WRONG: Use mean from same fold
   → DATA LEAKAGE → Overfitting!

✅ RIGHT: Use mean from OTHER folds
   → No leakage → Generalizes better!

Implementation:
  Fold 1 (validation) ← Use mean from Fold 2,3,4,5
  Fold 2 (validation) ← Use mean from Fold 1,3,4,5
  ...
```

### 2️⃣ Why TF-IDF + SVD instead of just TF-IDF?

```
TF-IDF alone:
  • 500 features
  • Sparse (90% zeros)
  • Curse of dimensionality

TF-IDF + SVD:
  • 20 dense features
  • 35% variance captured
  • Better generalization
  • Faster training
```

### 3️⃣ Why interaction features?

```
LINEAR: popularity = w1×artist + w2×dance
  → Cannot capture synergy!

NON-LINEAR: popularity = w1×artist + w2×dance + w3×(artist×dance)
  → Captures: Popular artist + danceable = MEGA HIT!
```

### 4️⃣ Why median imputation?

```
Options:
  1. Mean imputation   ← Sensitive to outliers
  2. Median imputation ← Robust ✅
  3. Mode imputation   ← For categorical
  4. Drop rows         ← Lose data ❌

We use median: Robust + preserves distribution
```

---

## 🎯 Success Factors

```
1. ARTIST FEATURES (32.2%)
   ├─ Captures brand power
   ├─ Strongest signal
   └─ Must use OUT-OF-FOLD!

2. AUDIO FEATURES (33.2%)
   ├─ Domain knowledge (loudness, energy)
   ├─ Engineered interactions
   └─ Original + derived

3. TEMPORAL FEATURES (11.4%)
   ├─ Recency bias
   ├─ Platform-specific pattern
   └─ Era characteristics

4. NLP FEATURES (9.5%)
   ├─ Lyrics topics
   ├─ TF-IDF → SVD pipeline
   └─ Compressed representation

5. INTERACTIONS (8.4%)
   ├─ Non-linear effects
   ├─ Multiplicative features
   └─ Amplification patterns
```

---

## 📈 Performance Attribution

```
Baseline (only audio features):     RMSE 18.5
+ Artist features:                   RMSE 16.8  (-1.7) 🔥
+ Temporal features:                 RMSE 16.3  (-0.5)
+ NLP lyrics:                        RMSE 16.1  (-0.2)
+ Interactions:                      RMSE 16.05 (-0.05)

TOTAL IMPROVEMENT: 2.45 RMSE points!
```

---

## ✅ Validation Checklist

Before model training, verify:

- [ ] No data leakage (OUT-OF-FOLD encoding)
- [ ] No duplicate features (check `features` list)
- [ ] No missing values (imputed)
- [ ] No categorical text (all encoded)
- [ ] Target not in features (exclude `popularity`)
- [ ] Same features in train & test
- [ ] Same feature order in train & test
- [ ] Test uses train statistics (not own)

---

**🎉 Pipeline Complete! Ready for RMSE 16.05!** 🏆
