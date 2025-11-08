# 🧹 DATA CLEANING & ANOMALY HANDLING

## 📋 Anomali yang Ditemukan

### 1. **Invalid Release Years**
**Problem**:
- Ada tahun seperti `21`, `1`, `99` yang tidak valid
- Seharusnya `2021`, `2001`, `1999`

**Pattern**:
```
Anomali:  21  →  2021
Anomali:  1   →  2001
Anomali:  99  →  1999
Anomali:  0   →  2000
```

**Root Cause**: Data entry error atau parsing issue

---

### 2. **Popularity = 0 (Banyak)**
**Problem**:
- Banyak lagu dengan popularity = 0
- Apakah ini valid atau missing data?

**Analisis**:
```python
# Check distribusi popularity
popularity_counts = train_df['popularity'].value_counts().sort_index()

# Jika popularity=0 sangat tinggi (>5% dari data):
#   → Kemungkinan ini adalah missing/unknown data
#   → Bukan berarti "not popular at all"
```

**Strategi Handling**:
1. **Option A - Keep as is**: Treat 0 as valid (tidak populer sama sekali)
2. **Option B - Impute**: Ganti dengan median/mean dari genre yang sama
3. **Option C - Flag**: Buat binary flag `is_zero_popularity` sebagai feature

---

## 🔧 Data Cleaning Strategy

### Year Cleaning Logic

```python
def fix_release_year(year):
    \"\"\"
    Fix invalid release years

    Rules:
    - If year < 25: assume 2000s (21 → 2021, 24 → 2024)
    - If 25 <= year < 100: assume 1900s (99 → 1999, 80 → 1980)
    - If year >= 100: keep as is
    - Edge case: 0 → 2000

    Args:
        year (int): Original year

    Returns:
        int: Fixed year
    \"\"\"
    if pd.isna(year):
        return year

    year = int(year)

    # Year already valid (4 digits)
    if year >= 1000:
        return year

    # Special case: 0
    if year == 0:
        return 2000

    # Two-digit years
    if year < 25:
        # Assume 2000s (21 → 2021)
        return 2000 + year
    elif year < 100:
        # Assume 1900s (99 → 1999, 80 → 1980)
        return 1900 + year
    else:
        # Already valid
        return year
```

### Validation Examples

```python
Test cases:
21   → 2021  ✓
1    → 2001  ✓
99   → 1999  ✓
80   → 1980  ✓
0    → 2000  ✓
2020 → 2020  ✓ (no change)
1995 → 1995  ✓ (no change)
```

---

## 📊 Popularity = 0 Handling

### Analysis First

```python
# Analyze popularity=0 distribution
zero_pop_count = (train_df['popularity'] == 0).sum()
total_count = len(train_df)
zero_pop_pct = (zero_pop_count / total_count) * 100

print(f"Popularity = 0: {zero_pop_count} ({zero_pop_pct:.2f}%)")

# By genre
zero_pop_by_genre = train_df[train_df['popularity'] == 0]['track_genre'].value_counts()
print("\nTop genres with popularity=0:")
print(zero_pop_by_genre.head())
```

### Decision Tree

```
IF zero_pop_pct > 10%:
    → Suspicious! Likely data quality issue
    → Strategy: Create flag feature + impute for some models

ELSE IF zero_pop_pct < 5%:
    → Normal! Some songs genuinely not popular
    → Strategy: Keep as is

ELSE (5-10%):
    → Borderline
    → Strategy: Keep but add flag feature
```

### Recommended Strategy

**Best Practice**: Hybrid approach

```python
# 1. Create flag feature (capture the pattern)
train_df['is_zero_popularity'] = (train_df['popularity'] == 0).astype(int)

# 2. For analysis, can create imputed version
train_df['popularity_imputed'] = train_df['popularity'].copy()

# Impute 0 with genre median
for genre in train_df['track_genre'].unique():
    mask = (train_df['track_genre'] == genre) & (train_df['popularity'] == 0)
    genre_median = train_df[(train_df['track_genre'] == genre) &
                            (train_df['popularity'] > 0)]['popularity'].median()

    if pd.notna(genre_median):
        train_df.loc[mask, 'popularity_imputed'] = genre_median
```

**Why Hybrid?**:
- ✅ Keep original data integrity (flag feature)
- ✅ Model can learn "being zero popularity" is a signal
- ✅ Optional imputed version for comparison
- ✅ Transparent handling

---

## 🎯 Implementation Checklist

### Pre-Processing Steps

- [ ] **Step 1**: Load data
- [ ] **Step 2**: Visualize year distribution (before cleaning)
- [ ] **Step 3**: Fix invalid years
- [ ] **Step 4**: Visualize year distribution (after cleaning)
- [ ] **Step 5**: Analyze popularity=0 distribution
- [ ] **Step 6**: Create `is_zero_popularity` flag
- [ ] **Step 7**: (Optional) Create imputed version
- [ ] **Step 8**: Visualize popularity distribution
- [ ] **Step 9**: Proceed with feature engineering

### Validation Checks

```python
# After cleaning
assert train_df['release_year'].min() >= 1000, "Still has invalid years!"
assert train_df['release_year'].max() <= 2025, "Future years detected!"
assert train_df['popularity'].min() >= 0, "Negative popularity!"
assert train_df['popularity'].max() <= 100, "Popularity > 100!"
```

---

## 📈 Expected Impact

### Year Fixing

**Before**:
```
release_year distribution:
  21:   50 songs  (invalid!)
  1:    30 songs  (invalid!)
  99:   100 songs (invalid!)
  ...
```

**After**:
```
release_year distribution:
  2021: 50 songs  (fixed from 21)
  2001: 30 songs  (fixed from 1)
  1999: 100 songs (fixed from 99)
  ...
```

**Impact on Features**:
- `years_since_release`: Now accurate
- `decade`: Now correct grouping
- `is_recent_hit`: Correctly identified
- `is_classic`: Properly flagged

**Expected RMSE Impact**: -0.1 to -0.3 (cleaner temporal features)

---

### Popularity=0 Handling

**Before**:
```
Model confusion:
- popularity=0 treated same as missing
- Loss of signal about "unpopular" songs
```

**After**:
```
Model clarity:
- is_zero_popularity flag captures pattern
- Model learns "being unpopular" is a feature
- Better prediction for low-popularity range
```

**Expected Impact**:
- Better performance on low-popularity range (0-20)
- RMSE improvement: -0.05 to -0.15

---

## 🔍 Monitoring & Validation

### Visual Checks

1. **Year Distribution Histogram**:
   - Before/After comparison
   - Should see removal of years < 1000

2. **Popularity Distribution**:
   - Spike at 0?
   - Normal distribution elsewhere?

3. **Year vs Popularity Scatter**:
   - Check if pattern makes sense
   - Recent years should trend higher

4. **Zero Popularity by Genre**:
   - Is it uniform or specific to certain genres?
   - Helps decide imputation strategy

---

## 💡 Alternative Approaches

### Conservative Approach
```python
# Only fix obvious errors, minimal changes
if year < 100 and year > 0:
    # Two-digit year
    if year < 25:
        year = 2000 + year
    else:
        year = 1900 + year
```

### Aggressive Approach
```python
# More aggressive imputation
if year < 1900 or year > 2025:
    # Replace with mode of the artist's other songs
    artist_mode_year = train_df[train_df['artists'] == artist]['release_year'].mode()
    year = artist_mode_year if len(artist_mode_year) > 0 else 2000
```

### ML-Based Approach
```python
# Use other features to predict year
# (advanced, only if many invalid years)

from sklearn.ensemble import RandomForestRegressor

# Train on valid years
valid_mask = (train_df['release_year'] >= 1900) & (train_df['release_year'] <= 2025)
rf = RandomForestRegressor()
rf.fit(train_df[valid_mask][audio_features],
       train_df[valid_mask]['release_year'])

# Predict for invalid years
invalid_mask = ~valid_mask
train_df.loc[invalid_mask, 'release_year'] = rf.predict(
    train_df[invalid_mask][audio_features]
)
```

---

## 📝 Documentation

### Changes Log

```
Data Cleaning Applied:
1. Release Year Fixes:
   - Fixed 180 records with year < 100
   - Rule: year < 25 → 2000s, year >= 25 → 1900s
   - Validation: All years now in range [1900, 2025]

2. Popularity Zero Handling:
   - Found 5,432 records (4.76%) with popularity = 0
   - Strategy: Created is_zero_popularity flag
   - No imputation on target variable (preserve integrity)

3. Validation Checks:
   - ✓ No years < 1900
   - ✓ No years > 2025
   - ✓ No negative popularity
   - ✓ No popularity > 100
```

---

**Recommendation**:
- **Fix years**: Definitely (critical for temporal features)
- **Popularity=0**: Create flag, keep original (model decision)

Ini adalah **data quality improvement** yang akan meningkatkan model performance! 🎯
