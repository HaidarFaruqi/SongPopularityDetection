# 🚀 Quick Start: Song Popularity Predictor with Data Cleaning

## 📋 Apa yang Baru di Version 2.0?

### ✨ New Features

1. **Data Cleaning Automatic**
   - Fix invalid years (21 → 2021, 1 → 2001, 99 → 1999)
   - Handle popularity = 0 dengan flag feature
   - Validation checks

2. **Comprehensive Visualizations**
   - Data quality inspection (9 plots)
   - Before/After cleaning comparison (6 plots)
   - Detailed EDA visualizations (6 plots)
   - Feature engineering insights
   - Model performance analysis (15 plots)

3. **Insights at Every Step**
   - Anomaly detection reports
   - Statistical summaries
   - Pattern identification

---

## 🎯 Invalid Years Problem & Solution

### Problem yang Ditemukan

```python
# Data mentah punya years seperti:
21   # Seharusnya 2021
1    # Seharusnya 2001
99   # Seharusnya 1999
0    # Seharusnya 2000
```

### Cleaning Logic

```python
def fix_release_year(year):
    """
    Fix invalid release years

    Rules:
    - year < 25: assume 2000s (21 → 2021)
    - 25 <= year < 100: assume 1900s (99 → 1999)
    - year >= 1000: keep as is
    - Special: 0 → 2000
    """
    if year >= 1000:
        return year  # Already valid

    if year == 0:
        return 2000

    if year < 25:
        return 2000 + year  # 2000s
    elif year < 100:
        return 1900 + year  # 1900s
    else:
        return year
```

### Examples

| Before | After | Logic |
|--------|-------|-------|
| 21 | 2021 | < 25 → 2000s |
| 1 | 2001 | < 25 → 2000s |
| 24 | 2024 | < 25 → 2000s |
| 99 | 1999 | >= 25 & < 100 → 1900s |
| 80 | 1980 | >= 25 & < 100 → 1900s |
| 0 | 2000 | Special case |
| 2020 | 2020 | >= 1000 → no change |

### Impact

✅ **Temporal features now accurate**:
- `years_since_release`: Correct calculation
- `decade`: Proper grouping
- `is_recent_hit`: Correctly flagged
- `is_classic`: Properly identified

Expected RMSE improvement: **-0.1 to -0.3**

---

## 📊 Popularity = 0 Handling

### Analysis

```python
# Check how many songs have popularity = 0
zero_count = (train_df['popularity'] == 0).sum()
zero_pct = (zero_count / len(train_df)) * 100

# Example output:
# "Found 5,432 records (4.76%) with popularity = 0"
```

### Decision Framework

```
IF zero_pct > 10%:
    → Suspicious! Likely data quality issue
    → Strategy: Create flag + consider imputation

ELSE IF zero_pct < 5%:
    → Normal! Some songs genuinely unpopular
    → Strategy: Keep as-is

ELSE (5-10%):
    → Borderline
    → Strategy: Create flag feature
```

### Our Strategy

**Hybrid Approach** (best practice):

```python
# 1. Create flag feature (model dapat belajar pattern ini)
train_df['is_zero_popularity'] = (train_df['popularity'] == 0).astype(int)

# 2. Keep original value (preserve data integrity)
# NO imputation on target variable!
```

### Why This Works

✅ **Preserve data integrity**: No manipulation of target
✅ **Capture pattern**: Model learns "being unpopular" is a signal
✅ **Transparent**: Clear what we did
✅ **Flexible**: Can experiment with imputed version separately

Expected improvement: **-0.05 to -0.15 RMSE**

---

## 📈 Visualizations Included

### 1. Data Quality Inspection (9 plots)

```
Before cleaning visualization:
├── Release Year Distribution (with anomalies highlighted)
├── Invalid Years Detail (bar chart)
├── Popularity Distribution (with zeros highlighted)
├── Popularity = 0 by Genre
├── Missing Values Heatmap
├── Year vs Popularity Scatter
├── Popularity Boxplot
├── Top Genres
└── Summary Statistics Table
```

### 2. Before/After Cleaning (6 plots)

```
Comparison visualization:
├── Year Distribution BEFORE
├── Year Distribution AFTER
├── Year Range Comparison
├── Year vs Popularity BEFORE
├── Year vs Popularity AFTER
└── Cleaning Impact Summary Table
```

### 3. EDA Visualizations (6 plots)

```
Detailed EDA:
├── Target Distribution (with mean/median)
├── Audio Features Correlation
├── Top 15 Genres by Popularity
├── Popularity Trend Over Years
├── Song Duration Distribution
└── Popularity by Decade
```

### 4. Model Performance (15 plots from original)

```
Comprehensive analysis:
├── CV Scores by Fold
├── Model Performance Metrics
├── Top 25 Feature Importance
├── Actual vs Predicted
├── Residuals Distribution
├── Residual Plot
├── RMSE by Prediction Range
├── Distribution Comparison
├── Q-Q Plot
├── RMSE by Popularity Range
├── Feature Category Importance
├── Worst Predictions Highlighted
├── Error Distribution Boxplot
├── Top 10 Features Detail
└── Summary Statistics Table
```

**Total: 36 visualizations!** 📊

---

## 🔧 How to Use

### Option 1: Python Script (v2.py)

```python
from song_popularity_predictor_v2 import SongPopularityPredictorV2

# Initialize
predictor = SongPopularityPredictorV2(data_path='.', output_path='./outputs')

# Pipeline with visualizations
predictor.load_data()
predictor.visualize_data_quality()  # 9 plots
predictor.clean_data()
predictor.visualize_after_cleaning()  # 6 plots
predictor.eda()  # 6 plots + console output
predictor.engineer_features()
predictor.process_lyrics()
predictor.prepare_features()
predictor.train_models()
# ... rest of pipeline
```

### Option 2: Jupyter Notebook (recommended)

```bash
jupyter notebook song_popularity_predictor_enhanced.ipynb
```

Benefits:
- ✅ Interactive execution
- ✅ See plots inline
- ✅ Easy to modify
- ✅ Can save outputs

---

## 📝 Validation Checks

After cleaning, the script automatically validates:

```python
# Validation checks
assert train_df['release_year'].min() >= 1000, "Still has invalid years!"
assert train_df['release_year'].max() <= 2025, "Future years detected!"
assert train_df['popularity'].min() >= 0, "Negative popularity!"
assert train_df['popularity'].max() <= 100, "Popularity > 100!"

print("✓ All validation checks passed!")
```

---

## 📊 Expected Results

### Data Quality

**Before Cleaning**:
```
Invalid Years: 180 records
Year Range: [1, 2024]  ← Invalid!
Suspicious patterns in temporal features
```

**After Cleaning**:
```
Invalid Years: 0 records
Year Range: [1955, 2024]  ← Valid!
Clean temporal features
```

### Model Performance

**Without Cleaning** (estimated):
```
RMSE: 16.3
MAE:  12.5
R²:   0.71
```

**With Cleaning** (estimated):
```
RMSE: 16.0-16.1  (↓ 0.2-0.3)
MAE:  12.2-12.3  (↓ 0.2-0.3)
R²:   0.73-0.74  (↑ 0.02-0.03)
```

**Improvement**: ~1.5-2% better performance! 🎯

---

## 🎯 Key Takeaways

1. **Always inspect data first**
   - Run `visualize_data_quality()` before anything
   - Identify anomalies early

2. **Clean systematically**
   - Document every change
   - Keep before/after comparisons
   - Run validation checks

3. **Visualize everything**
   - Pictures worth 1000 words
   - Easier to spot issues
   - Better communication with stakeholders

4. **Preserve integrity**
   - Don't delete data without reason
   - Create flags instead of imputing targets
   - Be transparent about changes

---

## 📚 Files Generated

```
outputs/
├── data_quality_inspection.png     # Before cleaning (9 plots)
├── before_after_cleaning.png       # Cleaning comparison (6 plots)
├── eda_visualizations.png          # Detailed EDA (6 plots)
├── siklus4_comprehensive_analysis.png  # Model analysis (15 plots)
└── submission_siklus4_enhanced.csv     # Final predictions
```

---

## 💡 Pro Tips

1. **Always backup original data**
   ```python
   train_df_original = train_df.copy()
   ```

2. **Document cleaning decisions**
   ```python
   cleaning_stats = {
       'invalid_years_fixed': 180,
       'zero_popularity_count': 5432,
       'strategy': 'flag_feature_approach'
   }
   ```

3. **Visual validation**
   - Check histograms before/after
   - Scatter plots for relationships
   - Boxplots for outliers

4. **Incremental improvements**
   - Clean one issue at a time
   - Measure impact
   - Iterate

---

## 🚀 Next Steps

1. Run data quality inspection
2. Clean data dengan strategy yang explained
3. Re-engineer features (now with clean data!)
4. Train model
5. Compare with previous results
6. Iterate!

**Happy cleaning & modeling!** 🎵✨
