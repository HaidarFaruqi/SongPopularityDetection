# 📘 Cara Menambahkan Insights ke Notebook

## 🎯 Overview

Guide ini menjelaskan cara menambahkan **visualization insights** dan **automatic feature selection** ke notebook Anda menggunakan utility modules yang sudah dibuat.

## ✅ Files yang Digunakan

1. **`visualization_utils.py`** - Visualization functions
2. **`feature_selection_utils.py`** - Feature selection functions
3. **`TPW_AhThatsHot_Siklus5_Complete.ipynb`** - Notebook Anda

## 📋 Steps to Integrate

### Step 1: Import Utilities di Cell 1

Tambahkan di Cell 1 (setelah imports yang ada):

```python
# === CELL 1: IMPORTS & CONFIGURATION ===
# ... (imports yang sudah ada)

# Import visualization & feature selection utilities
from visualization_utils import (
    data_profiling,
    create_preprocessing_insights,
    create_postprocessing_insights
)

from feature_selection_utils import (
    automatic_feature_selection,
    print_feature_selection_impact
)

print("✅ Utility modules loaded!")
```

### Step 2: Tambah Data Profiling (AFTER Cell 4: Load Data)

Tambahkan cell baru setelah load data:

```python
# === NEW CELL: DATA PROFILING ===
print("\n" + "="*80)
print("📊 DATA PROFILING & INITIAL DETECTION")
print("="*80)

# Profile training data
train_missing = data_profiling(df_train, "Training Data")

# Profile test data
test_missing = data_profiling(df_test, "Test Data")

print("\n✅ Data profiling complete!")
```

**Expected output:**
- Basic information (shape, memory)
- Data types breakdown
- Missing values analysis
- Duplicates check
- Potential issues

### Step 3: Tambah Pre-processing Insights (AFTER Cell 5: EDA)

Ganti atau tambahkan setelah EDA:

```python
# === UPDATED CELL: COMPREHENSIVE EDA WITH VISUALIZATION ===
print("\n" + "="*80)
print("📊 COMPREHENSIVE EDA WITH VISUALIZATION")
print("="*80)

# Create comprehensive visualizations
top_correlations = create_preprocessing_insights(
    df_train,
    target_col='popularity',
    output_path=OUTPUT_PATH
)

# Show text-based insights
print("\n[Additional Text Insights]")
print(f"  Shape: {df_train.shape}")
print(f"  Genres: {df_train['track_genre'].nunique()}")
print(f"  Artists: {df_train['artists'].nunique()}")
print(f"  Year range: {df_train['release_year'].min()} - {df_train['release_year'].max()}")

print("\n✅ Pre-processing insights complete!")
print(f"   Visualization saved: {OUTPUT_PATH}/preprocessing_insights.png")
```

**Expected output:**
- 9 subplot visualization saved as PNG
- Correlation heatmap
- Distribution plots
- Genre/temporal analysis
- Key insights summary

### Step 4: Tambah Feature Selection (AFTER Cell 8: Prepare Base Features)

Tambahkan cell baru setelah prepare base features:

```python
# === NEW CELL: AUTOMATIC FEATURE SELECTION ===
print("\n" + "="*80)
print("🎯 AUTOMATIC FEATURE SELECTION")
print("="*80)

# Save initial count
initial_feature_count = len(base_features)
print(f"\nInitial base features: {initial_feature_count}")

# Perform automatic feature selection
selected_features, feature_importance_df, all_features_used = automatic_feature_selection(
    df_train=df_train,
    base_features=base_features,
    target_col='popularity',
    target_num_features=50,      # Target: keep ~50 features
    importance_threshold=0.5,     # Keep features with >0.5% importance
    create_fold_features_func=create_fold_features,  # From Cell 2
    output_path=OUTPUT_PATH
)

# Print impact
print_feature_selection_impact(initial_feature_count, len(selected_features))

# Update base_features with selected features
base_features = selected_features

print(f"\n✅ Feature selection complete!")
print(f"   Updated base_features: {len(base_features)} features")
print(f"   Visualization saved: {OUTPUT_PATH}/feature_selection.png")
```

**Expected output:**
- Quick 3-fold CV for importance analysis
- Feature selection using multiple methods
- Dropped vs kept features list
- Visualization with cumulative importance
- Updated `base_features` variable

### Step 5: Tambah Post-processing Insights (AFTER Cell 9: Training)

Tambahkan cell baru setelah training:

```python
# === NEW CELL: POST-PROCESSING INSIGHTS ===
print("\n" + "="*80)
print("📈 POST-PROCESSING INSIGHTS & ANALYSIS")
print("="*80)

# Create comprehensive post-processing visualizations
create_postprocessing_insights(
    results=results,
    df_train=df_train,
    oof_predictions=oof_predictions,
    output_path=OUTPUT_PATH
)

print("\n✅ Post-processing insights complete!")
print(f"   Visualization saved: {OUTPUT_PATH}/postprocessing_insights.png")
```

**Expected output:**
- 9 subplot visualization saved as PNG
- Feature importance plots
- CV scores analysis
- Prediction vs actual scatter
- Residual analysis
- Error by prediction range

### Step 6: Update Final Summary (UPDATE Cell 12)

Update cell terakhir untuk include feature reduction info:

```python
# === UPDATED CELL 12: SUMMARY & INSIGHTS ===
print("\n" + "="*80)
print("✅ PIPELINE COMPLETE - SUMMARY & INSIGHTS")
print("="*80)

# ... (existing summary code)

# Add feature reduction summary
print(f"\n" + "="*80)
print("📊 FEATURE ENGINEERING SUMMARY")
print(f"="*80)
print(f"\nInitial base features: {initial_feature_count}")  # From Cell: Feature Selection
print(f"Final base features:   {len(base_features)}")
print(f"Reduction:             {initial_feature_count - len(base_features)} features")
print(f"Percentage:            {(initial_feature_count - len(base_features))/initial_feature_count*100:.1f}%")

# ... (rest of summary)

print(f"\n📁 Generated Files:")
print(f"  1. {OUTPUT_PATH}/preprocessing_insights.png")
print(f"  2. {OUTPUT_PATH}/feature_selection.png")
print(f"  3. {OUTPUT_PATH}/postprocessing_insights.png")
print(f"  4. {OUTPUT_PATH}/submission_siklus5_complete.csv")

print("\n✨ Check OUTPUT_PATH for all visualizations!")
```

---

## 📊 Expected Output Structure

After running enhanced notebook:

```
outputs/
├── preprocessing_insights.png      # 9 plots: correlations, distributions, etc.
├── feature_selection.png           # 2 plots: top features, cumulative importance
├── postprocessing_insights.png     # 9 plots: importance, CV scores, residuals
└── submission_siklus5_complete.csv # Final submission
```

---

## 🎯 Benefits Summary

| Stage | Before | After |
|-------|--------|-------|
| **Load Data** | Basic shape check | Full data profiling + issues detection |
| **EDA** | Text output only | 9 visual plots + insights |
| **Features** | 67 features (all) | 40-50 features (optimized) |
| **Training** | Just RMSE output | 9 visual plots + detailed metrics |
| **Final** | Basic summary | Comprehensive with feature reduction tracking |

---

## 📝 Alur Lengkap (Enhanced)

```
1. Load Data
   └─→ Data Profiling (NEW!)
       └─→ Missing values, duplicates, issues

2. EDA
   └─→ Pre-processing Insights (ENHANCED!)
       └─→ 9 visualization plots
       └─→ Correlation heatmap
       └─→ Distribution analysis

3. Feature Engineering
   └─→ Create all features (23+ new)

4. Prepare Base Features
   └─→ Automatic Feature Selection (NEW!)
       └─→ Quick CV for importance
       └─→ Select top 40-50 features
       └─→ Remove low-importance features

5. Train Model
   └─→ Proper CV with selected features
   └─→ Post-processing Insights (NEW!)
       └─→ 9 visualization plots
       └─→ Feature importance
       └─→ Residual analysis

6. Final Summary
   └─→ Feature reduction metrics
   └─→ Performance comparison
   └─→ File locations
```

---

## ⚙️ Customization Options

### Adjust Feature Selection Target

```python
# More aggressive reduction (keep fewer features)
selected_features, _, _ = automatic_feature_selection(
    # ...
    target_num_features=40,      # Instead of 50
    importance_threshold=1.0,     # Instead of 0.5
)

# Less aggressive (keep more features)
selected_features, _, _ = automatic_feature_selection(
    # ...
    target_num_features=60,
    importance_threshold=0.3,
)
```

### Change Visualization Output Path

```python
# Save to different folder
OUTPUT_PATH = './my_visualizations'

create_preprocessing_insights(
    df_train,
    target_col='popularity',
    output_path=OUTPUT_PATH  # Custom path
)
```

### Skip Certain Visualizations

```python
# Skip pre-processing if you want
# create_preprocessing_insights(...)  # Comment this out

# But keep post-processing
create_postprocessing_insights(...)  # Keep this
```

---

## 🐛 Troubleshooting

### Error: "module not found"

**Problem:** `visualization_utils` or `feature_selection_utils` not found

**Solution:**
```python
# Make sure files are in same directory as notebook
import sys
sys.path.append('/path/to/SongPopularityDetection/')

from visualization_utils import data_profiling
```

### Error: "create_fold_features not defined"

**Problem:** `create_fold_features` function not available for feature selection

**Solution:**
```python
# Make sure you define create_fold_features in Cell 2
# Then pass it to automatic_feature_selection:

selected_features, _, _ = automatic_feature_selection(
    # ...
    create_fold_features_func=create_fold_features,  # Pass the function!
)
```

### Visualizations not saving

**Problem:** Output directory doesn't exist

**Solution:**
```python
# Create directory first
from pathlib import Path
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
```

### Feature selection takes too long

**Problem:** Large dataset, 3-fold CV still slow

**Solution:**
```python
# Use smaller sample for feature selection
df_sample = df_train.sample(n=10000, random_state=42)

selected_features, _, _ = automatic_feature_selection(
    df_train=df_sample,  # Use sample instead
    # ...
)
```

---

## 📊 Expected Performance Impact

| Metric | Before (67 features) | After (40-50 features) | Change |
|--------|---------------------|------------------------|--------|
| **Training time** | ~20-30 min | ~15-20 min | -25% |
| **OOF RMSE** | ~15.82 | ~15.75-15.85 | Similar or better |
| **CV Stability** | Std ~0.035 | Std ~0.025 | More stable |
| **Overfitting risk** | Moderate | Lower | Better! |

**Key benefit:** Fewer features = less overfitting = better generalization!

---

## ✅ Checklist

Before running enhanced notebook:

- [ ] `visualization_utils.py` in same directory as notebook
- [ ] `feature_selection_utils.py` in same directory as notebook
- [ ] Updated Cell 1 with imports
- [ ] Added Data Profiling cell after Load Data
- [ ] Updated EDA cell with visualizations
- [ ] Added Feature Selection cell after Prepare Features
- [ ] Added Post-processing Insights after Training
- [ ] Updated Final Summary with feature reduction
- [ ] `OUTPUT_PATH` directory exists or will be created

After running:

- [ ] Check `preprocessing_insights.png` - looks good?
- [ ] Check `feature_selection.png` - ~40-50 features selected?
- [ ] Check `postprocessing_insights.png` - CV stable?
- [ ] OOF RMSE similar or better than before?
- [ ] All features in `base_features` are actually important?

---

## 🎓 Tips untuk Mahasiswa

1. **Jalankan sequential** - Jangan skip cells
2. **Check visualizations** - Lihat PNG files di OUTPUT_PATH
3. **Analyze feature selection** - Apakah dropped features masuk akal?
4. **Compare RMSE** - Apakah feature reduction improve atau maintain performa?
5. **Iterasi** - Coba adjust `target_num_features` kalau perlu

**Expected workflow time:**
- First run: ~30-40 min (with feature selection)
- Subsequent runs: ~20-25 min (after features selected)

---

**Ready to enhance your notebook!** 🚀

Copy-paste cells di atas ke notebook Anda, adjust paths kalau perlu, dan run!
