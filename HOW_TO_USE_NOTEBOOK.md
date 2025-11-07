# 📖 How to Use: TPW_AhThatsHot_Siklus5_Complete.ipynb

## 🎯 Overview

`TPW_AhThatsHot_Siklus5_Complete.ipynb` adalah **Jupyter Notebook** yang berisi semua improvements Siklus 5 dalam format yang siap pakai. Tidak perlu manual cell splitting - cells sudah terpisah otomatis!

---

## 🚀 Quick Start

### Option 1: Google Colab (RECOMMENDED)

1. **Upload Notebook ke Colab:**
   - Buka https://colab.research.google.com/
   - Click **File → Upload notebook**
   - Select `TPW_AhThatsHat_Siklus5_Complete.ipynb`

2. **Upload Data:**
   ```python
   # Upload train.csv dan test.csv
   from google.colab import files
   uploaded = files.upload()
   ```
   Atau mount Google Drive jika data sudah ada di Drive.

3. **Adjust Paths (Cell 1):**
   ```python
   DATA_PATH = '/content'        # Lokasi train.csv & test.csv
   OUTPUT_PATH = './outputs'      # Lokasi output submission
   ```

4. **Run All Cells:**
   - Click **Runtime → Run all**
   - Atau run cell by cell (Shift+Enter)

5. **Download Submission:**
   ```python
   from google.colab import files
   files.download('outputs/submission_siklus5_complete.csv')
   ```

### Option 2: Jupyter Notebook (Local)

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
   ```

2. **Open Notebook:**
   ```bash
   jupyter notebook TPW_AhThatsHot_Siklus5_Complete.ipynb
   ```

3. **Adjust Paths (Cell 1):**
   ```python
   DATA_PATH = './data'           # Your local data folder
   OUTPUT_PATH = './outputs'
   ```

4. **Run All Cells:**
   - Kernel → Restart & Run All
   - Atau run cell by cell

5. **Get Submission:**
   - File akan tersimpan di `./outputs/submission_siklus5_complete.csv`

---

## 📚 Notebook Structure

### 12 Cells Total:

| Cell | Title | Duration | Description |
|------|-------|----------|-------------|
| **0** | **Introduction (Markdown)** | - | Overview & structure |
| **1** | **Imports & Configuration** | ~10s | Load libraries, set paths |
| **2** | **Feature Engineering Functions** | ~5s | Define helper functions |
| **3** | **CV Training Functions** | ~5s | Define training functions |
| **4** | **Load Data** | ~10s | Read train.csv & test.csv |
| **5** | **Comprehensive EDA** | ~30s | Analyze data (5 categories) |
| **6** | **Enhanced Feature Engineering** | ~20s | Create 23+ new features |
| **7** | **Process Lyrics (NLP)** | ~1-2m | TF-IDF + SVD processing |
| **8** | **Prepare Base Features** | ~10s | Encode categoricals |
| **9** | **Train Model (Proper CV)** | ~10-30m | **MAIN TRAINING** |
| **10** | **Train Final & Predict** | ~2-5m | Full training + predictions |
| **11** | **Create Submission** | ~5s | Save submission file |
| **12** | **Summary & Insights** | ~5s | Results & recommendations |

**Total Runtime:** ~15-40 minutes (mostly Cell 9)

---

## ⚙️ Customization Options

### 1. Change CV Folds

**Cell 9:**
```python
results = train_with_proper_cv(
    df_train,
    base_features,
    target_col='popularity',
    cv_folds=5  # ← Change to 3 for faster, 10 for more robust
)
```

### 2. Adjust Lyrics Components

**Cell 7:**
```python
svd = TruncatedSVD(n_components=20, random_state=42)  # ← Change to 10-50
```

Lower = faster, less info
Higher = slower, more info

### 3. Modify Model Hyperparameters

**Cell 3 (in `train_with_proper_cv` function):**
```python
lgbm_params = {
    'n_estimators': 2000,      # ← Max trees (higher = slower but better)
    'learning_rate': 0.01,     # ← Learning rate (lower = slower but more precise)
    'num_leaves': 31,          # ← Tree complexity (higher = more complex)
    'reg_alpha': 0.1,          # ← L1 regularization
    'reg_lambda': 0.1,         # ← L2 regularization
    # ... etc
}
```

### 4. Disable Lyrics Processing

**Cell 7:**
Add `and False` to skip:
```python
if 'lyrics' in df_train.columns and False:  # ← Will skip lyrics
```

---

## 🎯 What Each Cell Does

### Cell 0: Introduction (Markdown)
- Overview of improvements
- Notebook structure
- Quick reference

**Action:** Just read

---

### Cell 1: Imports & Configuration
```python
import pandas as pd
import numpy as np
# ... etc

DATA_PATH = '/content'        # ← ADJUST THIS
OUTPUT_PATH = './outputs'      # ← AND THIS
```

**Action:**
1. Review imports (install if missing)
2. **ADJUST PATHS** to your environment
3. Run cell

---

### Cell 2: Feature Engineering Functions

Defines 5 helper functions:
- `add_base_audio_features()` - Basic combinations
- `add_audio_ratios()` - **NEW:** 8+ ratio features
- `add_temporal_features()` - Year/decade/era
- `add_track_name_features()` - Track name analysis (9 features)
- `create_fold_features()` - **CRITICAL:** Prevents data leakage!

**Action:** Just run - no changes needed

---

### Cell 3: CV Training Functions

Defines `train_with_proper_cv()`:
- Main training function
- Target encoding INSIDE CV loop (no leakage!)
- Early stopping (100 rounds)
- Comprehensive logging

**Action:** Just run - no changes needed (unless tuning hyperparameters)

---

### Cell 4: Load Data

```python
df_train = pd.read_csv(Path(DATA_PATH) / 'train.csv')
df_test = pd.read_csv(Path(DATA_PATH) / 'test.csv')
```

Displays:
- Dataset shapes
- Target statistics
- Missing values

**Action:** Verify data loaded correctly

---

### Cell 5: Comprehensive EDA

5 analysis categories:
1. **Target Distribution** - Popularity bins
2. **Genre Analysis** - Top genres, counts, stats
3. **Artist Analysis** - Prolific artists, single-song artists
4. **Audio Correlations** - Feature correlations with target
5. **Temporal Analysis** - Popularity by decade

**Action:** Review insights, note patterns

---

### Cell 6: Enhanced Feature Engineering

Applies all feature functions:
- Base audio features (4 features)
- Audio ratios (8+ features)
- Temporal features (5+ features)
- Track name features (9 features)

**Output:** df_train & df_test dengan ~23 new features

**NOTE:** Genre & artist features created later in CV loop!

**Action:** Verify features added successfully

---

### Cell 7: Process Lyrics (NLP)

If lyrics column exists:
1. TF-IDF vectorization (max 500 features)
2. SVD dimensionality reduction (20 components)
3. Add lyrics_feature_0 to lyrics_feature_19

**Duration:** 1-2 minutes

**Action:** Check explained variance (should be >20%)

---

### Cell 8: Prepare Base Features

1. Encode categorical features (key_mode, tempo_category)
2. Create base feature list
3. Exclude non-feature columns

**Output:** `base_features` list (~50+ features)

**Action:** Verify feature count looks reasonable

---

### Cell 9: Train Model (Proper CV) ⚠️ MAIN CELL

**CRITICAL CELL - This is where the magic happens!**

- 5-Fold Cross-Validation
- Target encoding INSIDE CV loop (no leakage!)
- Early stopping (100 rounds)
- L1/L2 regularization
- Per-fold logging

**Duration:** 10-30 minutes (depending on data size)

**Output:**
- OOF predictions
- CV scores (per fold)
- Feature importance
- Trained models (5 models)

**Action:** Go get coffee ☕ - this takes a while!

---

### Cell 10: Train Final Model & Predict

1. Create features for full training data
2. Train final model on ALL training data
3. Prepare test set with same features
4. Make predictions
5. Clip to [0, 100] range

**Duration:** 2-5 minutes

**Output:** `predictions` array for test set

**Action:** Verify prediction statistics look reasonable

---

### Cell 11: Create Submission

```python
submission = pd.DataFrame({
    'track_id': df_test['track_id'],
    'popularity': predictions
})

submission.to_csv('outputs/submission_siklus5_complete.csv', index=False)
```

**Output:** `submission_siklus5_complete.csv`

**Action:** Download/copy submission file

---

### Cell 12: Summary & Insights

Displays:
- Final OOF RMSE, MAE, R²
- CV statistics
- Performance vs baseline
- Top 15 most important features
- Key takeaways
- Next steps recommendations

**Action:** Review results & celebrate! 🎉

---

## 🐛 Troubleshooting

### Issue 1: "No module named 'lightgbm'"

**Solution:**
```bash
# Colab
!pip install lightgbm

# Local
pip install lightgbm
```

### Issue 2: "FileNotFoundError: train.csv"

**Solution:** Adjust `DATA_PATH` in Cell 1:
```python
DATA_PATH = '/path/to/your/data/folder'
```

### Issue 3: "MemoryError" during lyrics processing

**Solution:** Reduce TF-IDF features in Cell 7:
```python
tfidf = TfidfVectorizer(
    max_features=200,  # ← Reduce from 500
    # ...
)
```

### Issue 4: Training takes too long

**Solutions:**
1. Reduce CV folds in Cell 9: `cv_folds=3`
2. Reduce estimators in Cell 3: `n_estimators=1000`
3. Skip lyrics processing (see customization above)

### Issue 5: Kernel dies / Out of memory

**Solutions:**
1. Use Google Colab (free GPU & more RAM)
2. Reduce data sample for testing
3. Reduce lyrics SVD components

### Issue 6: "KeyError: 'genre_avg_pop'" in train data

**This is NORMAL!** Genre & artist features are created INSIDE CV loop (Cell 9). Don't worry if you see warnings about these features not existing in base data.

---

## 📊 Expected Output

### Console Output Summary:

```
================================================================================
📦 IMPORTING LIBRARIES & CONFIGURATION
================================================================================
✅ All libraries imported successfully!
...

================================================================================
📂 LOADING DATA
================================================================================
✓ Train shape: (XXXX, XX)
✓ Test shape:  (XXXX, XX)
...

================================================================================
📊 COMPREHENSIVE EXPLORATORY DATA ANALYSIS
================================================================================
[1/5] Target Distribution Analysis...
[2/5] Genre Analysis...
...

================================================================================
🔧 ENHANCED FEATURE ENGINEERING
================================================================================
[1/4] Adding base audio features...
[2/4] Adding audio ratios (NEW)...
...

================================================================================
📝 PROCESSING LYRICS (NLP)
================================================================================
✓ Explained variance: XX.XX%
...

================================================================================
🤖 TRAINING MODEL WITH PROPER CROSS-VALIDATION
================================================================================
[Fold 1/5]
  Features: 67
  Best iteration: XXX
  RMSE: XX.XXXX
...

CROSS-VALIDATION RESULTS
OOF RMSE: XX.XXXX
OOF MAE:  XX.XXXX
OOF R²:   X.XXXX

Top 10 Most Important Features:
  1. artist_avg_pop           : XXXX.X (±XX.X)
  2. genre_avg_pop            : XXXX.X (±XX.X)
  ...

================================================================================
🎯 TRAINING FINAL MODEL & PREDICTING TEST SET
================================================================================
✓ Model trained successfully
✓ Predictions complete!
...

================================================================================
📤 CREATING SUBMISSION FILE
================================================================================
✅ Submission file saved!
   📁 Location: outputs/submission_siklus5_complete.csv
...

================================================================================
✅ PIPELINE COMPLETE - SUMMARY & INSIGHTS
================================================================================
🎯 FINAL RESULTS
  OOF RMSE: XX.XXXX
  OOF MAE:  XX.XXXX
  OOF R²:   X.XXXX
...

🎉 ALL DONE - SIKLUS 5 COMPLETE!
✨ Thank you for using this pipeline! ✨
```

### Files Generated:

- `outputs/submission_siklus5_complete.csv` - Your predictions!

---

## 📈 Performance Expectations

### Baseline (Siklus 4):
- RMSE: ~16.0-16.5
- Issues: Data leakage, missing features

### Current (Siklus 5):
- **Expected RMSE: 15.5-16.0**
- Improvements:
  - ✅ Fixed data leakage
  - ✅ 23+ new features
  - ✅ Better model config
  - ✅ Comprehensive insights

**Expected improvement: 0.3-0.5 RMSE vs baseline**

---

## 🎯 Quick Checklist

**Before running:**
- [ ] Notebook uploaded to Colab/Jupyter
- [ ] train.csv & test.csv uploaded/accessible
- [ ] DATA_PATH & OUTPUT_PATH adjusted (Cell 1)
- [ ] All libraries installed

**While running:**
- [ ] Cell 1: Libraries imported without errors
- [ ] Cell 4: Data loaded successfully (check shapes)
- [ ] Cell 5: EDA shows reasonable statistics
- [ ] Cell 9: CV training completes (~10-30 min)
- [ ] Cell 11: Submission file created

**After running:**
- [ ] Check OOF RMSE in Cell 12 (should be ~15.5-16.0)
- [ ] Review top features (Cell 12)
- [ ] Verify submission file exists
- [ ] Check predictions in range [0, 100]

---

## 💡 Tips

1. **First Run:** Just run as-is to see baseline results
2. **Second Run:** Experiment with hyperparameters (Cell 3)
3. **Third Run:** Try feature selection (remove low-importance features)
4. **Fourth Run:** Ensemble multiple runs

---

## 🔗 Related Files

- `Complete_Siklus5_SingleFile.py` - Python script version (needs manual cell splitting)
- `HOW_TO_USE_SINGLE_FILE.md` - Guide for Python script version
- `README_SIKLUS5_IMPROVED.md` - Complete technical documentation
- `SUMMARY_IMPROVEMENTS.md` - Detailed summary of all improvements

---

## 🆚 Notebook vs Python Script

### Use Notebook (.ipynb) if:
- ✅ You want ready-to-use cells (no manual splitting)
- ✅ You're using Google Colab
- ✅ You want interactive cell-by-cell execution
- ✅ You want to see outputs between cells

### Use Python Script (.py) if:
- ✅ You prefer command-line execution
- ✅ You want to integrate into a pipeline
- ✅ You're comfortable with manual cell splitting in Jupyter

---

## ❓ FAQ

### Q: How long does the entire notebook take to run?
**A:** 15-40 minutes total. Most time is spent in Cell 9 (training).

### Q: Can I skip certain cells?
**A:** NO! Cells must be run in order. Each cell depends on previous ones.

### Q: What if I get different RMSE each time?
**A:** Small variations are normal due to:
- Random shuffle in CV (though we set random_state=42)
- Early stopping may stop at different iterations
- Variations should be <0.01 RMSE

### Q: Can I use this notebook for other datasets?
**A:** Yes, but you'll need to:
1. Adjust column names in exclude_cols (Cell 8)
2. Modify feature functions for your data (Cell 2)
3. Change target_col if not 'popularity'

### Q: Why is data leakage prevention so important?
**A:** Without it:
- CV scores are overly optimistic
- Real performance will be worse
- Model won't generalize well
See `REVIEW_FEATURE_AND_DATA.md` for detailed explanation.

---

## 📞 Support

**For issues:**
1. Check Troubleshooting section above
2. Review `README_SIKLUS5_IMPROVED.md` for technical details
3. Check `IMPLEMENTATION_FIXES.md` for common problems

---

## 🎉 You're Ready!

**Just upload the notebook to Colab/Jupyter and run all cells!**

**Expected runtime:** 15-40 minutes total
**Expected RMSE:** ~15.5-16.0
**Expected improvement:** 0.3-0.5 RMSE vs baseline

Good luck! 🚀🎵

---

**Last Updated:** 2025-11-07
**Version:** Siklus 5 - Complete Notebook
**Format:** Jupyter Notebook (.ipynb)
**Status:** ✅ Production Ready
