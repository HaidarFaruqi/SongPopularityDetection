# 📖 How to Use: Complete_Siklus5_SingleFile.py

## 🎯 Overview

File `Complete_Siklus5_SingleFile.py` adalah versi **LENGKAP** dari semua improvements yang sudah dibuat, digabungkan dalam **SATU FILE** yang bisa langsung digunakan di Jupyter Notebook atau Google Colab.

---

## 🚀 Option 1: Copy-Paste ke Jupyter/Colab (RECOMMENDED)

### Step-by-Step:

#### 1. Buka File
```bash
cat Complete_Siklus5_SingleFile.py
```
Atau buka dengan text editor favorit Anda.

#### 2. Copy Seluruh Isi File
- Select all (Ctrl+A atau Cmd+A)
- Copy (Ctrl+C atau Cmd+C)

#### 3. Buka Jupyter/Colab
- Jupyter: Buat new notebook
- Colab: https://colab.research.google.com/ → New notebook

#### 4. Paste & Split ke Cells

File sudah dibagi dengan marker `# === CELL X ===`

**Cara Split:**

**Option A: Manual (Simple)**
1. Paste semua kode ke satu cell
2. Cari marker `# === CELL 2 ===`
3. Letakkan cursor sebelum marker
4. Tekan `Ctrl+Shift+-` (atau menu: Edit → Split Cell)
5. Ulangi untuk semua marker `# === CELL X ===`

**Option B: Automatic (Advanced)**
Gunakan script helper berikut di cell pertama:

```python
# Helper untuk split cells
code = """
# PASTE ALL CODE HERE
"""

cells = code.split('# === CELL')
for i, cell in enumerate(cells):
    if cell.strip():
        print(f"=== CELL {i} ===")
        print(cell[:100])  # Preview
```

#### 5. Run Cells Secara Berurutan
- Cell 1: Introduction (markdown)
- Cell 2: Imports
- Cell 3: Helper Functions - Feature Engineering
- Cell 4: Helper Functions - CV Training
- Cell 5: Load Data
- Cell 6: EDA
- Cell 7: Feature Engineering
- Cell 8: Process Lyrics
- Cell 9: Prepare Features
- Cell 10: Train Model
- Cell 11: Train Final & Predict
- Cell 12: Create Submission
- Cell 13: Summary

**IMPORTANT:** Jalankan cell by cell, jangan skip!

---

## 🔧 Option 2: Run Langsung sebagai Python Script

### Prerequisites:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
```

### Adjust Paths:
Edit baris 21-22 di file:
```python
DATA_PATH = '/content'        # Change to your data folder
OUTPUT_PATH = './outputs'      # Change to your output folder
```

### Run:
```bash
python Complete_Siklus5_SingleFile.py
```

---

## 📊 Cell Structure & What Each Cell Does

### Cell 1: Introduction (Markdown)
- Overview of improvements
- Expected results
- File structure explanation

### Cell 2: Imports & Configuration
- Import all libraries
- Set visualization configs
- Set data paths
- Create output directory

**Action:** Review paths, adjust if needed

### Cell 3: Helper Functions - Feature Engineering
- `add_base_audio_features()` - Basic audio combinations
- `add_audio_ratios()` - NEW: Energy/valence ratios, etc.
- `add_temporal_features()` - Year, decade, era features
- `add_track_name_features()` - Track name analysis
- `create_fold_features()` - **CRITICAL: Prevents data leakage!**

**Action:** Just run, no changes needed

### Cell 4: Helper Functions - CV Training
- `train_with_proper_cv()` - Main training function
  - Implements proper CV (no leakage)
  - Early stopping
  - Regularization
  - Comprehensive logging

**Action:** Just run, no changes needed

### Cell 5: Load Data
- Load train.csv & test.csv
- Display shapes
- Show target statistics
- Check missing values

**Action:** Verify data loaded correctly

### Cell 6: Comprehensive EDA
1. Target distribution analysis
2. Genre analysis
3. Artist analysis
4. Audio correlations
5. Temporal trends

**Action:** Review insights, note any anomalies

### Cell 7: Enhanced Feature Engineering
- Applies all feature engineering functions
- Creates 23+ new features
- **NOTE:** Genre & artist features created later in CV loop

**Output:** df_train & df_test dengan fitur baru

### Cell 8: Process Lyrics (NLP)
- TF-IDF vectorization
- SVD dimensionality reduction
- Creates 20 lyrics features

**Action:** Check explained variance (should be >20%)

### Cell 9: Prepare Base Features
- Encode categorical features
- Create feature list
- Exclude non-feature columns

**Output:** `base_features` list

### Cell 10: Train Model (PROPER CV)
- **CRITICAL CELL:** Training dengan proper CV
- Target encoding INSIDE CV loop (no leakage!)
- Early stopping
- Comprehensive logging per fold

**Output:**
- `results` dictionary
- OOF predictions
- CV scores
- Feature importance

**Duration:** 10-30 minutes (depending on data size)

### Cell 11: Train Final Model & Predict
- Train on full training data
- Prepare test data with same features
- Make predictions
- Clip to [0, 100] range

**Output:** `predictions` array

### Cell 12: Create Submission
- Create submission DataFrame
- Save to CSV
- Display sample predictions

**Output:** `submission_siklus5_complete.csv`

### Cell 13: Summary & Insights
- Display final results
- CV statistics
- Top features
- Key takeaways
- Next steps

**Action:** Review and celebrate! 🎉

---

## ⚙️ Customization Options

### 1. Change CV Folds
In Cell 10, line:
```python
results = train_with_proper_cv(
    df_train,
    base_features,
    target_col='popularity',
    cv_folds=5  # ← Change this (default: 5)
)
```

### 2. Change Lyrics Components
In Cell 8:
```python
svd = TruncatedSVD(n_components=20, random_state=42)  # ← Change n_components
```

### 3. Change Model Hyperparameters
In Cell 4, function `train_with_proper_cv()`:
```python
lgbm_params = {
    'n_estimators': 2000,      # ← Adjust these
    'learning_rate': 0.01,
    'num_leaves': 31,
    # ... etc
}
```

### 4. Disable Lyrics Processing
In Cell 8, change:
```python
if 'lyrics' in df_train.columns and False:  # ← Add 'and False'
```

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'lightgbm'"
**Solution:**
```bash
pip install lightgbm
```

### Issue 2: "FileNotFoundError: train.csv not found"
**Solution:**
Adjust `DATA_PATH` in Cell 2:
```python
DATA_PATH = '/path/to/your/data'  # Folder containing train.csv & test.csv
```

### Issue 3: "MemoryError" during lyrics processing
**Solution:**
Reduce TF-IDF features in Cell 8:
```python
tfidf = TfidfVectorizer(
    max_features=200,  # ← Reduce from 500
    # ...
)
```

### Issue 4: Training takes too long
**Solutions:**
- Reduce CV folds (Cell 10): `cv_folds=3`
- Reduce estimators (Cell 4): `n_estimators=1000`
- Use smaller data sample for testing

### Issue 5: "KeyError: 'genre_avg_pop'" in train data
**This is NORMAL!** Genre & artist features dibuat INSIDE CV loop untuk prevent leakage.

Don't worry if you see this warning - it's intentional design.

---

## 📊 Expected Output

### Console Output:
```
================================================================================
📦 CELL 2: IMPORTING LIBRARIES & CONFIGURATION
================================================================================
✅ All libraries imported successfully!
...

================================================================================
📂 CELL 5: LOADING DATA
================================================================================
✓ Train: (XXXX, XX)
✓ Test:  (XXXX, XX)
...

================================================================================
🤖 CELL 10: TRAINING MODEL (PROPER CV - NO LEAKAGE)
================================================================================
[Fold 1/5]
  Features: 67
  Best iteration: XXX
  RMSE: XX.XXXX
...

CROSS-VALIDATION RESULTS
OOF RMSE: XX.XXXX
...

================================================================================
✅ CELL 13: PIPELINE COMPLETE - SUMMARY & INSIGHTS
================================================================================
🎯 FINAL RESULTS:
   OOF RMSE: XX.XXXX
...
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

---

## 🎯 Quick Checklist

Before running:
- [ ] Installed all required libraries
- [ ] Adjusted DATA_PATH to your data folder
- [ ] Adjusted OUTPUT_PATH for submission file
- [ ] Verified train.csv & test.csv exist

While running:
- [ ] Cell 2: Libraries imported without errors
- [ ] Cell 5: Data loaded successfully
- [ ] Cell 6: EDA shows reasonable stats
- [ ] Cell 10: CV training completes (10-30 min)
- [ ] Cell 12: Submission file created

After running:
- [ ] Check OOF RMSE in Cell 13
- [ ] Review top features
- [ ] Verify submission file exists
- [ ] Predictions in range [0, 100]

---

## 💡 Tips

1. **First Run:** Just run as-is to see baseline results
2. **Second Run:** Experiment with hyperparameters
3. **Third Run:** Try feature selection (remove low-importance)
4. **Fourth Run:** Ensemble multiple runs

---

## 📚 Additional Resources

- Full Review: `REVIEW_FEATURE_AND_DATA.md`
- Implementation Guide: `IMPLEMENTATION_FIXES.md`
- Complete Guide: `README_SIKLUS5_IMPROVED.md`
- Summary: `SUMMARY_IMPROVEMENTS.md`

---

## 🎉 You're Ready!

Just copy the file to Jupyter/Colab and run cell by cell!

**Expected runtime:** 15-40 minutes total
**Expected improvement:** 0.3-0.5 RMSE vs baseline

Good luck! 🚀🎵
