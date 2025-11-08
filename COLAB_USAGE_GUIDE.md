# 📓 Google Colab Usage Guide

## 🚀 Quick Start untuk Google Colab

### ✨ Mengapa Colab?
- ✅ **Gratis GPU/TPU** (optional untuk speed up)
- ✅ **No setup required** - langsung jalan
- ✅ **Easy collaboration** - share notebook
- ✅ **Persistent outputs** - hasil tetap tersimpan
- ✅ **Visual inline** - plot langsung terlihat

---

## 📦 File untuk Colab

Pilih salah satu notebook berikut:

### 1. **SongPopularityPredictor_Colab_Complete.ipynb** ⭐ RECOMMENDED
   - ✅ Complete pipeline dengan semua fitur
   - ✅ Data cleaning automatic
   - ✅ 36 comprehensive visualizations
   - ✅ Ready to run

### 2. **song_popularity_predictor_enhanced.ipynb**
   - Interactive version (original)
   - Lebih banyak explanation cells
   - Suitable untuk pembelajaran

---

## 🎯 Cara Menggunakan di Google Colab

### Step 1: Upload Notebook ke Colab

**Option A - Dari GitHub** (easiest):
```
1. Buka https://colab.research.google.com
2. Klik "File" → "Open Notebook"
3. Tab "GitHub"
4. Paste URL repo Anda:
   https://github.com/HaidarFaruqi/SongPopularityDetection
5. Pilih notebook: SongPopularityPredictor_Colab_Complete.ipynb
6. Klik untuk open
```

**Option B - Upload Manual**:
```
1. Buka https://colab.research.google.com
2. Klik "File" → "Upload Notebook"
3. Pilih file .ipynb dari komputer Anda
4. Upload
```

**Option C - Dari Google Drive**:
```
1. Upload .ipynb ke Google Drive Anda
2. Klik kanan → Open with → Google Colaboratory
3. Done!
```

---

### Step 2: Upload Dataset

Dalam notebook, ada cell untuk upload files:

```python
from google.colab import files

uploaded = files.upload()
# Pilih train.csv dan test.csv dari komputer Anda
```

**Tips**:
- Upload kedua file sekaligus
- File akan tersimpan selama session aktif
- Jika runtime restart, perlu upload ulang

**Alternative - Mount Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')

# Kemudian akses files dari Drive
train_df = pd.read_csv('/content/drive/MyDrive/path/to/train.csv')
```

---

### Step 3: Run Cells Sequentially

```
1. Klik "Runtime" → "Run all"
   ATAU
   Run cell by cell dengan:
   - Shift + Enter (run & move to next)
   - Ctrl + Enter (run & stay)

2. First run akan install packages (~30 seconds)

3. Pipeline akan run automatic:
   ├── Load & inspect data
   ├── Data quality visualization (9 plots)
   ├── Clean data (fix years, handle zeros)
   ├── Before/After comparison (6 plots)
   ├── Feature engineering
   ├── NLP processing (lyrics)
   ├── Model training (5-Fold CV)
   ├── Visualizations (15 plots)
   └── Create submission
```

---

### Step 4: Download Results

Setelah selesai, download files:

```python
# Di notebook, ada cell:
from google.colab import files

# Download submission
files.download('submission_siklus4_enhanced.csv')

# Download visualizations
files.download('data_quality_inspection.png')
files.download('before_after_cleaning.png')
files.download('siklus4_comprehensive_analysis.png')
```

---

## 💡 Tips & Best Practices

### 1. Save Progress Frequently

```python
# Colab auto-saves, tapi untuk safety:
# Klik "File" → "Save a copy in Drive"
```

### 2. Use GPU (Optional - untuk speed up)

```python
# Klik "Runtime" → "Change runtime type"
# Hardware accelerator: GPU
# Save

# Verify:
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

Catatan: LightGBM tidak pakai GPU by default, tapi dataset loading & preprocessing lebih cepat.

### 3. Increase RAM if Needed

```python
# Jika dapat "RAM limit exceeded":
# Klik "Runtime" → "Change runtime type"
# Runtime shape: High-RAM
```

### 4. Keep Session Alive

```python
# Colab timeout setelah 90 menit idle
# Run this untuk prevent (optional):

import time
from IPython.display import Javascript

def keep_alive():
    display(Javascript('''
        function ClickConnect(){
            console.log("Keeping alive...");
            document.querySelector("colab-toolbar-button#connect").click()
        }
        setInterval(ClickConnect, 60000)
    '''))

keep_alive()
```

### 5. Monitor Resource Usage

```python
# Check RAM usage:
!cat /proc/meminfo | grep MemAvailable

# Check disk usage:
!df -h

# Check GPU (if enabled):
!nvidia-smi
```

---

## 🔧 Troubleshooting

### Problem: "Module not found"

**Solution**:
```python
# Re-run install cell:
!pip install lightgbm
```

### Problem: "File not found: train.csv"

**Solution**:
```python
# Verify files uploaded:
import os
print(os.listdir('.'))

# Should see: ['train.csv', 'test.csv', ...]
```

### Problem: "Runtime crashed"

**Solution**:
```python
# 1. Reduce sample size in visualizations:
sample = train_df.sample(min(3000, len(train_df)))  # Instead of 5000

# 2. Clear outputs before re-running:
# Edit → Clear all outputs

# 3. Restart runtime:
# Runtime → Restart runtime
```

### Problem: "Timeout during upload"

**Solution**:
```python
# For large files, use Google Drive instead:
from google.colab import drive
drive.mount('/content/drive')

# Then read from Drive (no upload needed)
```

---

## 📊 Expected Runtime

Dengan dataset standar (~114K rows):

| Step | Time | Notes |
|------|------|-------|
| Install packages | ~30s | First time only |
| Load data | ~5s | Depends on file size |
| Data quality viz | ~20s | 9 plots |
| Data cleaning | ~10s | Fast |
| Feature engineering | ~30s | Artist encoding takes time |
| NLP processing | ~2-3min | TF-IDF + SVD |
| Model training (5-CV) | ~3-5min | Depends on n_estimators |
| Visualizations | ~30s | 15 plots |
| Total | **~8-10min** | End-to-end |

---

## 🎯 Output Files

Setelah run complete, Anda akan punya:

```
📁 Output Files:
├── submission_siklus4_enhanced.csv       # 🎯 Final predictions
├── data_quality_inspection.png           # 📊 9 quality plots
├── before_after_cleaning.png             # 📊 6 comparison plots
├── eda_visualizations.png                # 📊 6 EDA plots
└── siklus4_comprehensive_analysis.png    # 📊 15 model plots

Total: 36 plots + 1 submission file
```

---

## 🔄 Modifications & Iterations

### Untuk Eksperimen dengan Features:

```python
# Di cell feature engineering, tambahkan:

# Contoh: Add new feature
train_df['new_feature'] = train_df['energy'] * train_df['valence']
test_df['new_feature'] = test_df['energy'] * test_df['valence']
```

### Untuk Tune Hyperparameters:

```python
# Di cell model training, ubah:

lgbm = LGBMRegressor(
    n_estimators=1500,  # Default: 1000
    learning_rate=0.005,  # Default: 0.01 (smaller = slower but better)
    num_leaves=63,  # Default: 31 (more complex)
    max_depth=8,  # Default: 6
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

### Untuk Coba Model Lain:

```python
# Tambah cell baru:

from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    random_state=42
)

# Train dengan CV
cv_scores_xgb = cross_val_score(xgb, X, y, cv=kfold,
                                scoring='neg_root_mean_squared_error')
print(f"XGBoost RMSE: {-cv_scores_xgb.mean():.4f}")
```

---

## 📚 Additional Resources

### Colab Tutorials:
- [Official Colab Welcome](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab Tips & Tricks](https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb)

### LightGBM:
- [Documentation](https://lightgbm.readthedocs.io/)
- [Parameter Tuning Guide](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

### Feature Engineering:
- [Kaggle Feature Engineering Course](https://www.kaggle.com/learn/feature-engineering)

---

## ✅ Checklist untuk Successful Run

Before running:
- [ ] Notebook uploaded to Colab
- [ ] train.csv ready (uploaded or in Drive)
- [ ] test.csv ready (uploaded or in Drive)
- [ ] Runtime type selected (CPU or GPU)

During run:
- [ ] Install cell completed successfully
- [ ] Data loaded without errors
- [ ] Visualizations displaying correctly
- [ ] No RAM/timeout issues

After run:
- [ ] Submission file created
- [ ] All visualizations saved
- [ ] Results downloaded
- [ ] Notebook saved to Drive (for future use)

---

## 🎉 Ready to Start!

Sekarang Anda siap untuk:
1. Open notebook di Colab
2. Upload dataset
3. Run all cells
4. Get results in ~10 minutes!

**Happy modeling di Colab!** 🚀

---

**Need Help?**
- Check [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Review error messages carefully
- Try restarting runtime if stuck
- Re-upload files if "file not found"
