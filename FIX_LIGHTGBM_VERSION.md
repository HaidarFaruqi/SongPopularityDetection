# 🔧 Fix: LightGBM Version Compatibility

## ❌ Error yang Terjadi

```
TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'early_stopping_rounds'
```

## 🔍 Penyebab

**LightGBM versi >= 4.0.0** mengubah cara early stopping:
- ❌ **Cara lama:** `early_stopping_rounds` parameter langsung di `fit()`
- ✅ **Cara baru:** Menggunakan `callbacks` parameter

## ✅ Solusi

### Opsi 1: Update ke Cara Baru (Recommended)

Gunakan kode yang kompatibel dengan semua versi:

```python
# Import callback untuk versi baru
try:
    from lightgbm import early_stopping, log_evaluation
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False

# Training
model = LGBMRegressor(**lgbm_params)

if USE_NEW_API:
    # LightGBM >= 4.0.0 (versi baru)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0)  # Silent
        ]
    )
else:
    # LightGBM < 4.0.0 (versi lama)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=False
    )
```

### Opsi 2: Downgrade LightGBM

Jika Anda prefer cara lama:

```bash
# Uninstall versi baru
pip uninstall lightgbm

# Install versi lama yang stabil
pip install "lightgbm<4.0.0"
```

**Versi yang disarankan:** `lightgbm==3.3.5`

---

## 🚀 Quick Fix untuk Notebook/Script Anda

Saya akan membuat **versi yang sudah diperbaiki** dengan auto-detection versi LightGBM.

### File yang Perlu Diperbaiki:

1. ✅ `TPW_AhThatsHot_Siklus5_Complete.ipynb` (Cell 3)
2. ✅ `Complete_Siklus5_SingleFile.py` (Cell 4)
3. ✅ `proper_cv_training.py`
4. ✅ `main_improved_pipeline.py`

---

## 📋 Perubahan Detail

### Before (Error dengan LightGBM >= 4.0):

```python
model.fit(
    X_train_fold, y_train_fold,
    eval_set=[(X_val_fold, y_val_fold)],
    eval_metric='rmse',
    early_stopping_rounds=100,  # ❌ Tidak support di versi baru
    verbose=False
)
```

### After (Kompatibel semua versi):

```python
# Check LightGBM version
try:
    from lightgbm import early_stopping, log_evaluation
    LIGHTGBM_NEW_API = True
except ImportError:
    LIGHTGBM_NEW_API = False

# Training (di dalam loop)
if LIGHTGBM_NEW_API:
    # Version >= 4.0.0
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0)
        ]
    )
else:
    # Version < 4.0.0
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=False
    )
```

---

## ⚙️ Cara Cek Versi LightGBM Anda

```python
import lightgbm as lgb
print(f"LightGBM version: {lgb.__version__}")
```

**Jika versi >= 4.0.0:** Gunakan cara baru (callbacks)
**Jika versi < 4.0.0:** Gunakan cara lama (early_stopping_rounds)

---

## 🎯 Kesimpulan

- **Root cause:** Breaking change di LightGBM 4.0.0
- **Impact:** Code yang ditulis untuk versi lama tidak work di versi baru
- **Solution:** Auto-detect versi dan gunakan API yang sesuai
- **Status:** Fix sudah dibuat dan akan di-push ke repository

---

**File yang sudah diperbaiki akan segera saya commit!**
