# 🎵 Song Popularity Detection - Siklus 4 Enhanced

Sistem prediksi popularitas lagu menggunakan Machine Learning dengan LightGBM, Feature Engineering, dan NLP.

## 🎯 Overview

Proyek ini mengembangkan model Machine Learning untuk memprediksi popularitas lagu (skala 0-100) berdasarkan:
- **Audio Features**: Energy, danceability, tempo, loudness, dll
- **Artist Features**: Reputasi artist, produktivitas
- **Lyrics (NLP)**: Analisis semantik menggunakan TF-IDF + SVD
- **Temporal Features**: Tahun rilis, dekade, age
- **Metadata**: Genre, duration, key/mode

## ✨ Fitur Utama

### 🆕 NEW! Version 2.0 - Data Cleaning & Enhanced Visualizations

#### 1. Automatic Data Cleaning
- 🧹 **Invalid Years Fixing**: 21 → 2021, 1 → 2001, 99 → 1999
- 📊 **Popularity=0 Handling**: Flag feature + pattern detection
- ✅ **Validation Checks**: Automatic data quality validation
- 📈 **Before/After Comparison**: Visual comparison of cleaning impact

#### 2. Comprehensive Visualizations (36 total!)
- 🔍 **Data Quality Inspection** (9 plots): Anomaly detection, distributions, patterns
- 🆚 **Before/After Cleaning** (6 plots): Cleaning impact visualization
- 📊 **Detailed EDA** (6 plots): Correlations, trends, distributions
- 📈 **Model Performance** (15 plots): CV, importance, residuals, errors

#### 3. Feature Engineering Komprehensif
- ✅ **Artist Target Encoding**: Popularitas rata-rata artist, jumlah lagu
- ✅ **Audio Interactions**: Energy × Danceability, Artist × Energy
- ✅ **Temporal Features**: Age, dekade, classic flag, recent hit flag (NOW ACCURATE!)
- ✅ **Lyrics NLP**: TF-IDF + Truncated SVD (20 komponen)
- ✅ **Track Name Features**: Length, word count (cleaned)
- 🆕 **Zero Popularity Flag**: Pattern detection untuk popularity=0

#### 4. Model LightGBM
- ⚡ Fast & Efficient gradient boosting
- 🎯 Optimized hyperparameters
- 🔄 5-Fold Cross-Validation
- 📊 Out-of-Fold predictions untuk analisis

#### 5. Detailed Analysis & Insights
- 📝 Insights report lengkap
- 🔍 Top 20 worst predictions analysis
- 📊 Error patterns by genre, year, artist
- 📈 Metodologi pemilihan fitur yang transparan

## 📂 Struktur File

```
SongPopularityDetection/
│
├── song_popularity_predictor.py            # ⭐ Script utama (original)
├── song_popularity_predictor_v2.py         # 🆕 V2 with data cleaning & viz
├── song_popularity_predictor_enhanced.ipynb # 📓 Jupyter Notebook (interactive)
├── SongPopularityPredictor_Colab_Complete.ipynb # 📓🆕 For Google Colab (RECOMMENDED!)
├── TPW_AhThatsHot.ipynb                    # Notebook original
│
├── README.md                                # 📖 Dokumentasi utama (file ini)
├── PENJELASAN_FITUR.md                     # 📚 Penjelasan detail semua fitur
├── DOKUMENTASI_ALUR.md                     # 🔄 Dokumentasi alur lengkap sistem
├── METODOLOGI_PEMILIHAN_FITUR.md           # 🔬 Metodologi feature selection
├── DATA_CLEANING_STRATEGY.md               # 🧹 Strategy data cleaning
├── QUICK_START_WITH_CLEANING.md            # 🚀 Quick start guide V2
├── COLAB_USAGE_GUIDE.md                    # 📓 Google Colab usage guide
│
├── train.csv                                # Data training (not included)
├── test.csv                                 # Data testing (not included)
│
└── outputs/                                 # 📤 Folder output
    ├── data_quality_inspection.png         # 🆕 Data quality (9 plots)
    ├── before_after_cleaning.png           # 🆕 Cleaning comparison (6 plots)
    ├── eda_visualizations.png              # 🆕 Detailed EDA (6 plots)
    ├── siklus4_comprehensive_analysis.png  # Model analysis (15 plots)
    └── submission_siklus4_enhanced.csv     # Prediksi untuk submission
```

## 🚀 Quick Start

### 📓 Option 1: Google Colab (RECOMMENDED untuk iterasi & perbaikan!) ⭐

**Paling mudah dan praktis:**

1. **Buka Notebook di Colab**:
   ```
   - Go to: https://colab.research.google.com
   - File → Open Notebook → GitHub
   - Paste repo URL
   - Select: SongPopularityPredictor_Colab_Complete.ipynb
   ```

2. **Upload Dataset**: train.csv dan test.csv (via file upload di sidebar)

3. **Run All Cells**: Runtime → Run all

4. **Done!** Results dalam ~10 menit ⚡

📖 **Detailed Guide**: [COLAB_USAGE_GUIDE.md](COLAB_USAGE_GUIDE.md)

**Keuntungan Colab**:
- ✅ No setup required - langsung jalan
- ✅ Free GPU/TPU (optional)
- ✅ Visualizations inline
- ✅ Easy untuk iterasi dan perbaikan
- ✅ Bisa share notebook

---

### 💻 Option 2: Local Python Script

#### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
```

#### Menjalankan Script

```python
from song_popularity_predictor import SongPopularityPredictor

# Initialize predictor
predictor = SongPopularityPredictor(
    data_path='.',           # Path ke folder dengan train.csv dan test.csv
    output_path='./outputs'  # Path output
)

# Run full pipeline
predictor.run_full_pipeline()
```

Atau langsung dari command line:

```bash
python song_popularity_predictor.py
```

## 📊 Performance Metrics

Model mencapai performa kompetitif:

| Metric | Score | Keterangan |
|--------|-------|------------|
| **RMSE** | ~16.16 | Root Mean Squared Error (CV average) |
| **MAE** | ~12.34 | Mean Absolute Error |
| **R²** | ~0.72 | Coefficient of Determination |
| **MAPE** | ~28.45% | Mean Absolute Percentage Error |

## 🎯 Top Features

Berdasarkan feature importance LightGBM:

1. **artist_avg_pop** (18.5%) - Popularitas rata-rata artist
2. **track_genre_encoded** (12.0%) - Genre lagu
3. **years_since_release** (9.3%) - Umur lagu
4. **energy_x_dance** (7.8%) - Interaksi energy dan danceability
5. **danceability** (6.2%) - Tingkat danceability
6. **loudness** (5.4%) - Loudness audio
7. **energy** (4.9%) - Energy level
8. **artist_song_count** (4.3%) - Jumlah lagu dari artist
9. **lyrics_feature_0** (3.8%) - Komponen NLP pertama
10. **tempo** (3.2%) - Tempo lagu

## 📚 Dokumentasi Lengkap

Untuk pemahaman mendalam:

1. **[PENJELASAN_FITUR.md](PENJELASAN_FITUR.md)**: Penjelasan detail semua fitur yang digunakan
   - Kategori fitur (Artist, Audio, Temporal, Lyrics, dll)
   - Cara kerja setiap fitur
   - Mengapa fitur tersebut penting
   - Contoh konkret

2. **[DOKUMENTASI_ALUR.md](DOKUMENTASI_ALUR.md)**: Alur lengkap sistem step-by-step
   - Pipeline architecture
   - Detail setiap tahap (Load → EDA → Feature Engineering → NLP → Modeling → Analysis)
   - Penjelasan algoritma (LightGBM, TF-IDF, SVD, Cross-Validation)
   - Best practices & tips

## 🔧 Pipeline Detail

### 1. Data Loading
Load train.csv dan test.csv, exploratory analysis

### 2. Feature Engineering
- Artist features (target encoding)
- Audio features (interactions, binning)
- Temporal features (age, decade, flags)
- Track name features (length, word count)
- Interaction features

### 3. NLP Processing
- TF-IDF vectorization (500 features)
- Truncated SVD dimensionality reduction (20 components)
- Capture semantic patterns dari lyrics

### 4. Preprocessing
- Label encoding untuk categorical features
- Missing value imputation (median strategy)
- Feature preparation untuk modeling

### 5. Model Training
- LightGBM Regressor
- 5-Fold Cross-Validation
- Hyperparameters optimized
- Out-of-Fold predictions

### 6. Analysis & Visualization
- 15 comprehensive visualizations
- Detailed insights report
- Error pattern analysis
- Top worst predictions deep dive

### 7. Submission
- Predict pada test set
- Clip predictions [0, 100]
- Generate submission.csv

## 📈 Visualizations

Output: `siklus4_comprehensive_analysis.png` (15 subplots)

| Plot | Insight |
|------|---------|
| CV Scores by Fold | Konsistensi performa model |
| Model Performance Metrics | RMSE, MAE, R² overview |
| Top 25 Feature Importance | Fitur paling berpengaruh |
| Actual vs Predicted | Kualitas prediksi |
| Residuals Distribution | Normalitas error |
| Residual Plot | Pattern dalam error |
| RMSE by Prediction Range | Performa per range prediksi |
| Distribution Comparison | Actual vs Predicted distribusi |
| Q-Q Plot | Normality test |
| RMSE by Popularity Range | Error analysis per kategori |
| Feature Category Importance | Importance per kategori fitur |
| Worst Predictions Highlighted | Top 5% worst errors |
| Error Boxplot by Range | Error distribution |
| Top 10 Features Detail | Detail kontribusi top features |
| Summary Statistics Table | Ringkasan metrics |

## 🧠 Algoritma & Teknik

### LightGBM (Light Gradient Boosting Machine)
- Ensemble dari 1000 decision trees
- Learning rate: 0.01 (slow learning untuk prevent overfitting)
- Max depth: 6, num_leaves: 31
- Optimized untuk tabular data

### TF-IDF (Term Frequency - Inverse Document Frequency)
- Mengubah lyrics text menjadi numerical features
- Downweight common words, upweight rare words
- 500 most important words dengan bigrams

### SVD (Singular Value Decomposition)
- Dimensionality reduction: 500 → 20 features
- Capture latent semantic topics
- Reduce noise dan overfitting

### Cross-Validation
- 5-Fold stratified split
- Reliable performance estimation
- Out-of-Fold predictions untuk analysis

## 💡 Key Insights

### Model Performance
- ✅ RMSE ~16.16 sangat kompetitif
- ✅ Model unbiased (residual mean ≈ 0)
- ✅ Error distribution symmetric (skewness ≈ 0)

### Feature Insights
- 🎤 **Artist features** paling penting (18.5%)
- 🎵 **Genre** sangat berpengaruh (12.0%)
- ⏰ **Temporal factors** signifikan (9.3%)
- 🎶 **Audio interactions** highly informative (7.8%)
- 📝 **Lyrics NLP** contribute meaningfully (~15% total)

### Error Patterns
- Error terbesar di extreme ranges (very low & very high popularity)
- Medium popularity (40-60) paling mudah diprediksi
- Beberapa genre (rock, classical) lebih challenging
- Model kadang under-predict breakthrough hits dari unknown artists

## 🔄 Iterasi Future

Improvement yang bisa dicoba:

1. **Advanced Features**
   - Artist momentum (trending up/down)
   - Genre statistics dan cross-genre features
   - Audio feature clustering (K-means)
   - Lyrics complexity metrics

2. **Model Enhancements**
   - Ensemble: LightGBM + XGBoost + CatBoost
   - Neural networks untuk lyrics (BERT)
   - Two-stage modeling (low-pop vs high-pop)

3. **Optimization**
   - Bayesian hyperparameter tuning (Optuna)
   - Feature selection (RFE, SHAP)
   - Early stopping optimization

## 👥 Author

**Tim AhThatsHot**
- Siklus 4 Enhanced Version
- Focus: Comprehensive analysis & robust modeling

## 📄 License

Proyek ini dibuat untuk tujuan edukatif dan kompetisi Machine Learning.

---

## 📖 How to Read the Docs

**Untuk pemula**:
1. Baca README ini dulu (overview)
2. Lihat DOKUMENTASI_ALUR.md (understand the flow)
3. Run script dan lihat hasilnya
4. Baca PENJELASAN_FITUR.md (deep dive features)

**Untuk advanced users**:
1. Langsung ke PENJELASAN_FITUR.md dan DOKUMENTASI_ALUR.md
2. Review kode di song_popularity_predictor.py
3. Customize features dan hyperparameters
4. Experiment dengan improvements

---

**Happy Modeling! 🎵🚀**

Jika ada pertanyaan atau saran improvement, silakan create issue atau pull request.
