#!/usr/bin/env python3
"""
Main Pipeline - SIKLUS 5 IMPROVED
==================================

Complete pipeline dengan:
1. Comprehensive EDA
2. Enhanced feature engineering (NO DATA LEAKAGE)
3. Proper CV training
4. Comprehensive insights & visualizations

Run:
    python main_improved_pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Import our modules
from enhanced_features import engineer_features_enhanced
from proper_cv_training import train_model_with_proper_cv, train_final_model, predict_test_set


def process_lyrics(df_train, df_test, n_components=20):
    """
    Process lyrics dengan TF-IDF + SVD
    """
    print("\\n" + "="*80)
    print("📝 PROCESSING LYRICS (NLP)")
    print("="*80)

    if 'lyrics' not in df_train.columns:
        print("⚠ No lyrics column, skipping")
        return df_train, df_test

    df_train['lyrics'] = df_train['lyrics'].fillna('')
    df_test['lyrics'] = df_test['lyrics'].fillna('')

    print(f"\\nExtracting TF-IDF features (n_components={n_components})...")

    tfidf = TfidfVectorizer(
        max_features=500,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )

    train_tfidf = tfidf.fit_transform(df_train['lyrics'])
    test_tfidf = tfidf.transform(df_test['lyrics'])

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_lyrics_features = svd.fit_transform(train_tfidf)
    test_lyrics_features = svd.transform(test_tfidf)

    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"✓ Explained variance: {explained_variance:.2%}")

    lyrics_cols = [f'lyrics_feature_{i}' for i in range(n_components)]
    train_lyrics_df = pd.DataFrame(train_lyrics_features, columns=lyrics_cols, index=df_train.index)
    test_lyrics_df = pd.DataFrame(test_lyrics_features, columns=lyrics_cols, index=df_test.index)

    df_train = pd.concat([df_train, train_lyrics_df], axis=1)
    df_test = pd.concat([df_test, test_lyrics_df], axis=1)

    print(f"✓ Added {n_components} lyrics features")

    return df_train, df_test


def comprehensive_eda(df_train):
    """
    Comprehensive EDA dengan insights
    """
    print("\\n" + "="*80)
    print("📊 COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("="*80)

    # Target distribution
    print("\\n[1/5] Target Distribution...")
    print(f"\\n  Popularity Statistics:")
    print(df_train['popularity'].describe())

    bins = [0, 20, 40, 60, 80, 100]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    pop_bins = pd.cut(df_train['popularity'], bins=bins, labels=labels)

    print(f"\\n  Distribution by range:\")
    for label in labels:
        count = (pop_bins == label).sum()
        pct = (count / len(df_train)) * 100
        print(f\"    {label:12s}: {count:5d} ({pct:5.1f}%)\")

    # Genre analysis
    print(f\"\\n[2/5] Genre Analysis...\")
    genre_stats = df_train.groupby('track_genre')['popularity'].agg(['count', 'mean', 'std'])
    genre_stats = genre_stats.sort_values('mean', ascending=False)

    print(f\"\\n  Top 10 Genres by Popularity:\")
    print(f\"  {'Genre':<25} {'Count':>7} {'Mean':>7} {'Std':>7}\")
    print(f\"  {'-'*50}\")
    for idx, row in genre_stats.head(10).iterrows():
        print(f\"  {idx:<25} {row['count']:>7.0f} {row['mean']:>7.1f} {row['std']:>7.1f}\")

    # Artist analysis
    print(f\"\\n[3/5] Artist Analysis...\")
    artist_counts = df_train['artists'].value_counts()

    print(f\"  Total artists: {len(artist_counts)}\")
    print(f\"  Single-song artists: {(artist_counts == 1).sum()} ({(artist_counts == 1).sum()/len(artist_counts)*100:.1f}%)\")
    print(f\"  Artists with >10 songs: {(artist_counts > 10).sum()}\")

    # Audio features correlation
    print(f\"\\n[4/5] Audio Features Correlation...\")
    audio_features = ['energy', 'danceability', 'valence', 'loudness', 'tempo',
                     'acousticness', 'speechiness', 'instrumentalness', 'liveness']
    audio_features = [f for f in audio_features if f in df_train.columns]

    correlations = df_train[audio_features + ['popularity']].corr()['popularity'].drop('popularity')
    correlations = correlations.sort_values(ascending=False)

    print(f\"\\n  Correlation with Popularity:\")
    for feat, corr in correlations.items():
        strength = \"Strong\" if abs(corr) > 0.2 else \"Moderate\" if abs(corr) > 0.1 else \"Weak\"
        print(f\"    {feat:20s}: {corr:+.4f} ({strength})\")

    # Temporal analysis
    print(f\"\\n[5/5] Temporal Analysis...\")
    if 'release_year' in df_train.columns:
        print(f\"  Year range: {df_train['release_year'].min()} - {df_train['release_year'].max()}\")

        df_train['decade_temp'] = (df_train['release_year'] // 10) * 10
        decade_stats = df_train.groupby('decade_temp')['popularity'].mean()

        print(f\"\\n  Popularity by Decade:\")
        for decade, pop in decade_stats.items():
            print(f\"    {int(decade):4d}s: {pop:.1f}\")

        df_train.drop('decade_temp', axis=1, inplace=True)

    print(f\"\\n✅ EDA Complete!\")

    return df_train


def create_submission(predictions, test_ids, filename='submission_siklus5_improved.csv'):
    """
    Create submission file
    \"\"\"
    submission = pd.DataFrame({
        'track_id': test_ids,
        'popularity': predictions
    })

    submission['popularity'] = np.clip(submission['popularity'], 0, 100)

    submission.to_csv(filename, index=False)

    print(f\"\\n✅ Submission saved: {filename}\")
    print(f\"   Predictions: {len(submission)}\")
    print(f\"   Range: [{submission['popularity'].min():.2f}, {submission['popularity'].max():.2f}]\")
    print(f\"   Mean:  {submission['popularity'].mean():.2f}\")

    return submission


def run_improved_pipeline(data_path='/content', output_path='./outputs'):
    \"\"\"
    Run complete improved pipeline
    \"\"\"
    print(\"\\n\" + \"🎵\"*40)
    print(\"SONG POPULARITY PREDICTION - SIKLUS 5 IMPROVED\")
    print(\"Fixed Data Leakage + Enhanced Features + Comprehensive Insights\")
    print(\"🎵\"*40)

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"📂 STEP 1: LOADING DATA\")
    print(\"=\"*80)

    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(data_path / 'train.csv', engine='python')
    df_test = pd.read_csv(data_path / 'test.csv', engine='python')

    print(f\"\\n✓ Train: {df_train.shape}\")
    print(f\"✓ Test:  {df_test.shape}\")

    # =========================================================================
    # 2. COMPREHENSIVE EDA
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"📊 STEP 2: COMPREHENSIVE EDA\")
    print(\"=\"*80)

    df_train = comprehensive_eda(df_train)

    # =========================================================================
    # 3. ENHANCED FEATURE ENGINEERING
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"🔧 STEP 3: ENHANCED FEATURE ENGINEERING\")
    print(\"=\"*80)

    df_train, df_test, mappings = engineer_features_enhanced(df_train, df_test)

    # =========================================================================
    # 4. PROCESS LYRICS (NLP)
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"📝 STEP 4: PROCESS LYRICS\")
    print(\"=\"*80)

    df_train, df_test = process_lyrics(df_train, df_test, n_components=20)

    # =========================================================================
    # 5. PREPARE FEATURE LIST
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"🎯 STEP 5: PREPARE FEATURES\")
    print(\"=\"*80)

    # Get base features (exclude target-encoded ones, mereka dibuat di CV loop)
    exclude_cols = ['popularity', 'track_id', 'track_name', 'artists', 'lyrics',
                   'release_year', 'track_genre', 'decade', 'era',
                   'genre_avg_pop', 'genre_std_pop', 'genre_song_count',
                   'genre_min_pop', 'genre_max_pop',
                   'artist_avg_pop', 'artist_std_pop', 'artist_song_count',
                   'artist_min_pop', 'artist_max_pop',
                   'artist_x_dance', 'artist_x_energy']

    # Categorical features to encode
    categorical_features = ['key_mode', 'tempo_category']

    # Encode categorical
    from sklearn.preprocessing import LabelEncoder

    for col in categorical_features:
        if col in df_train.columns:
            le = LabelEncoder()
            combined = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
            le.fit(combined)

            df_train[col + '_encoded'] = le.transform(df_train[col].astype(str))
            df_test[col + '_encoded'] = le.transform(df_test[col].astype(str))

            df_train[col + '_encoded'] = df_train[col + '_encoded'].astype('category')
            df_test[col + '_encoded'] = df_test[col + '_encoded'].astype('category')

            exclude_cols.append(col)  # Add original to exclude list

    # Get base features
    base_features = [f for f in df_train.columns if f not in exclude_cols]

    print(f\"\\n✓ Base features prepared: {len(base_features)}\")
    print(f\"   (Target-encoded features will be added in CV loop)\")

    # =========================================================================
    # 6. TRAIN MODEL WITH PROPER CV (NO LEAKAGE!)
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"🤖 STEP 6: TRAIN MODEL (PROPER CV - NO LEAKAGE)\")
    print(\"=\"*80)

    results = train_model_with_proper_cv(
        df_train,
        base_features,
        target_col='popularity',
        cv_folds=5
    )

    # =========================================================================
    # 7. TRAIN FINAL MODEL & PREDICT
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"🎯 STEP 7: TRAIN FINAL MODEL & PREDICT TEST SET\")
    print(\"=\"*80)

    final_model, final_features = train_final_model(
        df_train,
        base_features,
        target_col='popularity'
    )

    # Prepare test set with same features
    from enhanced_features import create_fold_features
    df_test_with_features = create_fold_features(df_test, df_train, 'popularity')

    # Predict
    predictions = predict_test_set(final_model, df_test_with_features, final_features)

    # =========================================================================
    # 8. CREATE SUBMISSION
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"📤 STEP 8: CREATE SUBMISSION\")
    print(\"=\"*80)

    submission = create_submission(
        predictions,
        df_test['track_id'],
        filename=str(output_path / 'submission_siklus5_improved.csv')
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(\"\\n\" + \"=\"*80)
    print(\"✅ PIPELINE COMPLETE! - SIKLUS 5 IMPROVED\")
    print(\"=\"*80)

    print(f\"\\n🎯 FINAL RESULTS:\")
    print(f\"   OOF RMSE: {results['overall_rmse']:.4f}\")
    print(f\"   OOF MAE:  {results['overall_mae']:.4f}\")
    print(f\"   OOF R²:   {results['overall_r2']:.4f}\")

    print(f\"\\n📊 CV Statistics:\")
    print(f\"   Mean RMSE: {results['cv_scores'].mean():.4f}\")
    print(f\"   Std RMSE:  {results['cv_scores'].std():.4f}\")

    print(f\"\\n📁 Output Files:\")
    print(f\"   • submission_siklus5_improved.csv\")

    print(f\"\\n🔧 Key Improvements Implemented:\")
    print(f\"   ✅ Fixed data leakage (target encoding in CV loop)\")
    print(f\"   ✅ Added genre statistics features\")
    print(f\"   ✅ Added artist variance features\")
    print(f\"   ✅ Added audio feature ratios\")
    print(f\"   ✅ Early stopping & regularization\")
    print(f\"   ✅ Comprehensive EDA & insights\")

    print(f\"\\n✨ Expected improvement: 0.3-0.5 RMSE vs baseline\")

    return results, submission


if __name__ == \"__main__\":
    # Run pipeline
    results, submission = run_improved_pipeline(
        data_path='/content',        # Adjust if needed
        output_path='./outputs'
    )

    print(\"\\n\" + \"=\"*80)
    print(\"🎉 ALL DONE!\")
    print(\"=\"*80)
