#!/usr/bin/env python3
"""
Song Popularity Predictor - SIKLUS 5 IMPROVED
=============================================

Major Improvements:
- Fixed data leakage in target encoding
- Added comprehensive EDA with insights
- Added genre statistics features
- Added artist variance features
- Added audio feature ratios
- Improved model configuration
- Before/After processing insights
- Feature selection & importance analysis

Expected improvement: 0.3-0.5 RMSE reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from lightgbm import LGBMRegressor
from scipy import stats
import joblib
from datetime import datetime

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


class SongPopularityPredictorImproved:
    """
    SIKLUS 5 IMPROVED: Fixed Data Leakage + Advanced Features + Comprehensive Insights

    Key Improvements:
    1. Proper target encoding (no leakage)
    2. Genre & artist statistics
    3. Audio feature ratios
    4. Early stopping & regularization
    5. Comprehensive insights at each stage
    6. Feature importance analysis
    """

    def __init__(self, data_path='/content', output_path='./outputs'):
        """Initialize predictor dengan konfigurasi path"""
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.train_df = None
        self.test_df = None
        self.features = []
        self.models = {}
        self.scalers = {}
        self.imputer = None
        self.label_encoders = {}
        self.categorical_features_raw = []
        self.categorical_features_encoded = []
        self.model_results = None
        self.oof_predictions = None
        self.cv_scores = None
        self.feature_importance_df = None

        # Insights storage
        self.insights = {
            'data_loading': {},
            'eda': {},
            'feature_engineering': {},
            'preprocessing': {},
            'modeling': {},
            'feature_importance': {}
        }

        self.output_path.mkdir(parents=True, exist_ok=True)

        print("✅ SongPopularityPredictorImproved initialized!")
        print(f"📂 Data path: {self.data_path}")
        print(f"📁 Output path: {self.output_path}")

    def load_data(self):
        """
        Load data dengan comprehensive validation dan insights
        """
        print("\\n" + "="*80)
        print("📂 LOADING DATA & INITIAL VALIDATION")
        print("="*80)

        # Load data
        self.train_df = pd.read_csv(self.data_path / 'train.csv', engine='python')
        self.test_df = pd.read_csv(self.data_path / 'test.csv', engine='python')

        # Basic info
        print(f"\\n✓ Training data: {self.train_df.shape}")
        print(f"✓ Testing data: {self.test_df.shape}")

        # Target statistics
        target_stats = {
            'mean': self.train_df['popularity'].mean(),
            'median': self.train_df['popularity'].median(),
            'std': self.train_df['popularity'].std(),
            'min': self.train_df['popularity'].min(),
            'max': self.train_df['popularity'].max(),
            'skew': stats.skew(self.train_df['popularity']),
            'kurtosis': stats.kurtosis(self.train_df['popularity'])
        }

        print(f"\\n📊 Target Statistics:")
        print(f"  Mean:     {target_stats['mean']:.2f}")
        print(f"  Median:   {target_stats['median']:.2f}")
        print(f"  Std:      {target_stats['std']:.2f}")
        print(f"  Range:    [{target_stats['min']:.0f}, {target_stats['max']:.0f}]")
        print(f"  Skewness: {target_stats['skew']:.3f}")
        print(f"  Kurtosis: {target_stats['kurtosis']:.3f}")

        # Missing values check
        train_missing = self.train_df.isnull().sum().sum()
        test_missing = self.test_df.isnull().sum().sum()

        print(f"\\n🔍 Missing Values:")
        print(f"  Train: {train_missing} total")
        print(f"  Test:  {test_missing} total")

        if train_missing > 0 or test_missing > 0:
            print(f"\\n  Columns with missing:")
            for col in self.train_df.columns:
                train_miss = self.train_df[col].isnull().sum()
                test_miss = self.test_df[col].isnull().sum() if col in self.test_df.columns else 0
                if train_miss > 0 or test_miss > 0:
                    print(f"    {col}: Train={train_miss}, Test={test_miss}")

        # Store insights
        self.insights['data_loading'] = {
            'train_shape': self.train_df.shape,
            'test_shape': self.test_df.shape,
            'target_stats': target_stats,
            'missing_train': train_missing,
            'missing_test': test_missing
        }

        return self

    def comprehensive_eda(self):
        """
        Exploratory Data Analysis yang sangat comprehensive dengan insights
        """
        print("\\n" + "="*80)
        print("📊 COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*80)

        insights = {}

        # ============================================================================
        # 1. TARGET DISTRIBUTION ANALYSIS
        # ============================================================================
        print("\\n[1/8] Target Distribution Analysis...")

        # Distribution bins
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        pop_bins = pd.cut(self.train_df['popularity'], bins=bins, labels=labels)

        print(f\"\\n  Distribution by popularity range:\")
        dist = pop_bins.value_counts().sort_index()
        for label, count in dist.items():
            pct = (count / len(self.train_df)) * 100
            print(f\"    {label:12s}: {count:5d} ({pct:5.1f}%)\")

        insights['target_distribution'] = dist.to_dict()

        # Check for imbalance
        max_pct = (dist.max() / len(self.train_df)) * 100
        min_pct = (dist.min() / len(self.train_df)) * 100
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')

        if imbalance_ratio > 3:
            print(f\"\\n  ⚠️ WARNING: Imbalanced target (ratio: {imbalance_ratio:.1f}x)\")
            print(f\"     Consider stratified CV or sampling techniques\")

        # ============================================================================
        # 2. CATEGORICAL FEATURES ANALYSIS
        # ============================================================================
        print(f\"\\n[2/8] Categorical Features Analysis...\")

        # Genre analysis
        genre_stats = self.train_df.groupby('track_genre')['popularity'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).sort_values('mean', ascending=False)

        print(f\"\\n  Genre Statistics (Top 10 by avg popularity):\")
        print(f\"  {'Genre':<25} {'Count':>7} {'Mean':>7} {'Std':>7}\")
        print(f\"  {'-'*50}\")
        for idx, row in genre_stats.head(10).iterrows():
            print(f\"  {idx:<25} {row['count']:>7.0f} {row['mean']:>7.1f} {row['std']:>7.1f}\")

        insights['genre_stats'] = {
            'n_genres': self.train_df['track_genre'].nunique(),
            'top_genre': genre_stats.index[0],
            'top_genre_mean_pop': genre_stats['mean'].iloc[0],
            'genre_popularity_range': (genre_stats['mean'].min(), genre_stats['mean'].max())
        }

        # Artist analysis
        artist_counts = self.train_df['artists'].value_counts()
        artist_stats = self.train_df.groupby('artists')['popularity'].agg(['count', 'mean', 'std'])

        print(f\"\\n  Artist Statistics:\")
        print(f\"    Total unique artists: {len(artist_counts)}\")
        print(f\"    Artists with 1 song: {(artist_counts == 1).sum()} ({(artist_counts == 1).sum()/len(artist_counts)*100:.1f}%)\")
        print(f\"    Artists with >10 songs: {(artist_counts > 10).sum()}\")
        print(f\"    Artists with >50 songs: {(artist_counts > 50).sum()}\")

        # Most prolific artists
        print(f\"\\n  Top 5 Most Prolific Artists:\")
        for artist, count in artist_counts.head(5).items():
            avg_pop = artist_stats.loc[artist, 'mean']
            print(f\"    {artist[:40]:40s}: {count:3d} songs (avg pop: {avg_pop:.1f})\")

        insights['artist_stats'] = {
            'n_artists': len(artist_counts),
            'single_song_artists': (artist_counts == 1).sum(),
            'prolific_artists': (artist_counts > 10).sum()
        }

        # ============================================================================
        # 3. AUDIO FEATURES ANALYSIS
        # ============================================================================
        print(f\"\\n[3/8] Audio Features Analysis...\")

        audio_features = ['energy', 'danceability', 'valence', 'loudness', 'tempo',
                         'acousticness', 'speechiness', 'instrumentalness', 'liveness']
        audio_features = [f for f in audio_features if f in self.train_df.columns]

        # Correlation with target
        correlations = self.train_df[audio_features + ['popularity']].corr()['popularity'].drop('popularity').sort_values(ascending=False)

        print(f\"\\n  Correlation with Popularity:\")
        print(f\"  {'Feature':<20} {'Correlation':>12} {'Strength':>12}\")
        print(f\"  {'-'*48}\")
        for feat, corr in correlations.items():
            if abs(corr) > 0.2:
                strength = \"🔴 Strong\"
            elif abs(corr) > 0.1:
                strength = \"🟡 Moderate\"
            else:
                strength = \"🟢 Weak\"
            print(f\"  {feat:<20} {corr:>+12.4f} {strength:>12}\")

        insights['audio_correlations'] = correlations.to_dict()

        # ============================================================================
        # 4. TEMPORAL ANALYSIS
        # ============================================================================
        print(f\"\\n[4/8] Temporal Analysis (Release Year)...\")

        if 'release_year' in self.train_df.columns:
            year_stats = self.train_df.groupby('release_year')['popularity'].agg([
                'count', 'mean', 'std'
            ]).sort_index()

            print(f\"\\n  Year Range: {self.train_df['release_year'].min()} - {self.train_df['release_year'].max()}\")

            # Decade analysis
            self.train_df['decade_temp'] = (self.train_df['release_year'] // 10) * 10
            decade_stats = self.train_df.groupby('decade_temp')['popularity'].agg(['count', 'mean'])

            print(f\"\\n  Popularity by Decade:\")
            print(f\"  {'Decade':>8} {'Count':>8} {'Mean Pop':>10}\")
            print(f\"  {'-'*30}\")
            for decade, row in decade_stats.iterrows():
                print(f\"  {int(decade):>8d} {row['count']:>8.0f} {row['mean']:>10.1f}\")

            # Trend analysis
            year_trend = np.corrcoef(self.train_df['release_year'], self.train_df['popularity'])[0, 1]
            print(f\"\\n  📈 Time Trend: Correlation(year, popularity) = {year_trend:+.3f}\")
            if year_trend > 0.1:
                print(f\"     → Newer songs tend to be more popular\")
            elif year_trend < -0.1:
                print(f\"     → Older songs tend to be more popular\")
            else:
                print(f\"     → No clear time trend\")

            insights['temporal'] = {
                'year_range': (int(self.train_df['release_year'].min()), int(self.train_df['release_year'].max())),
                'year_trend': year_trend,
                'decade_stats': decade_stats.to_dict()
            }

            self.train_df.drop('decade_temp', axis=1, inplace=True)

        # ============================================================================
        # 5. LYRICS ANALYSIS (if available)
        # ============================================================================
        print(f\"\\n[5/8] Lyrics Analysis...\")

        if 'lyrics' in self.train_df.columns:
            lyrics_present = self.train_df['lyrics'].notna().sum()
            lyrics_pct = (lyrics_present / len(self.train_df)) * 100

            print(f\"\\n  Songs with lyrics: {lyrics_present} ({lyrics_pct:.1f}%)\")
            print(f\"  Songs without lyrics: {len(self.train_df) - lyrics_present}\")

            if lyrics_present > 0:
                # Lyrics length analysis
                lyrics_lengths = self.train_df['lyrics'].fillna('').str.len()

                print(f\"\\n  Lyrics Length Statistics:\")
                print(f\"    Mean:   {lyrics_lengths.mean():.0f} characters\")
                print(f\"    Median: {lyrics_lengths.median():.0f} characters\")
                print(f\"    Max:    {lyrics_lengths.max():.0f} characters\")

                # Word count
                word_counts = self.train_df['lyrics'].fillna('').str.split().str.len()
                print(f\"\\n  Lyrics Word Count:\")
                print(f\"    Mean:   {word_counts.mean():.0f} words\")
                print(f\"    Median: {word_counts.median():.0f} words\")

                insights['lyrics'] = {
                    'coverage': lyrics_pct,
                    'avg_length': float(lyrics_lengths.mean()),
                    'avg_words': float(word_counts.mean())
                }
        else:
            print(f\"\\n  ⚠️ No lyrics column found\")
            insights['lyrics'] = {'coverage': 0}

        # ============================================================================
        # 6. DATA QUALITY CHECKS
        # ============================================================================
        print(f\"\\n[6/8] Data Quality Checks...\")

        quality_issues = []

        # Check duplicates
        duplicates = self.train_df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f\"Duplicate rows: {duplicates}\")

        # Check for outliers in audio features
        for feat in audio_features:
            if feat in ['energy', 'danceability', 'valence', 'acousticness',
                       'speechiness', 'instrumentalness', 'liveness']:
                outliers = ((self.train_df[feat] < 0) | (self.train_df[feat] > 1)).sum()
                if outliers > 0:
                    quality_issues.append(f\"{feat}: {outliers} values outside [0,1]\")

        # Check negative duration
        if 'duration_ms' in self.train_df.columns:
            neg_duration = (self.train_df['duration_ms'] <= 0).sum()
            if neg_duration > 0:
                quality_issues.append(f\"Negative/zero duration: {neg_duration}\")

        if quality_issues:
            print(f\"\\n  ⚠️ Quality Issues Found:\")
            for issue in quality_issues:
                print(f\"    • {issue}\")
        else:
            print(f\"\\n  ✅ No data quality issues detected!\")

        insights['data_quality'] = {
            'issues_found': len(quality_issues),
            'issues': quality_issues
        }

        # ============================================================================
        # 7. MULTICOLLINEARITY CHECK
        # ============================================================================
        print(f\"\\n[7/8] Multicollinearity Check...\")

        if len(audio_features) > 0:
            audio_corr = self.train_df[audio_features].corr()

            # Find high correlations
            high_corr_pairs = []
            for i in range(len(audio_corr.columns)):
                for j in range(i+1, len(audio_corr.columns)):
                    if abs(audio_corr.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((
                            audio_corr.columns[i],
                            audio_corr.columns[j],
                            audio_corr.iloc[i, j]
                        ))

            if high_corr_pairs:
                print(f\"\\n  ⚠️ High Correlations Detected (|r| > 0.7):\")
                for feat1, feat2, corr in high_corr_pairs:
                    print(f\"    {feat1} <-> {feat2}: {corr:+.3f}\")
                print(f\"\\n  💡 Consider: Feature selection or dimensionality reduction\")
            else:
                print(f\"\\n  ✅ No severe multicollinearity detected\")

            insights['multicollinearity'] = {
                'high_corr_pairs': len(high_corr_pairs),
                'pairs': [(f1, f2, float(c)) for f1, f2, c in high_corr_pairs]
            }

        # ============================================================================
        # 8. FEATURE ENGINEERING OPPORTUNITIES
        # ============================================================================
        print(f\"\\n[8/8] Feature Engineering Opportunities...\")

        opportunities = []

        # Artist-based
        if (artist_counts > 1).sum() > 100:
            opportunities.append(\"✓ Artist statistics (many artists with multiple songs)\")\n
        # Genre-based
        if self.train_df['track_genre'].nunique() > 10:
            opportunities.append(\"✓ Genre statistics (diverse genres)\")

        # Time-based
        if 'release_year' in self.train_df.columns:
            if self.train_df['release_year'].nunique() > 10:
                opportunities.append(\"✓ Temporal features (wide year range)\")

        # Track name patterns
        if 'track_name' in self.train_df.columns:
            has_feat = self.train_df['track_name'].str.contains('feat|ft\.|featuring', case=False, na=False).sum()
            if has_feat > 100:
                opportunities.append(f\"✓ Track name patterns ({has_feat} songs with featuring)\")

        # Audio interactions
        if len(audio_features) >= 3:
            opportunities.append(\"✓ Audio feature interactions & ratios\")

        print(f\"\\n  Detected Opportunities:\")
        for opp in opportunities:
            print(f\"    {opp}\")

        insights['opportunities'] = opportunities

        # Store all insights
        self.insights['eda'] = insights

        print(f\"\\n✅ Comprehensive EDA Complete!\")
        print(f\"   {len(insights)} analysis categories completed\")

        return self

    # CONTINUATION IN NEXT MESSAGE...
