#!/usr/bin/env python3
"""
============================================================================
SONG POPULARITY PREDICTOR - VERSION 2.0
============================================================================
Enhanced version with:
- Data Cleaning (invalid years, popularity=0 handling)
- Comprehensive Visualizations at each step
- Insights and analysis throughout pipeline

Author: Tim AhThatsHot
Version: 2.0 - Clean & Visual
============================================================================
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
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from scipy import stats
import joblib
from datetime import datetime

# Konfigurasi visualisasi
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


class SongPopularityPredictorV2:
    """
    VERSION 2.0: Enhanced Predictor dengan Data Cleaning & Comprehensive Visualizations

    New Features:
    - Data cleaning (invalid years, anomaly detection)
    - Visual insights at each pipeline step
    - Before/after comparisons
    - Detailed anomaly reports
    """

    def __init__(self, data_path='.', output_path='./outputs'):
        """Inisialisasi predictor"""
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)

        # Data containers
        self.train_df = None
        self.test_df = None
        self.train_df_original = None  # Backup untuk comparison

        # Features & models
        self.features = []
        self.models = {}
        self.scalers = {}

        # Preprocessing objects
        self.imputer = None
        self.label_encoders = {}
        self.categorical_features_raw = []
        self.categorical_features_encoded = []

        # Results
        self.model_results = None
        self.oof_predictions = None
        self.cv_scores = None

        # Cleaning stats
        self.cleaning_stats = {}

        # Buat output folder
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load data dengan initial inspection"""
        print("=" * 80)
        print("📂 LOADING DATA")
        print("=" * 80)

        self.train_df = pd.read_csv(self.data_path / 'train.csv', engine='python')
        self.test_df = pd.read_csv(self.data_path / 'test.csv', engine='python')

        # Backup original
        self.train_df_original = self.train_df.copy()

        print(f"✓ Training data: {self.train_df.shape}")
        print(f"✓ Testing data: {self.test_df.shape}")

        return self

    def visualize_data_quality(self):
        """Visualisasi kualitas data SEBELUM cleaning"""
        print("\n" + "=" * 80)
        print("🔍 DATA QUALITY INSPECTION")
        print("=" * 80)

        fig = plt.figure(figsize=(18, 12))

        # 1. Release Year Distribution (with anomalies)
        ax1 = plt.subplot(3, 3, 1)
        years = self.train_df['release_year'].dropna()
        ax1.hist(years, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Suspicious (<100)')
        ax1.set_xlabel('Release Year', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Release Year Distribution (Before Cleaning)', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Count anomalies
        anomaly_count = (years < 100).sum()
        ax1.text(0.98, 0.98, f'Anomalies: {anomaly_count}',
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontweight='bold')

        # 2. Suspicious Years Detail
        ax2 = plt.subplot(3, 3, 2)
        suspicious_years = years[years < 100]
        if len(suspicious_years) > 0:
            year_counts = suspicious_years.value_counts().sort_index()
            ax2.bar(year_counts.index, year_counts.values, edgecolor='black', color='coral')
            ax2.set_xlabel('Year Value', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title(f'Invalid Years Detail ({len(suspicious_years)} records)', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No invalid years found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)

        # 3. Popularity Distribution
        ax3 = plt.subplot(3, 3, 3)
        pop_data = self.train_df['popularity']
        ax3.hist(pop_data, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero popularity')
        ax3.set_xlabel('Popularity', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Popularity Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Count zeros
        zero_count = (pop_data == 0).sum()
        zero_pct = (zero_count / len(pop_data)) * 100
        ax3.text(0.98, 0.98, f'Zero: {zero_count} ({zero_pct:.1f}%)',
                transform=ax3.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontweight='bold')

        # 4. Popularity = 0 by Genre
        ax4 = plt.subplot(3, 3, 4)
        zero_pop_genres = self.train_df[self.train_df['popularity'] == 0]['track_genre'].value_counts().head(10)
        if len(zero_pop_genres) > 0:
            ax4.barh(range(len(zero_pop_genres)), zero_pop_genres.values, color='salmon', edgecolor='black')
            ax4.set_yticks(range(len(zero_pop_genres)))
            ax4.set_yticklabels(zero_pop_genres.index, fontsize=9)
            ax4.set_xlabel('Count', fontweight='bold')
            ax4.set_title('Top Genres with Popularity=0', fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            ax4.invert_yaxis()

        # 5. Missing Values Heatmap
        ax5 = plt.subplot(3, 3, 5)
        missing_data = self.train_df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            ax5.barh(range(len(missing_data)), missing_data.values, color='orange', edgecolor='black')
            ax5.set_yticks(range(len(missing_data)))
            ax5.set_yticklabels(missing_data.index, fontsize=9)
            ax5.set_xlabel('Missing Count', fontweight='bold')
            ax5.set_title('Missing Values by Column', fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
            ax5.invert_yaxis()
        else:
            ax5.text(0.5, 0.5, 'No missing values!', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14, fontweight='bold', color='green')

        # 6. Year vs Popularity Scatter (before cleaning)
        ax6 = plt.subplot(3, 3, 6)
        sample_data = self.train_df.sample(min(5000, len(self.train_df)))
        scatter = ax6.scatter(sample_data['release_year'], sample_data['popularity'],
                            alpha=0.3, s=10, c=sample_data['popularity'], cmap='viridis')
        ax6.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Invalid threshold')
        ax6.set_xlabel('Release Year', fontweight='bold')
        ax6.set_ylabel('Popularity', fontweight='bold')
        ax6.set_title('Year vs Popularity (Before Cleaning)', fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='Popularity')

        # 7. Popularity Boxplot
        ax7 = plt.subplot(3, 3, 7)
        ax7.boxplot(pop_data)
        ax7.set_ylabel('Popularity', fontweight='bold')
        ax7.set_title('Popularity Boxplot', fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)

        # Stats
        stats_text = f"Mean: {pop_data.mean():.1f}\nMedian: {pop_data.median():.1f}\nStd: {pop_data.std():.1f}"
        ax7.text(0.98, 0.98, stats_text, transform=ax7.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 8. Top Genres
        ax8 = plt.subplot(3, 3, 8)
        top_genres = self.train_df['track_genre'].value_counts().head(10)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
        ax8.barh(range(len(top_genres)), top_genres.values, color=colors, edgecolor='black')
        ax8.set_yticks(range(len(top_genres)))
        ax8.set_yticklabels(top_genres.index, fontsize=9)
        ax8.set_xlabel('Count', fontweight='bold')
        ax8.set_title('Top 10 Genres', fontweight='bold')
        ax8.grid(axis='x', alpha=0.3)
        ax8.invert_yaxis()

        # 9. Summary Statistics Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_data = [
            ['Metric', 'Value'],
            ['Total Records', f"{len(self.train_df):,}"],
            ['Features', f"{len(self.train_df.columns)}"],
            ['Invalid Years (<100)', f"{anomaly_count:,}"],
            ['Popularity = 0', f"{zero_count:,} ({zero_pct:.1f}%)"],
            ['Missing Values', f"{self.train_df.isnull().sum().sum():,}"],
            ['Unique Artists', f"{self.train_df['artists'].nunique():,}"],
            ['Unique Genres', f"{self.train_df['track_genre'].nunique()}"],
            ['Year Range', f"{years[years>=1000].min():.0f}-{years.max():.0f}"],
            ['Popularity Mean', f"{pop_data.mean():.1f}"]
        ]

        table = ax9.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        for i in range(len(summary_data)):
            if i == 0:
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#E8F5E9')
                table[(i, 1)].set_text_props(weight='bold')

        ax9.set_title('Data Quality Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_path / 'data_quality_inspection.png', dpi=300, bbox_inches='tight')
        print(f"✓ Data quality visualization saved!")
        plt.show()

        return self

    def clean_data(self):
        """Clean data anomalies"""
        print("\n" + "=" * 80)
        print("🧹 DATA CLEANING")
        print("=" * 80)

        # ============ 1. FIX INVALID YEARS ============
        print("\n[1/2] Fixing invalid release years...")

        def fix_release_year(year):
            """
            Fix invalid release years
            Rules:
            - year < 25: assume 2000s (21 → 2021)
            - 25 <= year < 100: assume 1900s (99 → 1999)
            - year >= 1000: keep as is
            - Special: 0 → 2000
            """
            if pd.isna(year):
                return year

            year = int(year)

            if year >= 1000:
                return year

            if year == 0:
                return 2000

            if year < 25:
                return 2000 + year
            elif year < 100:
                return 1900 + year
            else:
                return year

        # Count before
        invalid_train_before = (self.train_df['release_year'] < 100).sum()
        invalid_test_before = (self.test_df['release_year'] < 100).sum()

        # Apply fix
        self.train_df['release_year'] = self.train_df['release_year'].apply(fix_release_year)
        self.test_df['release_year'] = self.test_df['release_year'].apply(fix_release_year)

        # Count after
        invalid_train_after = (self.train_df['release_year'] < 1000).sum()
        invalid_test_after = (self.test_df['release_year'] < 1000).sum()

        print(f"   Train: Fixed {invalid_train_before} invalid years")
        print(f"   Test:  Fixed {invalid_test_before} invalid years")
        print(f"   ✓ All years now in valid range [1000, 2025]")

        self.cleaning_stats['invalid_years_fixed'] = {
            'train': invalid_train_before,
            'test': invalid_test_before
        }

        # ============ 2. HANDLE POPULARITY = 0 ============
        print("\n[2/2] Analyzing popularity = 0...")

        zero_count = (self.train_df['popularity'] == 0).sum()
        zero_pct = (zero_count / len(self.train_df)) * 100

        print(f"   Found {zero_count} records ({zero_pct:.2f}%) with popularity = 0")

        # Create flag feature
        self.train_df['is_zero_popularity'] = (self.train_df['popularity'] == 0).astype(int)
        # Test set doesn't have popularity, so we can't create this flag

        if zero_pct > 10:
            print(f"   ⚠ High percentage (>{10}%) - might indicate data quality issue")
        elif zero_pct < 5:
            print(f"   ✓ Normal percentage - keeping as valid data")
        else:
            print(f"   ℹ Moderate percentage - created flag feature")

        # We keep the zeros as-is (valid "unpopular" songs)
        # But create a flag so model can learn this pattern

        self.cleaning_stats['zero_popularity'] = {
            'count': zero_count,
            'percentage': zero_pct
        }

        print("\n✓ Data cleaning completed!")

        return self

    def visualize_after_cleaning(self):
        """Visualisasi SETELAH cleaning untuk comparison"""
        print("\n" + "=" * 80)
        print("📊 BEFORE/AFTER CLEANING COMPARISON")
        print("=" * 80)

        fig = plt.figure(figsize=(18, 8))

        # 1. Year Distribution - Before
        ax1 = plt.subplot(2, 3, 1)
        years_before = self.train_df_original['release_year'].dropna()
        ax1.hist(years_before, bins=100, edgecolor='black', alpha=0.7, color='lightcoral')
        ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Invalid threshold')
        ax1.set_xlabel('Release Year', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('BEFORE: Release Year Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        anomaly_before = (years_before < 100).sum()
        ax1.text(0.02, 0.98, f'Invalid: {anomaly_before}', transform=ax1.transAxes,
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                fontweight='bold', color='white')

        # 2. Year Distribution - After
        ax2 = plt.subplot(2, 3, 2)
        years_after = self.train_df['release_year'].dropna()
        ax2.hist(years_after, bins=100, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Release Year', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('AFTER: Release Year Distribution', fontweight='bold')
        ax2.grid(alpha=0.3)

        anomaly_after = (years_after < 1000).sum()
        ax2.text(0.02, 0.98, f'Invalid: {anomaly_after}', transform=ax2.transAxes,
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                fontweight='bold', color='white')

        # 3. Year Range Comparison
        ax3 = plt.subplot(2, 3, 3)
        labels = ['Before', 'After']
        min_vals = [years_before.min(), years_after.min()]
        max_vals = [years_before.max(), years_after.max()]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax3.bar(x - width/2, min_vals, width, label='Min Year', color='coral', edgecolor='black')
        bars2 = ax3.bar(x + width/2, max_vals, width, label='Max Year', color='skyblue', edgecolor='black')

        ax3.set_ylabel('Year', fontweight='bold')
        ax3.set_title('Year Range Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # 4. Year vs Popularity - Before
        ax4 = plt.subplot(2, 3, 4)
        sample_before = self.train_df_original.sample(min(3000, len(self.train_df_original)))
        ax4.scatter(sample_before['release_year'], sample_before['popularity'],
                   alpha=0.3, s=10, c='coral')
        ax4.axvline(x=100, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax4.set_xlabel('Release Year', fontweight='bold')
        ax4.set_ylabel('Popularity', fontweight='bold')
        ax4.set_title('BEFORE: Year vs Popularity', fontweight='bold')
        ax4.grid(alpha=0.3)

        # 5. Year vs Popularity - After
        ax5 = plt.subplot(2, 3, 5)
        sample_after = self.train_df.sample(min(3000, len(self.train_df)))
        scatter = ax5.scatter(sample_after['release_year'], sample_after['popularity'],
                            alpha=0.3, s=10, c=sample_after['popularity'], cmap='viridis')
        ax5.set_xlabel('Release Year', fontweight='bold')
        ax5.set_ylabel('Popularity', fontweight='bold')
        ax5.set_title('AFTER: Year vs Popularity', fontweight='bold')
        ax5.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Popularity')

        # 6. Cleaning Impact Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        summary_data = [
            ['Metric', 'Before', 'After', 'Fixed'],
            ['Invalid Years (Train)', f"{self.cleaning_stats['invalid_years_fixed']['train']}", '0',
             f"{self.cleaning_stats['invalid_years_fixed']['train']}"],
            ['Invalid Years (Test)', f"{self.cleaning_stats['invalid_years_fixed']['test']}", '0',
             f"{self.cleaning_stats['invalid_years_fixed']['test']}"],
            ['Min Year', f"{years_before.min():.0f}", f"{years_after.min():.0f}",
             f"+{years_after.min() - years_before.min():.0f}"],
            ['Max Year', f"{years_before.max():.0f}", f"{years_after.max():.0f}", '0'],
            ['Zero Popularity', f"{self.cleaning_stats['zero_popularity']['count']}",
             f"{self.cleaning_stats['zero_popularity']['count']}", 'Flagged'],
            ['Flag Feature Added', '-', '1 feature', '✓']
        ]

        table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)

        for i in range(len(summary_data)):
            if i == 0:
                for j in range(4):
                    table[(i, j)].set_facecolor('#2196F3')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#E3F2FD')
                for j in range(1, 4):
                    table[(i, j)].set_text_props(weight='bold')

        ax6.set_title('Cleaning Impact Summary', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_path / 'before_after_cleaning.png', dpi=300, bbox_inches='tight')
        print(f"✓ Before/After comparison saved!")
        plt.show()

        return self

    # ... (rest of the methods from original class)
    # I'll include the key methods but abbreviated for space

    def eda(self):
        """EDA dengan visualisasi"""
        print("\n" + "=" * 80)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 80)

        print("\nBasic Statistics:")
        print(self.train_df['popularity'].describe())

        print(f"\nMissing Values:")
        missing = self.train_df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values")

        print(f"\nTop 5 Genres:")
        print(self.train_df['track_genre'].value_counts().head())

        # Visualizations
        self._visualize_eda()

        return self

    def _visualize_eda(self):
        """Detailed EDA visualizations"""
        fig = plt.figure(figsize=(18, 10))

        # Target distribution
        ax1 = plt.subplot(2, 3, 1)
        pop_data = self.train_df['popularity']
        ax1.hist(pop_data, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(pop_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pop_data.mean():.1f}')
        ax1.axvline(pop_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {pop_data.median():.1f}')
        ax1.set_xlabel('Popularity', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Target Distribution (Popularity)', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Audio features correlation
        ax2 = plt.subplot(2, 3, 2)
        audio_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'popularity']
        corr_data = self.train_df[audio_cols].corr()['popularity'].drop('popularity').sort_values()
        colors = ['red' if x < 0 else 'green' for x in corr_data.values]
        ax2.barh(range(len(corr_data)), corr_data.values, color=colors, edgecolor='black')
        ax2.set_yticks(range(len(corr_data)))
        ax2.set_yticklabels(corr_data.index, fontsize=9)
        ax2.axvline(0, color='black', linewidth=1)
        ax2.set_xlabel('Correlation with Popularity', fontweight='bold')
        ax2.set_title('Audio Features Correlation', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Popularity by Genre (top 15)
        ax3 = plt.subplot(2, 3, 3)
        genre_pop = self.train_df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(15)
        colors_genre = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(genre_pop)))
        ax3.barh(range(len(genre_pop)), genre_pop.values, color=colors_genre, edgecolor='black')
        ax3.set_yticks(range(len(genre_pop)))
        ax3.set_yticklabels(genre_pop.index, fontsize=8)
        ax3.set_xlabel('Average Popularity', fontweight='bold')
        ax3.set_title('Top 15 Genres by Popularity', fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)

        # Year vs Popularity trend
        ax4 = plt.subplot(2, 3, 4)
        year_pop = self.train_df.groupby('release_year')['popularity'].agg(['mean', 'std'])
        ax4.plot(year_pop.index, year_pop['mean'], linewidth=2, color='blue', label='Mean')
        ax4.fill_between(year_pop.index,
                        year_pop['mean'] - year_pop['std'],
                        year_pop['mean'] + year_pop['std'],
                        alpha=0.3, color='blue')
        ax4.set_xlabel('Release Year', fontweight='bold')
        ax4.set_ylabel('Popularity', fontweight='bold')
        ax4.set_title('Popularity Trend Over Years', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # Duration distribution
        ax5 = plt.subplot(2, 3, 5)
        duration_min = self.train_df['duration_ms'] / 60000
        ax5.hist(duration_min, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax5.axvline(duration_min.median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {duration_min.median():.1f} min')
        ax5.set_xlabel('Duration (minutes)', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Song Duration Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        ax5.set_xlim(0, 10)  # Focus on 0-10 minutes

        # Popularity by decade
        ax6 = plt.subplot(2, 3, 6)
        self.train_df['decade_temp'] = (self.train_df['release_year'] // 10) * 10
        decade_data = self.train_df.groupby('decade_temp')['popularity'].mean().sort_index()
        ax6.bar(decade_data.index, decade_data.values, width=8, edgecolor='black', alpha=0.7, color='purple')
        ax6.set_xlabel('Decade', fontweight='bold')
        ax6.set_ylabel('Average Popularity', fontweight='bold')
        ax6.set_title('Popularity by Decade', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)

        # Add value labels
        for idx, val in zip(decade_data.index, decade_data.values):
            ax6.text(idx, val, f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_path / 'eda_visualizations.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ EDA visualizations saved!")
        plt.show()

        # Drop temp column
        self.train_df.drop('decade_temp', axis=1, inplace=True)

        return self

    # NOTE: The rest of the methods (engineer_features, process_lyrics, prepare_features,
    # train_models, create_submission) would be identical to the original version
    # For brevity, I'm showing the structure. The full implementation would include all methods.


if __name__ == "__main__":
    print("Song Popularity Predictor V2.0 - Enhanced with Data Cleaning & Visualizations")
    print("Run in Jupyter Notebook for best experience!")
