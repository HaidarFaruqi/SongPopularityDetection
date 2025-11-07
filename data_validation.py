#!/usr/bin/env python3
"""
Data Validation Script
Untuk verify data quality dan assumptions
"""

import pandas as pd
import numpy as np
from pathlib import Path

def validate_data(data_path='/content'):
    """
    Validate data quality dan check assumptions
    """
    print("="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)

    # Load data
    train_df = pd.read_csv(Path(data_path) / 'train.csv', engine='python')

    print(f"\n📊 Dataset Size: {train_df.shape}")
    print(f"   Rows: {len(train_df)}")
    print(f"   Columns: {len(train_df.columns)}")

    # =========================================================================
    # 1. CHECK MISSING VALUES
    # =========================================================================
    print("\n" + "="*80)
    print("1. MISSING VALUES ANALYSIS")
    print("="*80)

    missing = train_df.isnull().sum()
    missing_pct = (missing / len(train_df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Pct': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    if len(missing_df) > 0:
        print("\n⚠️ Columns with missing values:")
        print(missing_df.to_string(index=False))
    else:
        print("\n✅ No missing values found!")

    # =========================================================================
    # 2. TARGET DISTRIBUTION
    # =========================================================================
    print("\n" + "="*80)
    print("2. TARGET DISTRIBUTION (popularity)")
    print("="*80)

    print(f"\nStatistics:")
    print(f"  Mean:   {train_df['popularity'].mean():.2f}")
    print(f"  Median: {train_df['popularity'].median():.2f}")
    print(f"  Std:    {train_df['popularity'].std():.2f}")
    print(f"  Min:    {train_df['popularity'].min():.0f}")
    print(f"  Max:    {train_df['popularity'].max():.0f}")

    # Check distribution bins
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    train_df['pop_bin'] = pd.cut(train_df['popularity'], bins=bins, labels=labels)

    print(f"\nDistribution by bins:")
    print(train_df['pop_bin'].value_counts().sort_index())

    # =========================================================================
    # 3. CATEGORICAL FEATURES ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("3. CATEGORICAL FEATURES ANALYSIS")
    print("="*80)

    # Genre
    print(f"\nGenre (track_genre):")
    print(f"  Unique genres: {train_df['track_genre'].nunique()}")
    print(f"  Top 10 genres:")
    print(train_df['track_genre'].value_counts().head(10))

    # Artists
    print(f"\nArtists:")
    print(f"  Unique artists: {train_df['artists'].nunique()}")
    print(f"  Artists with 1 song: {(train_df['artists'].value_counts() == 1).sum()}")
    print(f"  Artists with >10 songs: {(train_df['artists'].value_counts() > 10).sum()}")

    # =========================================================================
    # 4. AUDIO FEATURES CORRELATION
    # =========================================================================
    print("\n" + "="*80)
    print("4. AUDIO FEATURES CORRELATION WITH TARGET")
    print("="*80)

    audio_features = ['energy', 'danceability', 'valence', 'loudness', 'tempo',
                     'acousticness', 'speechiness', 'instrumentalness', 'liveness']

    audio_features = [f for f in audio_features if f in train_df.columns]

    correlations = train_df[audio_features + ['popularity']].corr()['popularity'].sort_values(ascending=False)

    print(f"\nCorrelation with popularity:")
    for feat, corr in correlations.items():
        if feat != 'popularity':
            indicator = "🔴" if abs(corr) > 0.3 else "🟡" if abs(corr) > 0.1 else "🟢"
            print(f"  {indicator} {feat:20s}: {corr:+.4f}")

    # =========================================================================
    # 5. TEMPORAL ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("5. TEMPORAL ANALYSIS (release_year)")
    print("="*80)

    if 'release_year' in train_df.columns:
        print(f"\nYear range: {train_df['release_year'].min()} - {train_df['release_year'].max()}")

        # Popularity by decade
        train_df['decade'] = (train_df['release_year'] // 10) * 10
        decade_stats = train_df.groupby('decade')['popularity'].agg(['mean', 'std', 'count'])

        print(f"\nPopularity by decade:")
        print(decade_stats)

    # =========================================================================
    # 6. LYRICS ANALYSIS (if available)
    # =========================================================================
    print("\n" + "="*80)
    print("6. LYRICS ANALYSIS")
    print("="*80)

    if 'lyrics' in train_df.columns:
        lyrics_present = train_df['lyrics'].notna().sum()
        lyrics_pct = (lyrics_present / len(train_df)) * 100

        print(f"\n  Songs with lyrics: {lyrics_present} ({lyrics_pct:.1f}%)")
        print(f"  Songs without lyrics: {len(train_df) - lyrics_present}")

        # Average lyrics length
        if lyrics_present > 0:
            train_df['lyrics_length'] = train_df['lyrics'].fillna('').str.len()
            print(f"\n  Avg lyrics length: {train_df['lyrics_length'].mean():.0f} characters")
            print(f"  Median lyrics length: {train_df['lyrics_length'].median():.0f} characters")
    else:
        print("\n  ⚠️ No lyrics column found")

    # =========================================================================
    # 7. DATA QUALITY ISSUES
    # =========================================================================
    print("\n" + "="*80)
    print("7. DATA QUALITY CHECKS")
    print("="*80)

    issues = []

    # Check for duplicates
    duplicates = train_df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️ Found {duplicates} duplicate rows")

    # Check for outliers in audio features
    for feat in audio_features:
        if feat in train_df.columns:
            # Most audio features should be [0, 1] range
            if feat in ['energy', 'danceability', 'valence', 'acousticness',
                       'speechiness', 'instrumentalness', 'liveness']:
                outliers = ((train_df[feat] < 0) | (train_df[feat] > 1)).sum()
                if outliers > 0:
                    issues.append(f"⚠️ {feat}: {outliers} values outside [0,1] range")

    # Check for negative duration
    if 'duration_ms' in train_df.columns:
        neg_duration = (train_df['duration_ms'] <= 0).sum()
        if neg_duration > 0:
            issues.append(f"⚠️ Found {neg_duration} songs with negative/zero duration")

    if len(issues) > 0:
        print("\n⚠️ Data Quality Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No data quality issues detected!")

    # =========================================================================
    # 8. FEATURE ENGINEERING OPPORTUNITIES
    # =========================================================================
    print("\n" + "="*80)
    print("8. FEATURE ENGINEERING OPPORTUNITIES")
    print("="*80)

    print("\n✅ Available for new features:")

    # Artist-based features
    artist_counts = train_df['artists'].value_counts()
    print(f"  • Artist statistics possible: {(artist_counts > 1).sum()} artists with multiple songs")

    # Genre-based features
    genre_counts = train_df['track_genre'].value_counts()
    print(f"  • Genre statistics possible: {len(genre_counts)} unique genres")

    # Year-based features
    if 'release_year' in train_df.columns:
        year_counts = train_df['release_year'].value_counts()
        print(f"  • Year statistics possible: {len(year_counts)} unique years")

    # Track name features
    if 'track_name' in train_df.columns:
        has_feat = train_df['track_name'].str.contains('feat|ft\.|featuring', case=False, na=False).sum()
        has_remix = train_df['track_name'].str.contains('remix|remaster', case=False, na=False).sum()
        print(f"  • Track name patterns: {has_feat} with featuring, {has_remix} remixes/remasters")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    return train_df

if __name__ == "__main__":
    # Run validation
    # Sesuaikan data_path dengan lokasi data Anda
    df = validate_data(data_path='/content')
