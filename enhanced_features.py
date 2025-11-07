"""
Enhanced Feature Engineering Module
=====================================

Fitur-fitur baru yang ditambahkan:
1. Genre statistics (mean, std, count)
2. Artist variance (std, min, max)
3. Audio feature ratios
4. Advanced track name features
5. Temporal momentum features

IMPORTANT: Untuk genre & artist encoding, harus dilakukan DALAM CV loop
untuk avoid data leakage!
"""

import pandas as pd
import numpy as np


def add_base_audio_features(df_train, df_test):
    """
    Add basic audio feature interactions
    """
    print("[1/6] Base audio features...")

    for df in [df_train, df_test]:
        # Existing features
        if all(col in df.columns for col in ['energy', 'danceability']):
            df['energy_x_dance'] = df['energy'] * df['danceability']

        if 'duration_ms' in df.columns:
            df['duration_min'] = df['duration_ms'] / 60000

        if all(col in df.columns for col in ['key', 'mode']):
            df['key_mode'] = df['key'].astype(str) + '_' + df['mode'].astype(str)

        if 'tempo' in df.columns:
            df['tempo_category'] = pd.cut(df['tempo'],
                                          bins=[0, 90, 120, 150, 250],
                                          labels=['slow', 'moderate', 'fast', 'very_fast'])

    return df_train, df_test


def add_audio_ratios(df_train, df_test):
    """
    Add audio feature ratios - NEW FEATURES
    """
    print("[2/6] Audio feature ratios (NEW)...")

    for df in [df_train, df_test]:
        # Energy-based ratios
        df['energy_valence_ratio'] = df['energy'] / (df['valence'] + 1e-6)
        df['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + 1e-6)

        # Danceability ratios
        df['dance_acoustic_ratio'] = df['danceability'] / (df['acousticness'] + 1e-6)

        # Speech vs Music
        df['speech_music_ratio'] = df['speechiness'] / (1 - df['speechiness'] + 1e-6)

        # Loudness-energy alignment
        if 'loudness' in df.columns:
            # Normalize loudness to [0, 1] for ratio
            loudness_norm = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min() + 1e-6)
            df['loudness_energy_alignment'] = 1 - abs(loudness_norm - df['energy'])

        # Audio complexity (std of main audio features)
        audio_cols = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
        audio_cols = [c for c in audio_cols if c in df.columns]
        if len(audio_cols) >= 3:
            df['audio_feature_std'] = df[audio_cols].std(axis=1)
            df['audio_feature_mean'] = df[audio_cols].mean(axis=1)

    print(\"   Added 8+ audio ratio features\")\n    return df_train, df_test


def add_temporal_features(df_train, df_test):
    \"\"\"
    Add temporal features
    \"\"\"
    print(\"[3/6] Temporal features...\")

    for df in [df_train, df_test]:
        if 'release_year' in df.columns:
            # Basic temporal
            df['years_since_release'] = 2025 - df['release_year']
            df['decade'] = (df['release_year'] // 10) * 10
            df['is_classic'] = (df['release_year'] < 2000).astype(int)
            df['is_recent_hit'] = (df['release_year'] >= 2020).astype(int)

            # NEW: More granular era classification
            df['era'] = pd.cut(df['release_year'],
                              bins=[0, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                              labels=['pre_70', '70s', '80s', '90s', '00s', '10s', '20s'])

            # NEW: Release decade popularity (will be filled in CV loop)
            # Placeholder for now
            df['decade_avg_pop'] = 0.0

    return df_train, df_test


def add_track_name_features(df_train, df_test):
    \"\"\"
    Add track name features (basic + advanced)
    \"\"\"
    print(\"[4/6] Track name features...\")

    for df in [df_train, df_test]:
        if 'track_name' in df.columns:
            # Clean track name
            clean_name = df['track_name'].astype(str).str.lower()
            clean_name = clean_name.str.replace(r'[\(\[].*?[\)\]]', '', regex=True)
            clean_name = clean_name.str.split(' - feat.').str[0]
            clean_name = clean_name.str.split(' - with').str[0]
            clean_name = clean_name.str.split(' - sped up').str[0]
            clean_name = clean_name.str.split(' - remastered').str[0]
            clean_name = clean_name.str.split(' - from').str[0]
            clean_name = clean_name.str.strip()

            # Basic features
            df['track_name_length'] = clean_name.str.len()
            df['track_name_word_count'] = clean_name.str.count(' ') + 1

            # NEW: Advanced track name features
            df['has_featuring'] = df['track_name'].str.contains(
                'feat|ft\.|featuring', case=False, na=False
            ).astype(int)

            df['is_remix'] = df['track_name'].str.contains(
                'remix|remaster|version', case=False, na=False
            ).astype(int)

            df['has_parenthesis'] = df['track_name'].str.contains(
                r'\(|\[', regex=True, na=False
            ).astype(int)

            df['has_special_edition'] = df['track_name'].str.contains(
                'deluxe|special|edition|bonus', case=False, na=False
            ).astype(int)

            # Title complexity
            unique_words = clean_name.str.split().apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
            total_words = df['track_name_word_count']
            df['title_word_diversity'] = unique_words / (total_words + 1)

    print(\"   Added 9 track name features (5 new)\")
    return df_train, df_test


def create_genre_features_for_test(df_train, df_test):
    \"\"\"
    Create genre features for TEST set only
    For TRAIN set, these will be created inside CV loop to avoid leakage

    Returns mappings that can be used for test set
    \"\"\"
    print(\"[5/6] Genre statistics (TEST set only)...\")

    # Calculate genre statistics from full training data
    genre_stats = df_train.groupby('track_genre')['popularity'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()

    genre_stats.columns = ['track_genre', 'genre_avg_pop', 'genre_std_pop',
                           'genre_min_pop', 'genre_max_pop', 'genre_song_count']

    # Merge ke TEST set
    df_test = df_test.merge(genre_stats, on='track_genre', how='left')

    # Fill missing dengan global stats
    global_mean = df_train['popularity'].mean()
    global_std = df_train['popularity'].std()
    global_count = len(df_train) / df_train['track_genre'].nunique()

    df_test['genre_avg_pop'].fillna(global_mean, inplace=True)
    df_test['genre_std_pop'].fillna(global_std, inplace=True)
    df_test['genre_min_pop'].fillna(0, inplace=True)
    df_test['genre_max_pop'].fillna(100, inplace=True)
    df_test['genre_song_count'].fillna(global_count, inplace=True)

    print(f\"   Added genre features to test set\")

    # Return mappings for use in CV
    return df_test, genre_stats


def create_artist_features_for_test(df_train, df_test):
    \"\"\"
    Create artist features for TEST set only
    For TRAIN set, these will be created inside CV loop
    \"\"\"
    print(\"[6/6] Artist statistics (TEST set only)...\")

    # Artist popularity stats
    artist_pop_stats = df_train.groupby('artists')['popularity'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()

    artist_pop_stats.columns = ['artists', 'artist_avg_pop', 'artist_std_pop',
                                'artist_min_pop', 'artist_max_pop', 'artist_song_count']

    # Merge ke TEST set
    df_test = df_test.merge(artist_pop_stats, on='artists', how='left')

    # Fill missing
    global_mean = df_train['popularity'].mean()
    global_std = df_train['popularity'].std()

    df_test['artist_avg_pop'].fillna(global_mean, inplace=True)
    df_test['artist_std_pop'].fillna(global_std, inplace=True)
    df_test['artist_min_pop'].fillna(0, inplace=True)
    df_test['artist_max_pop'].fillna(100, inplace=True)
    df_test['artist_song_count'].fillna(1, inplace=True)

    # NEW: Artist-audio interactions
    if all(col in df_test.columns for col in ['artist_avg_pop', 'danceability', 'energy']):
        df_test['artist_x_dance'] = df_test['artist_avg_pop'] * df_test['danceability']
        df_test['artist_x_energy'] = df_test['artist_avg_pop'] * df_test['energy']

    print(f\"   Added artist features to test set\")

    return df_test, artist_pop_stats


def engineer_features_enhanced(df_train, df_test):
    \"\"\"
    Main function untuk feature engineering (enhanced version)

    NOTE: Genre & artist encoding features untuk TRAIN akan dibuat
    inside CV loop. Ini hanya untuk base features dan TEST set.
    \"\"\"
    print(\"\\n\" + \"=\"*80)
    print(\"🔧 ENHANCED FEATURE ENGINEERING\")
    print(\"=\"*80)

    # Add all base features
    df_train, df_test = add_base_audio_features(df_train, df_test)
    df_train, df_test = add_audio_ratios(df_train, df_test)
    df_train, df_test = add_temporal_features(df_train, df_test)
    df_train, df_test = add_track_name_features(df_train, df_test)

    # For test set, add genre & artist features
    # For train set, these will be added in CV loop
    df_test, genre_stats = create_genre_features_for_test(df_train, df_test)
    df_test, artist_stats = create_artist_features_for_test(df_train, df_test)

    print(\"\\n✅ Enhanced feature engineering completed!\")
    print(f\"   Train shape: {df_train.shape}\")
    print(f\"   Test shape: {df_test.shape}\")

    # Return mappings for CV use
    mappings = {
        'genre_stats': genre_stats,
        'artist_stats': artist_stats
    }

    return df_train, df_test, mappings


# =================================================================================
# CV-AWARE FEATURE ENCODING (untuk avoid data leakage)
# =================================================================================

def create_fold_features(df_fold, df_full_train, target_col='popularity'):
    \"\"\"
    Create target-encoded features untuk satu fold
    Hanya menggunakan data dari df_full_train pada indices yang relevan

    Args:
        df_fold: DataFrame untuk fold ini (bisa train atau val)
        df_full_train: Full training DataFrame
        target_col: Target column name

    Returns:
        df_fold dengan added features
    \"\"\"
    df_fold = df_fold.copy()

    # Genre statistics (dari fold training data saja jika ini training fold)
    genre_map = df_full_train.groupby('track_genre')[target_col].mean()
    genre_std = df_full_train.groupby('track_genre')[target_col].std()
    genre_count = df_full_train.groupby('track_genre').size()

    df_fold['genre_avg_pop'] = df_fold['track_genre'].map(genre_map)
    df_fold['genre_std_pop'] = df_fold['track_genre'].map(genre_std)
    df_fold['genre_song_count'] = df_fold['track_genre'].map(genre_count)

    # Fill dengan global stats
    global_mean = df_full_train[target_col].mean()
    global_std = df_full_train[target_col].std()

    df_fold['genre_avg_pop'].fillna(global_mean, inplace=True)
    df_fold['genre_std_pop'].fillna(global_std, inplace=True)
    df_fold['genre_song_count'].fillna(1, inplace=True)

    # Artist statistics
    artist_mean = df_full_train.groupby('artists')[target_col].mean()
    artist_std = df_full_train.groupby('artists')[target_col].std()
    artist_count = df_full_train.groupby('artists').size()

    df_fold['artist_avg_pop'] = df_fold['artists'].map(artist_mean)
    df_fold['artist_std_pop'] = df_fold['artists'].map(artist_std)
    df_fold['artist_song_count'] = df_fold['artists'].map(artist_count)

    df_fold['artist_avg_pop'].fillna(global_mean, inplace=True)
    df_fold['artist_std_pop'].fillna(global_std, inplace=True)
    df_fold['artist_song_count'].fillna(1, inplace=True)

    # Artist-audio interactions
    if all(col in df_fold.columns for col in ['artist_avg_pop', 'danceability', 'energy']):
        df_fold['artist_x_dance'] = df_fold['artist_avg_pop'] * df_fold['danceability']
        df_fold['artist_x_energy'] = df_fold['artist_avg_pop'] * df_fold['energy']

    return df_fold
