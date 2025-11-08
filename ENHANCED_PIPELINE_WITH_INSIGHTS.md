# 📊 Enhanced Pipeline dengan Data Insights & Feature Selection

## 🎯 Overview

Pipeline ini menambahkan:
1. **Data Profiling** - Deteksi awal kualitas data
2. **Pre-processing Insights** - Visualisasi sebelum pemrosesan
3. **Automatic Feature Selection** - Reduce 67 → 40-50 features
4. **Post-processing Insights** - Visualisasi hasil
5. **Improvement Tracking** - Before/After comparison

## 📋 Struktur Alur Baru

```
┌─────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION & INITIAL DETECTION             │
│    - Load data                                       │
│    - Check missing values, duplicates               │
│    - Data types validation                          │
│    - Basic statistics                               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. PRE-PROCESSING INSIGHTS (VISUAL)                │
│    - Correlation heatmap (target vs features)      │
│    - Distribution plots (numerical features)        │
│    - Categorical feature analysis                   │
│    - Missing value patterns                         │
│    - Outlier detection                              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING                              │
│    - Create new features                            │
│    - Encode categorical                             │
│    - Process lyrics (NLP)                           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4. FEATURE SELECTION (AUTOMATIC)                    │
│    - Train initial model                            │
│    - Analyze feature importance                     │
│    - Remove low-importance features (<1%)           │
│    - Keep top 40-50 features                        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 5. MODEL TRAINING (PROPER CV)                       │
│    - 5-Fold Cross-Validation                        │
│    - Track per-fold metrics                         │
│    - Save OOF predictions                           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 6. POST-PROCESSING INSIGHTS (VISUAL)                │
│    - Feature importance plot (top 20)               │
│    - CV scores distribution                         │
│    - Prediction vs actual scatter plot              │
│    - Residual analysis                              │
│    - Learning curves                                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 7. FINAL RESULTS & TRACKING                         │
│    - Before vs After comparison                     │
│    - Improvement metrics                            │
│    - Feature reduction summary                      │
│    - Submission file                                │
└─────────────────────────────────────────────────────┘
```

## 🔧 Key Changes

### 1. Data Profiling (NEW!)

**Cell baru setelah load data:**

```python
# === CELL: DATA PROFILING & INITIAL DETECTION ===

def data_profiling(df, name="Data"):
    """Comprehensive data profiling"""
    print(f"\n{'='*80}")
    print(f"📊 DATA PROFILING: {name}")
    print(f"{'='*80}")

    # Basic info
    print(f"\n[1] Basic Information:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

    # Data types
    print(f"\n[2] Data Types:")
    print(df.dtypes.value_counts())

    # Missing values
    print(f"\n[3] Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)

    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("  ✅ No missing values!")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n[4] Duplicates:")
    print(f"  Total: {duplicates} ({duplicates/len(df)*100:.2f}%)")

    # Numerical features stats
    print(f"\n[5] Numerical Features Statistics:")
    numerical = df.select_dtypes(include=[np.number])
    print(f"  Count: {len(numerical.columns)}")

    # Categorical features
    print(f"\n[6] Categorical Features:")
    categorical = df.select_dtypes(include=['object', 'category'])
    print(f"  Count: {len(categorical.columns)}")
    for col in categorical.columns:
        print(f"    {col}: {df[col].nunique()} unique values")

    # Potential issues
    print(f"\n[7] Potential Issues:")
    issues = []

    # High cardinality
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > 100:
            issues.append(f"  ⚠️  {col}: High cardinality ({df[col].nunique()} unique)")

    # Constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"  ⚠️  {col}: Constant column")

    # High missing rate
    for col in missing_df.index:
        if missing_pct[col] > 50:
            issues.append(f"  ⚠️  {col}: High missing rate ({missing_pct[col]:.1f}%)")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ No critical issues detected!")

    return missing_df

# Profile both datasets
train_profile = data_profiling(df_train, "Training Data")
test_profile = data_profiling(df_test, "Test Data")
```

### 2. Pre-processing Insights (NEW!)

**Cell baru dengan visualizations:**

```python
# === CELL: PRE-PROCESSING INSIGHTS (VISUAL) ===

def create_preprocessing_insights(df, target_col='popularity'):
    """Create comprehensive pre-processing visualizations"""

    print(f"\n{'='*80}")
    print(f"📈 PRE-PROCESSING INSIGHTS")
    print(f"{'='*80}")

    # Setup figure
    fig = plt.figure(figsize=(20, 15))

    # 1. Target Distribution
    print("\n[1/6] Creating target distribution plot...")
    ax1 = plt.subplot(3, 3, 1)
    df[target_col].hist(bins=50, ax=ax1, edgecolor='black')
    ax1.axvline(df[target_col].mean(), color='red', linestyle='--', label=f'Mean: {df[target_col].mean():.1f}')
    ax1.axvline(df[target_col].median(), color='green', linestyle='--', label=f'Median: {df[target_col].median():.1f}')
    ax1.set_xlabel(target_col.capitalize())
    ax1.set_ylabel('Frequency')
    ax1.set_title('Target Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Correlation Heatmap (Top features)
    print("[2/6] Creating correlation heatmap...")
    ax2 = plt.subplot(3, 3, 2)

    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove non-features
    numerical_features = [f for f in numerical_features if f not in ['track_id', 'Unnamed: 0']]

    # Get top correlated features with target
    correlations = df[numerical_features].corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.head(11).index.tolist()  # Top 10 + target

    # Create heatmap
    corr_matrix = df[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Top 10 Features - Correlation Heatmap')

    # 3. Audio Features Distribution
    print("[3/6] Creating audio features distribution...")
    ax3 = plt.subplot(3, 3, 3)

    audio_features = ['energy', 'danceability', 'valence', 'acousticness']
    audio_features = [f for f in audio_features if f in df.columns]

    for feat in audio_features:
        ax3.hist(df[feat], bins=30, alpha=0.5, label=feat)
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Audio Features Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Genre Analysis
    print("[4/6] Creating genre analysis...")
    ax4 = plt.subplot(3, 3, 4)

    if 'track_genre' in df.columns:
        genre_pop = df.groupby('track_genre')[target_col].mean().sort_values(ascending=False).head(15)
        genre_pop.plot(kind='barh', ax=ax4, color='steelblue')
        ax4.set_xlabel('Average Popularity')
        ax4.set_title('Top 15 Genres by Popularity')
        ax4.grid(True, alpha=0.3, axis='x')

    # 5. Release Year Trend
    print("[5/6] Creating temporal trend...")
    ax5 = plt.subplot(3, 3, 5)

    if 'release_year' in df.columns:
        year_pop = df.groupby('release_year')[target_col].mean()
        ax5.plot(year_pop.index, year_pop.values, linewidth=2, color='darkgreen')
        ax5.set_xlabel('Release Year')
        ax5.set_ylabel('Average Popularity')
        ax5.set_title('Popularity Trend Over Time')
        ax5.grid(True, alpha=0.3)

    # 6. Feature Correlation with Target (Bar plot)
    print("[6/6] Creating feature correlation bar plot...")
    ax6 = plt.subplot(3, 3, 6)

    top_corr = correlations.head(16)[1:]  # Skip target itself
    top_corr.plot(kind='barh', ax=ax6, color='coral')
    ax6.set_xlabel('Absolute Correlation')
    ax6.set_title('Top 15 Features - Correlation with Target')
    ax6.grid(True, alpha=0.3, axis='x')

    # 7. Missing Values Pattern
    print("[7/6] Creating missing values pattern...")
    ax7 = plt.subplot(3, 3, 7)

    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        missing_pct = (missing_cols / len(df)) * 100
        missing_pct.plot(kind='barh', ax=ax7, color='red', alpha=0.7)
        ax7.set_xlabel('Missing %')
        ax7.set_title('Missing Values by Feature')
        ax7.grid(True, alpha=0.3, axis='x')
    else:
        ax7.text(0.5, 0.5, '✅ No Missing Values',
                ha='center', va='center', fontsize=14, transform=ax7.transAxes)
        ax7.axis('off')

    # 8. Outliers Detection (Box plot)
    print("[8/6] Creating outlier detection plot...")
    ax8 = plt.subplot(3, 3, 8)

    # Select a few key features for outlier detection
    key_features = ['energy', 'danceability', 'loudness', target_col]
    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        df[key_features].boxplot(ax=ax8)
        ax8.set_ylabel('Value')
        ax8.set_title('Outlier Detection (Key Features)')
        ax8.grid(True, alpha=0.3, axis='y')

    # 9. Target vs Top Feature Scatter
    print("[9/6] Creating scatter plot...")
    ax9 = plt.subplot(3, 3, 9)

    if len(top_corr) > 0:
        top_feature = top_corr.index[0]
        ax9.scatter(df[top_feature], df[target_col], alpha=0.3, s=10)
        ax9.set_xlabel(top_feature)
        ax9.set_ylabel(target_col.capitalize())
        ax9.set_title(f'{target_col} vs {top_feature}\n(Corr: {correlations[top_feature]:.3f})')
        ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/preprocessing_insights.png', dpi=100, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {OUTPUT_PATH}/preprocessing_insights.png")
    plt.show()

    # Print insights summary
    print(f"\n{'='*80}")
    print("📊 KEY INSIGHTS:")
    print(f"{'='*80}")
    print(f"\n1. Target Distribution:")
    print(f"   Mean: {df[target_col].mean():.2f}, Median: {df[target_col].median():.2f}, Std: {df[target_col].std():.2f}")
    print(f"   Skewness: {df[target_col].skew():.2f}")

    print(f"\n2. Top 5 Correlated Features:")
    for i, (feat, corr) in enumerate(top_corr.head(5).items(), 1):
        print(f"   {i}. {feat}: {corr:.4f}")

    print(f"\n3. Data Quality:")
    print(f"   Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)")
    print(f"   Duplicate rows: {df.duplicated().sum()}")

    return top_corr

# Create insights
top_correlations = create_preprocessing_insights(df_train)
```

### 3. Automatic Feature Selection (NEW!)

**Cell baru untuk feature selection:**

```python
# === CELL: AUTOMATIC FEATURE SELECTION ===

def automatic_feature_selection(df_train, base_features, target_col='popularity',
                                target_num_features=50):
    """
    Automatically select top features based on importance

    Args:
        df_train: Training dataframe
        base_features: List of all features
        target_col: Target column
        target_num_features: Target number of features to keep

    Returns:
        selected_features: List of selected features
        feature_importance: DataFrame with importance scores
    """

    print(f"\n{'='*80}")
    print(f"🎯 AUTOMATIC FEATURE SELECTION")
    print(f"{'='*80}")

    print(f"\nCurrent features: {len(base_features)}")
    print(f"Target features: {target_num_features}")

    # Quick 3-fold CV for feature selection
    print(f"\n[1/3] Training quick model for feature importance...")

    X = df_train[base_features].copy()
    y = df_train[target_col].copy()

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    feature_importance_list = []

    lgbm_params = {
        'n_estimators': 500,  # Fast training
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()

        # Create fold features (with target encoding)
        df_train_fold = create_fold_features(df_train_fold, df_train_fold, target_col)
        df_val_fold = create_fold_features(df_val_fold, df_train_fold, target_col)

        # Get all features (including target-encoded)
        all_features = [f for f in df_train_fold.columns
                       if f not in [target_col, 'track_id', 'track_name', 'artists',
                                   'lyrics', 'release_year', 'track_genre', 'decade', 'era',
                                   'key_mode', 'tempo_category']]

        X_train_fold = df_train_fold[all_features]
        y_train_fold = df_train_fold[target_col]

        # Train
        model = LGBMRegressor(**lgbm_params)

        if LIGHTGBM_NEW_API:
            model.fit(X_train_fold, y_train_fold,
                     callbacks=[log_evaluation(period=0)])
        else:
            model.fit(X_train_fold, y_train_fold, verbose=False)

        # Store importance
        fi_df = pd.DataFrame({
            'feature': all_features,
            'importance': model.feature_importances_,
            'fold': fold
        })
        feature_importance_list.append(fi_df)

        print(f"  Fold {fold}/3 complete")

    # Aggregate importance
    print(f"\n[2/3] Analyzing feature importance...")
    fi_all = pd.concat(feature_importance_list)
    fi_summary = fi_all.groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean', ascending=False)

    # Calculate importance percentage
    fi_summary['importance_pct'] = (fi_summary['mean'] / fi_summary['mean'].sum()) * 100
    fi_summary['cumulative_pct'] = fi_summary['importance_pct'].cumsum()

    # Select features
    print(f"\n[3/3] Selecting top {target_num_features} features...")

    # Method 1: Top N features
    selected_features = fi_summary.head(target_num_features).index.tolist()

    # Method 2: Keep features with importance > threshold
    importance_threshold = 0.5  # Keep features with >0.5% importance
    high_importance = fi_summary[fi_summary['importance_pct'] > importance_threshold].index.tolist()

    print(f"\n📊 Feature Selection Results:")
    print(f"  Method 1 (Top {target_num_features}): {len(selected_features)} features")
    print(f"  Method 2 (Importance >{importance_threshold}%): {len(high_importance)} features")

    # Use more conservative approach
    if len(high_importance) < target_num_features:
        final_features = high_importance
        print(f"\n✅ Using Method 2: {len(final_features)} features (more conservative)")
    else:
        final_features = selected_features
        print(f"\n✅ Using Method 1: {len(final_features)} features")

    # Show dropped features
    dropped_features = [f for f in base_features if f not in final_features]
    print(f"\n❌ Dropped {len(dropped_features)} features:")
    for feat in dropped_features[:10]:
        imp_pct = fi_summary.loc[feat, 'importance_pct'] if feat in fi_summary.index else 0
        print(f"  - {feat}: {imp_pct:.3f}% importance")
    if len(dropped_features) > 10:
        print(f"  ... and {len(dropped_features) - 10} more")

    # Visualization
    plt.figure(figsize=(14, 6))

    # Plot 1: Top features
    plt.subplot(1, 2, 1)
    top_20 = fi_summary.head(20)
    plt.barh(range(len(top_20)), top_20['mean'], color='steelblue')
    plt.yticks(range(len(top_20)), top_20.index)
    plt.xlabel('Importance')
    plt.title('Top 20 Features by Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Plot 2: Cumulative importance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(fi_summary)+1), fi_summary['cumulative_pct'].values, linewidth=2)
    plt.axhline(80, color='red', linestyle='--', label='80% threshold')
    plt.axhline(90, color='orange', linestyle='--', label='90% threshold')
    plt.axvline(target_num_features, color='green', linestyle='--', label=f'Target: {target_num_features}')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance %')
    plt.title('Cumulative Feature Importance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/feature_selection.png', dpi=100, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {OUTPUT_PATH}/feature_selection.png")
    plt.show()

    return final_features, fi_summary

# Perform feature selection
selected_features, feature_importance_df = automatic_feature_selection(
    df_train,
    base_features,
    target_col='popularity',
    target_num_features=50  # Target: keep 50 best features
)

# Update base_features
print(f"\n{'='*80}")
print(f"✅ FEATURE SELECTION COMPLETE")
print(f"{'='*80}")
print(f"  Before: {len(base_features)} features")
print(f"  After:  {len(selected_features)} features")
print(f"  Reduction: {len(base_features) - len(selected_features)} features ({(len(base_features) - len(selected_features))/len(base_features)*100:.1f}%)")

# Use selected features going forward
base_features = selected_features
```

### 4. Post-processing Insights (NEW!)

**Cell baru setelah training:**

```python
# === CELL: POST-PROCESSING INSIGHTS (VISUAL) ===

def create_postprocessing_insights(results, df_train, oof_predictions):
    """Create comprehensive post-processing visualizations"""

    print(f"\n{'='*80}")
    print(f"📈 POST-PROCESSING INSIGHTS")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(20, 12))

    # 1. Feature Importance (Top 20)
    print("\n[1/6] Creating feature importance plot...")
    ax1 = plt.subplot(3, 3, 1)
    top_20_features = results['feature_importance'].head(20)
    ax1.barh(range(len(top_20_features)), top_20_features['mean'], color='steelblue')
    ax1.set_yticks(range(len(top_20_features)))
    ax1.set_yticklabels(top_20_features.index, fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 20 Features by Importance')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. CV Scores Distribution
    print("[2/6] Creating CV scores plot...")
    ax2 = plt.subplot(3, 3, 2)
    cv_scores = results['cv_scores']
    x_pos = range(1, len(cv_scores) + 1)
    ax2.bar(x_pos, cv_scores, color='coral', alpha=0.7, edgecolor='black')
    ax2.axhline(cv_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {cv_scores.mean():.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Cross-Validation Scores by Fold')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Prediction vs Actual (Scatter)
    print("[3/6] Creating prediction vs actual plot...")
    ax3 = plt.subplot(3, 3, 3)
    y_true = df_train['popularity'].values
    ax3.scatter(y_true, oof_predictions, alpha=0.3, s=10)
    # Perfect prediction line
    min_val = min(y_true.min(), oof_predictions.min())
    max_val = max(y_true.max(), oof_predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Popularity')
    ax3.set_ylabel('Predicted Popularity')
    ax3.set_title(f'Predictions vs Actual\n(R² = {results["overall_r2"]:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Residual Plot
    print("[4/6] Creating residual plot...")
    ax4 = plt.subplot(3, 3, 4)
    residuals = y_true - oof_predictions
    ax4.scatter(oof_predictions, residuals, alpha=0.3, s=10)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Popularity')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot')
    ax4.grid(True, alpha=0.3)

    # 5. Residual Distribution
    print("[5/6] Creating residual distribution...")
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Residual Distribution\n(Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f})')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Feature Importance by Category
    print("[6/6] Creating feature category breakdown...")
    ax6 = plt.subplot(3, 3, 6)

    # Categorize features
    fi = results['feature_importance']
    categories = {
        'Target Encoded': [],
        'Audio Features': [],
        'Temporal': [],
        'Lyrics': [],
        'Track Name': [],
        'Other': []
    }

    for feat in fi.index:
        if 'artist' in feat or 'genre' in feat:
            categories['Target Encoded'].append(fi.loc[feat, 'mean'])
        elif 'lyrics' in feat:
            categories['Lyrics'].append(fi.loc[feat, 'mean'])
        elif any(x in feat for x in ['year', 'decade', 'classic', 'recent']):
            categories['Temporal'].append(fi.loc[feat, 'mean'])
        elif any(x in feat for x in ['track_name', 'has_', 'is_', 'title']):
            categories['Track Name'].append(fi.loc[feat, 'mean'])
        elif any(x in feat for x in ['energy', 'dance', 'valence', 'loudness', 'tempo',
                                      'acoustic', 'speech', 'instrument', 'liveness', 'ratio']):
            categories['Audio Features'].append(fi.loc[feat, 'mean'])
        else:
            categories['Other'].append(fi.loc[feat, 'mean'])

    category_importance = {k: sum(v) for k, v in categories.items() if len(v) > 0}

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    ax6.pie(category_importance.values(), labels=category_importance.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax6.set_title('Feature Importance by Category')

    # 7. CV Score Stability
    print("[7/6] Creating CV stability plot...")
    ax7 = plt.subplot(3, 3, 7)
    ax7.boxplot([cv_scores], labels=['CV Scores'])
    ax7.set_ylabel('RMSE')
    ax7.set_title(f'CV Score Stability\n(Std: {cv_scores.std():.4f})')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Error by Prediction Range
    print("[8/6] Creating error by range plot...")
    ax8 = plt.subplot(3, 3, 8)

    # Bin predictions
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    pred_bins = pd.cut(oof_predictions, bins=bins, labels=labels)

    errors_by_bin = []
    for label in labels:
        mask = pred_bins == label
        if mask.sum() > 0:
            errors_by_bin.append(np.abs(residuals[mask]).mean())
        else:
            errors_by_bin.append(0)

    ax8.bar(labels, errors_by_bin, color='lightcoral', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Prediction Range')
    ax8.set_ylabel('Mean Absolute Error')
    ax8.set_title('Error Distribution by Prediction Range')
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Top 10 Features Comparison
    print("[9/6] Creating feature comparison...")
    ax9 = plt.subplot(3, 3, 9)

    top_10 = results['feature_importance'].head(10)
    total_imp = results['feature_importance']['mean'].sum()
    top_10_pct = (top_10['mean'] / total_imp) * 100

    ax9.barh(range(len(top_10)), top_10_pct.values, color='#2ECC71')
    ax9.set_yticks(range(len(top_10)))
    ax9.set_yticklabels(top_10.index, fontsize=9)
    ax9.set_xlabel('Importance %')
    ax9.set_title(f'Top 10 Features (Total: {top_10_pct.sum():.1f}%)')
    ax9.invert_yaxis()
    ax9.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/postprocessing_insights.png', dpi=100, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {OUTPUT_PATH}/postprocessing_insights.png")
    plt.show()

    # Print insights
    print(f"\n{'='*80}")
    print("📊 KEY INSIGHTS:")
    print(f"{'='*80}")

    print(f"\n1. Model Performance:")
    print(f"   OOF RMSE: {results['overall_rmse']:.4f}")
    print(f"   OOF MAE:  {results['overall_mae']:.4f}")
    print(f"   OOF R²:   {results['overall_r2']:.4f}")

    print(f"\n2. Cross-Validation Stability:")
    print(f"   Mean RMSE: {cv_scores.mean():.4f}")
    print(f"   Std RMSE:  {cv_scores.std():.4f}")
    print(f"   Min RMSE:  {cv_scores.min():.4f}")
    print(f"   Max RMSE:  {cv_scores.max():.4f}")

    print(f"\n3. Feature Importance by Category:")
    for cat, imp in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
        pct = (imp / sum(category_importance.values())) * 100
        print(f"   {cat}: {pct:.1f}%")

    print(f"\n4. Prediction Quality:")
    print(f"   Mean Residual: {residuals.mean():.3f}")
    print(f"   Std Residual:  {residuals.std():.3f}")
    print(f"   MAE: {np.abs(residuals).mean():.3f}")

# Create post-processing insights
create_postprocessing_insights(results, df_train, oof_predictions)
```

### 5. Final Comparison & Tracking (NEW!)

**Cell terakhir:**

```python
# === CELL: FINAL RESULTS & IMPROVEMENT TRACKING ===

def create_final_summary(results, initial_features, final_features):
    """Create final summary and comparison"""

    print(f"\n{'='*80}")
    print(f"🎉 FINAL SUMMARY & IMPROVEMENT TRACKING")
    print(f"{'='*80}")

    # Metrics comparison table
    print(f"\n{'Metric':<30} {'Value':<15} {'Status'}")
    print("-" * 60)
    print(f"{'OOF RMSE':<30} {results['overall_rmse']:<15.4f} {'🏆 ' if results['overall_rmse'] < 16.0 else '✅ ' if results['overall_rmse'] < 16.3 else '💪 '}")
    print(f"{'OOF MAE':<30} {results['overall_mae']:<15.4f}")
    print(f"{'OOF R²':<30} {results['overall_r2']:<15.4f}")
    print(f"{'CV Mean RMSE':<30} {results['cv_scores'].mean():<15.4f}")
    print(f"{'CV Std RMSE':<30} {results['cv_scores'].std():<15.4f} {'✅ Stable' if results['cv_scores'].std() < 0.03 else '⚠️  Check'}")

    # Feature reduction summary
    print(f"\n{'='*80}")
    print(f"📊 FEATURE ENGINEERING SUMMARY")
    print(f"{'='*80}")
    print(f"\nInitial Features: {initial_features}")
    print(f"Final Features:   {final_features}")
    print(f"Reduction:        {initial_features - final_features} features ({(initial_features - final_features)/initial_features*100:.1f}%)")

    # Performance assessment
    print(f"\n{'='*80}")
    print(f"📈 PERFORMANCE ASSESSMENT")
    print(f"{'='*80}")

    baseline_rmse = 16.3
    current_rmse = results['overall_rmse']
    improvement = baseline_rmse - current_rmse

    print(f"\nBaseline (Siklus 4):  ~{baseline_rmse:.2f} RMSE")
    print(f"Current (Siklus 5):   {current_rmse:.4f} RMSE")
    print(f"Improvement:          {improvement:.4f} RMSE ({improvement/baseline_rmse*100:.2f}%)")

    if current_rmse < 15.0:
        status = "🏆 EXCELLENT! Top-tier performance!"
    elif current_rmse < 15.5:
        status = "🎉 VERY GOOD! Competitive result!"
    elif current_rmse < 16.0:
        status = "✅ GOOD! Solid baseline!"
    else:
        status = "💪 Room for improvement - see recommendations"

    print(f"\nStatus: {status}")

    # Save summary to file
    summary_file = Path(OUTPUT_PATH) / 'model_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL SUMMARY - SIKLUS 5\n")
        f.write("="*80 + "\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  OOF RMSE: {results['overall_rmse']:.4f}\n")
        f.write(f"  OOF MAE:  {results['overall_mae']:.4f}\n")
        f.write(f"  OOF R²:   {results['overall_r2']:.4f}\n")
        f.write(f"  CV Mean:  {results['cv_scores'].mean():.4f}\n")
        f.write(f"  CV Std:   {results['cv_scores'].std():.4f}\n\n")

        f.write("FEATURE SUMMARY:\n")
        f.write(f"  Initial: {initial_features} features\n")
        f.write(f"  Final:   {final_features} features\n")
        f.write(f"  Reduced: {initial_features - final_features} features\n\n")

        f.write("TOP 20 FEATURES:\n")
        for i, (feat, row) in enumerate(results['feature_importance'].head(20).iterrows(), 1):
            pct = (row['mean'] / results['feature_importance']['mean'].sum()) * 100
            f.write(f"  {i:2d}. {feat:<35s}: {row['mean']:8.1f} ({pct:5.2f}%)\n")

    print(f"\n✅ Summary saved: {summary_file}")

    # Recommendations
    print(f"\n{'='*80}")
    print(f"💡 NEXT STEPS RECOMMENDATIONS")
    print(f"{'='*80}")

    recommendations = []

    if results['cv_scores'].std() > 0.03:
        recommendations.append("⚠️  High CV variance - consider more regularization or data augmentation")

    if results['overall_rmse'] > 15.5:
        recommendations.append("📈 Try hyperparameter tuning (Optuna) for better performance")
        recommendations.append("🔧 Consider adding more interaction features (artist-genre, etc.)")

    if final_features < 40:
        recommendations.append("⚠️  Few features - might be underfitting, review dropped features")
    elif final_features > 60:
        recommendations.append("💡 Many features - consider more aggressive feature selection")

    recommendations.append("🎯 Try fold averaging ensemble for +0.03-0.10 RMSE improvement")
    recommendations.append("🚀 Consider multi-model ensemble (LightGBM + XGBoost + CatBoost)")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")

    print(f"\n{'='*80}")
    print(f"✨ Pipeline Complete! Check visualizations in {OUTPUT_PATH}/")
    print(f"{'='*80}\n")

# Create final summary
create_final_summary(
    results,
    initial_features=67,  # Adjust based on your actual count
    final_features=len(selected_features)
)
```

---

## 📁 Files Generated

After running enhanced pipeline:

```
outputs/
├── preprocessing_insights.png      # Pre-processing visualizations
├── feature_selection.png           # Feature selection analysis
├── postprocessing_insights.png     # Post-processing visualizations
├── model_summary.txt               # Text summary of results
└── submission_siklus5_complete.csv # Final submission
```

---

## 🎯 Expected Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Features** | 67 features | 40-50 features (optimized) |
| **Insights** | Basic EDA only | Visual insights at every stage |
| **Transparency** | Limited | Full pipeline tracking |
| **Performance** | Good | Better (less overfitting) |
| **Debugging** | Hard | Easy (visual feedback) |

---

## 📖 Usage

Replace existing cells with new structure:
- Add data profiling after load
- Add pre-processing insights before feature engineering
- Add feature selection after base features preparation
- Add post-processing insights after training
- Add final summary at end

Total cells: **~15 cells** (from original 12)

---

**Ready to implement! This creates a professional ML pipeline with full insights tracking.** 🚀
