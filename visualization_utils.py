"""
Visualization utilities untuk ML Pipeline
Provides comprehensive insights at each stage of the pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error


def data_profiling(df, name="Data"):
    """
    Comprehensive data profiling

    Args:
        df: DataFrame to profile
        name: Name for display

    Returns:
        missing_df: DataFrame with missing value analysis
    """
    print(f"\n{'='*80}")
    print(f"📊 DATA PROFILING: {name}")
    print(f"{'='*80}")

    # Basic info
    print(f"\n[1] Basic Information:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

    # Data types
    print(f"\n[2] Data Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Missing values
    print(f"\n[3] Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)

    if len(missing_df) > 0:
        print(f"  Columns with missing values: {len(missing_df)}")
        for col, row in missing_df.head(10).iterrows():
            print(f"    {col}: {row['Missing']} ({row['Percentage']:.2f}%)")
    else:
        print("  ✅ No missing values!")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n[4] Duplicates:")
    print(f"  Total: {duplicates} ({duplicates/len(df)*100:.2f}%)")

    # Numerical features
    numerical = df.select_dtypes(include=[np.number])
    print(f"\n[5] Numerical Features: {len(numerical.columns)}")

    # Categorical features
    categorical = df.select_dtypes(include=['object', 'category'])
    print(f"\n[6] Categorical Features: {len(categorical.columns)}")
    for col in categorical.columns[:5]:
        print(f"  {col}: {df[col].nunique()} unique values")
    if len(categorical.columns) > 5:
        print(f"  ... and {len(categorical.columns) - 5} more")

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

    if issues:
        for issue in issues[:5]:
            print(issue)
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("  ✅ No critical issues detected!")

    return missing_df


def create_preprocessing_insights(df, target_col='popularity', output_path='./outputs'):
    """
    Create comprehensive pre-processing visualizations

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        output_path: Path to save visualizations
    """
    print(f"\n{'='*80}")
    print(f"📈 PRE-PROCESSING INSIGHTS")
    print(f"{'='*80}")

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Setup figure
    fig = plt.figure(figsize=(20, 15))

    # 1. Target Distribution
    print("\n[1/9] Target distribution...")
    ax1 = plt.subplot(3, 3, 1)
    df[target_col].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
    ax1.axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df[target_col].mean():.1f}')
    ax1.axvline(df[target_col].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df[target_col].median():.1f}')
    ax1.set_xlabel(target_col.capitalize())
    ax1.set_ylabel('Frequency')
    ax1.set_title('Target Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Correlation Heatmap
    print("[2/9] Correlation heatmap...")
    ax2 = plt.subplot(3, 3, 2)

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [f for f in numerical_features if f not in ['track_id', 'Unnamed: 0']]

    if target_col in numerical_features:
        correlations = df[numerical_features].corr()[target_col].abs().sort_values(ascending=False)
        top_features = correlations.head(11).index.tolist()

        corr_matrix = df[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
        ax2.set_title('Top 10 Features - Correlation', fontsize=12, fontweight='bold')

    # 3. Audio Features Distribution
    print("[3/9] Audio features...")
    ax3 = plt.subplot(3, 3, 3)

    audio_features = ['energy', 'danceability', 'valence', 'acousticness']
    audio_features = [f for f in audio_features if f in df.columns]

    for feat in audio_features[:4]:
        ax3.hist(df[feat], bins=30, alpha=0.5, label=feat)
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Audio Features Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Genre Analysis
    print("[4/9] Genre analysis...")
    ax4 = plt.subplot(3, 3, 4)

    if 'track_genre' in df.columns and target_col in df.columns:
        genre_pop = df.groupby('track_genre')[target_col].mean().sort_values(ascending=False).head(15)
        genre_pop.plot(kind='barh', ax=ax4, color='steelblue', alpha=0.7)
        ax4.set_xlabel('Average Popularity')
        ax4.set_title('Top 15 Genres', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

    # 5. Temporal Trend
    print("[5/9] Temporal trend...")
    ax5 = plt.subplot(3, 3, 5)

    if 'release_year' in df.columns and target_col in df.columns:
        year_pop = df.groupby('release_year')[target_col].mean()
        ax5.plot(year_pop.index, year_pop.values, linewidth=2, color='darkgreen')
        ax5.set_xlabel('Release Year')
        ax5.set_ylabel('Average Popularity')
        ax5.set_title('Popularity Over Time', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # 6. Feature Correlation Bars
    print("[6/9] Correlation bars...")
    ax6 = plt.subplot(3, 3, 6)

    if target_col in numerical_features:
        top_corr = correlations.head(16)[1:]  # Skip target
        top_corr.plot(kind='barh', ax=ax6, color='coral', alpha=0.7)
        ax6.set_xlabel('Absolute Correlation')
        ax6.set_title('Top 15 Correlations with Target', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

    # 7. Missing Values
    print("[7/9] Missing values...")
    ax7 = plt.subplot(3, 3, 7)

    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0].sort_values(ascending=False).head(10)
        missing_pct = (missing_cols / len(df)) * 100
        missing_pct.plot(kind='barh', ax=ax7, color='red', alpha=0.7)
        ax7.set_xlabel('Missing %')
        ax7.set_title('Missing Values', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
    else:
        ax7.text(0.5, 0.5, '✅ No Missing Values',
                ha='center', va='center', fontsize=14, transform=ax7.transAxes)
        ax7.axis('off')

    # 8. Outliers (Box plot)
    print("[8/9] Outlier detection...")
    ax8 = plt.subplot(3, 3, 8)

    key_features = ['energy', 'danceability', 'loudness', target_col]
    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        df[key_features].boxplot(ax=ax8)
        ax8.set_ylabel('Value')
        ax8.set_title('Outlier Detection', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

    # 9. Scatter: Target vs Top Feature
    print("[9/9] Scatter plot...")
    ax9 = plt.subplot(3, 3, 9)

    if target_col in numerical_features and len(top_corr) > 0:
        top_feature = top_corr.index[0]
        ax9.scatter(df[top_feature], df[target_col], alpha=0.3, s=10)
        ax9.set_xlabel(top_feature)
        ax9.set_ylabel(target_col.capitalize())
        ax9.set_title(f'{target_col} vs {top_feature[:20]}...', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_path) / 'preprocessing_insights.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")
    plt.show()

    # Summary
    print(f"\n{'='*80}")
    print("📊 KEY INSIGHTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nTarget Statistics:")
    print(f"  Mean:     {df[target_col].mean():.2f}")
    print(f"  Median:   {df[target_col].median():.2f}")
    print(f"  Std:      {df[target_col].std():.2f}")
    print(f"  Skewness: {df[target_col].skew():.2f}")

    if target_col in numerical_features and len(top_corr) > 0:
        print(f"\nTop 5 Correlated Features:")
        for i, (feat, corr) in enumerate(top_corr.head(5).items(), 1):
            print(f"  {i}. {feat}: {corr:.4f}")

    return correlations if target_col in numerical_features else None


def create_postprocessing_insights(results, df_train, oof_predictions, output_path='./outputs'):
    """
    Create comprehensive post-processing visualizations

    Args:
        results: Dictionary with CV results
        df_train: Training dataframe
        oof_predictions: Out-of-fold predictions
        output_path: Path to save visualizations
    """
    print(f"\n{'='*80}")
    print(f"📈 POST-PROCESSING INSIGHTS")
    print(f"{'='*80}")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))

    # 1. Feature Importance
    print("\n[1/9] Feature importance...")
    ax1 = plt.subplot(3, 3, 1)
    top_20 = results['feature_importance'].head(20)
    ax1.barh(range(len(top_20)), top_20['mean'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20.index, fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 20 Features', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. CV Scores
    print("[2/9] CV scores...")
    ax2 = plt.subplot(3, 3, 2)
    cv_scores = results['cv_scores']
    x_pos = range(1, len(cv_scores) + 1)
    ax2.bar(x_pos, cv_scores, color='coral', alpha=0.7, edgecolor='black')
    ax2.axhline(cv_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {cv_scores.mean():.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('RMSE')
    ax2.set_title('CV Scores by Fold', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Predictions vs Actual
    print("[3/9] Predictions vs actual...")
    ax3 = plt.subplot(3, 3, 3)
    y_true = df_train['popularity'].values
    ax3.scatter(y_true, oof_predictions, alpha=0.3, s=10)
    min_val = min(y_true.min(), oof_predictions.min())
    max_val = max(y_true.max(), oof_predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')
    ax3.set_title(f'Predictions vs Actual\nR² = {results["overall_r2"]:.4f}',
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Residuals
    print("[4/9] Residual plot...")
    ax4 = plt.subplot(3, 3, 4)
    residuals = y_true - oof_predictions
    ax4.scatter(oof_predictions, residuals, alpha=0.3, s=10)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Residual Distribution
    print("[5/9] Residual distribution...")
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Residual Distribution\nMean: {residuals.mean():.3f}',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Feature Category Importance
    print("[6/9] Category importance...")
    ax6 = plt.subplot(3, 3, 6)

    fi = results['feature_importance']
    categories = {
        'Target Encoded': [],
        'Audio': [],
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
        elif any(x in feat for x in ['track', 'has_', 'is_', 'title']):
            categories['Track Name'].append(fi.loc[feat, 'mean'])
        elif any(x in feat for x in ['energy', 'dance', 'valence', 'acoustic', 'ratio']):
            categories['Audio'].append(fi.loc[feat, 'mean'])
        else:
            categories['Other'].append(fi.loc[feat, 'mean'])

    category_imp = {k: sum(v) for k, v in categories.items() if len(v) > 0}

    if category_imp:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        ax6.pie(category_imp.values(), labels=category_imp.keys(),
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax6.set_title('Feature Importance by Category', fontsize=12, fontweight='bold')

    # 7. CV Stability
    print("[7/9] CV stability...")
    ax7 = plt.subplot(3, 3, 7)
    ax7.boxplot([cv_scores], labels=['CV Scores'])
    ax7.set_ylabel('RMSE')
    ax7.set_title(f'CV Stability\nStd: {cv_scores.std():.4f}',
                  fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Error by Range
    print("[8/9] Error by range...")
    ax8 = plt.subplot(3, 3, 8)

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
    ax8.set_ylabel('MAE')
    ax8.set_title('Error by Range', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Top 10 Importance %
    print("[9/9] Top features %...")
    ax9 = plt.subplot(3, 3, 9)

    top_10 = results['feature_importance'].head(10)
    total_imp = results['feature_importance']['mean'].sum()
    top_10_pct = (top_10['mean'] / total_imp) * 100

    ax9.barh(range(len(top_10)), top_10_pct.values, color='#2ECC71', alpha=0.7)
    ax9.set_yticks(range(len(top_10)))
    ax9.set_yticklabels(top_10.index, fontsize=9)
    ax9.set_xlabel('Importance %')
    ax9.set_title(f'Top 10 Features\nTotal: {top_10_pct.sum():.1f}%',
                  fontsize=12, fontweight='bold')
    ax9.invert_yaxis()
    ax9.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file = Path(output_path) / 'postprocessing_insights.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")
    plt.show()

    # Summary
    print(f"\n{'='*80}")
    print("📊 POST-PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"\nModel Performance:")
    print(f"  OOF RMSE: {results['overall_rmse']:.4f}")
    print(f"  OOF MAE:  {results['overall_mae']:.4f}")
    print(f"  OOF R²:   {results['overall_r2']:.4f}")

    print(f"\nCV Stability:")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std:  {cv_scores.std():.4f}")

    if category_imp:
        print(f"\nFeature Category Importance:")
        for cat, imp in sorted(category_imp.items(), key=lambda x: x[1], reverse=True):
            pct = (imp / sum(category_imp.values())) * 100
            print(f"  {cat}: {pct:.1f}%")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - data_profiling(df, name)")
    print("  - create_preprocessing_insights(df, target_col, output_path)")
    print("  - create_postprocessing_insights(results, df_train, oof_predictions, output_path)")
