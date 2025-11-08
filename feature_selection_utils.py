"""
Feature Selection utilities untuk ML Pipeline
Automatic feature selection based on importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

# Check LightGBM version
try:
    from lightgbm import early_stopping, log_evaluation
    LIGHTGBM_NEW_API = True
except ImportError:
    LIGHTGBM_NEW_API = False


def automatic_feature_selection(df_train, base_features, target_col='popularity',
                                target_num_features=50, importance_threshold=0.5,
                                create_fold_features_func=None, output_path='./outputs'):
    """
    Automatically select top features based on importance

    Args:
        df_train: Training dataframe
        base_features: List of all base features (before target encoding)
        target_col: Target column name
        target_num_features: Target number of features to keep
        importance_threshold: Minimum importance % to keep feature
        create_fold_features_func: Function to create fold features
        output_path: Path to save visualizations

    Returns:
        selected_features: List of selected features (base features only)
        feature_importance_df: DataFrame with importance scores
        all_features_used: List of all features used in training (including target-encoded)
    """

    print(f"\n{'='*80}")
    print(f"🎯 AUTOMATIC FEATURE SELECTION")
    print(f"{'='*80}")

    print(f"\n📊 Initial State:")
    print(f"  Total base features: {len(base_features)}")
    print(f"  Target: Keep ~{target_num_features} features")
    print(f"  Threshold: Features with >{importance_threshold}% importance")

    # Validate function
    if create_fold_features_func is None:
        raise ValueError("create_fold_features_func is required!")

    # Quick 3-fold CV for feature selection
    print(f"\n[1/3] Training quick model for feature importance...")
    print(f"  Using 3-fold CV for speed...")

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
        df_train_fold = create_fold_features_func(df_train_fold, df_train_fold, target_col)
        df_val_fold = create_fold_features_func(df_val_fold, df_train_fold, target_col)

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

        print(f"    Fold {fold}/3: {len(all_features)} features")

    # Aggregate importance
    print(f"\n[2/3] Analyzing feature importance...")
    fi_all = pd.concat(feature_importance_list)
    fi_summary = fi_all.groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean', ascending=False)

    # Calculate importance percentage
    fi_summary['importance_pct'] = (fi_summary['mean'] / fi_summary['mean'].sum()) * 100
    fi_summary['cumulative_pct'] = fi_summary['importance_pct'].cumsum()

    # Identify base vs target-encoded features
    target_encoded_keywords = ['artist', 'genre']
    fi_summary['is_base_feature'] = fi_summary.index.map(
        lambda x: not any(kw in x for kw in target_encoded_keywords)
    )

    # Select features using multiple methods
    print(f"\n[3/3] Selecting features...")

    # Method 1: Top N features (among base features only)
    base_features_fi = fi_summary[fi_summary['is_base_feature']]
    method1_features = base_features_fi.head(target_num_features).index.tolist()

    # Method 2: Importance threshold (among base features)
    method2_features = base_features_fi[base_features_fi['importance_pct'] > importance_threshold].index.tolist()

    # Method 3: Cumulative importance (90% coverage)
    cum_90 = fi_summary[fi_summary['cumulative_pct'] <= 90.0]
    method3_features = cum_90[cum_90['is_base_feature']].index.tolist()

    print(f"\n📊 Selection Methods:")
    print(f"  Method 1 (Top {target_num_features}):       {len(method1_features)} features")
    print(f"  Method 2 (Importance >{importance_threshold}%): {len(method2_features)} features")
    print(f"  Method 3 (90% cumulative):  {len(method3_features)} features")

    # Use most conservative approach
    if len(method2_features) >= 30 and len(method2_features) <= target_num_features:
        selected_base_features = method2_features
        method_used = f"Method 2 (Importance >{importance_threshold}%)"
    elif len(method3_features) >= 30 and len(method3_features) <= target_num_features + 10:
        selected_base_features = method3_features
        method_used = "Method 3 (90% cumulative)"
    else:
        selected_base_features = method1_features
        method_used = f"Method 1 (Top {target_num_features})"

    print(f"\n✅ Selected Method: {method_used}")
    print(f"   Base features: {len(selected_base_features)}")

    # Show dropped features
    dropped_features = [f for f in base_features if f not in selected_base_features]
    print(f"\n❌ Dropped {len(dropped_features)} base features:")
    for feat in dropped_features[:10]:
        if feat in base_features_fi.index:
            imp_pct = base_features_fi.loc[feat, 'importance_pct']
            print(f"  - {feat:<35s}: {imp_pct:.3f}%")
    if len(dropped_features) > 10:
        print(f"  ... and {len(dropped_features) - 10} more")

    # Show kept features
    print(f"\n✅ Kept {len(selected_base_features)} base features:")
    for feat in sorted(selected_base_features, key=lambda x: base_features_fi.loc[x, 'mean'], reverse=True)[:10]:
        imp_pct = base_features_fi.loc[feat, 'importance_pct']
        print(f"  + {feat:<35s}: {imp_pct:.3f}%")
    if len(selected_base_features) > 10:
        print(f"  ... and {len(selected_base_features) - 10} more")

    # Visualization
    print(f"\n📊 Creating visualizations...")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Top features (all features including target-encoded)
    top_20 = fi_summary.head(20)
    colors = ['steelblue' if is_base else 'coral' for is_base in top_20['is_base_feature']]
    ax1.barh(range(len(top_20)), top_20['mean'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20.index, fontsize=9)
    ax1.set_xlabel('Importance', fontsize=11)
    ax1.set_title('Top 20 Features by Importance\n(Blue=Base, Orange=Target-Encoded)',
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Cumulative importance (base features only)
    ax2.plot(range(1, len(base_features_fi)+1), base_features_fi['cumulative_pct'].values,
             linewidth=2, color='darkblue', label='Cumulative Importance')
    ax2.axhline(80, color='red', linestyle='--', alpha=0.7, label='80%')
    ax2.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax2.axvline(len(selected_base_features), color='green', linestyle='--',
                linewidth=2, label=f'Selected: {len(selected_base_features)}')
    ax2.set_xlabel('Number of Base Features', fontsize=11)
    ax2.set_ylabel('Cumulative Importance %', fontsize=11)
    ax2.set_title('Cumulative Importance (Base Features)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_path) / 'feature_selection.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.show()

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"📈 FEATURE SELECTION SUMMARY")
    print(f"{'='*80}")

    total_imp_selected = base_features_fi.loc[selected_base_features, 'mean'].sum()
    total_imp_all = base_features_fi['mean'].sum()

    print(f"\nBase Features:")
    print(f"  Before: {len(base_features)}")
    print(f"  After:  {len(selected_base_features)}")
    print(f"  Dropped: {len(dropped_features)} ({len(dropped_features)/len(base_features)*100:.1f}%)")

    print(f"\nImportance Coverage:")
    print(f"  Selected features cover: {total_imp_selected/total_imp_all*100:.1f}% of total importance")

    print(f"\nTop 5 Selected Features:")
    top_5_selected = base_features_fi.loc[selected_base_features].head(5)
    for i, (feat, row) in enumerate(top_5_selected.iterrows(), 1):
        print(f"  {i}. {feat:<35s}: {row['importance_pct']:6.2f}%")

    # Return base features (target-encoded will be added in CV loop)
    return selected_base_features, fi_summary, all_features


def print_feature_selection_impact(before_count, after_count):
    """Print impact of feature selection"""

    print(f"\n{'='*80}")
    print(f"✅ FEATURE SELECTION COMPLETE")
    print(f"{'='*80}")

    reduction = before_count - after_count
    reduction_pct = (reduction / before_count) * 100

    print(f"\n📊 Impact:")
    print(f"  Before: {before_count} base features")
    print(f"  After:  {after_count} base features")
    print(f"  Reduction: {reduction} features ({reduction_pct:.1f}%)")

    print(f"\n💡 Benefits:")
    print(f"  ✅ Reduced overfitting risk")
    print(f"  ✅ Faster training time")
    print(f"  ✅ Better model interpretability")
    print(f"  ✅ Removed noise features")

    if reduction_pct > 30:
        print(f"\n  ⚡ Significant reduction - expect improved generalization!")
    elif reduction_pct > 15:
        print(f"\n  📈 Moderate reduction - good balance!")
    else:
        print(f"\n  💡 Small reduction - most features were important!")


if __name__ == "__main__":
    print("Feature selection utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - automatic_feature_selection(...)")
    print("  - print_feature_selection_impact(before, after)")
