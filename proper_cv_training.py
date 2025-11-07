"""
Proper Cross-Validation Training Module
========================================

FIX DATA LEAKAGE dengan:
1. Target encoding dilakukan DALAM CV loop
2. Early stopping
3. Improved hyperparameters dengan regularization

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from enhanced_features import create_fold_features


def train_model_with_proper_cv(df_train, features, target_col='popularity', cv_folds=5):
    \"\"\"
    Train model dengan proper CV (NO DATA LEAKAGE)

    Key improvements:
    1. Target encoding dilakukan per-fold
    2. Early stopping
    3. Better hyperparameters

    Args:
        df_train: Training DataFrame
        features: List of feature names (excluding target-encoded features)
        target_col: Target column name
        cv_folds: Number of CV folds

    Returns:
        results dictionary with model, predictions, scores
    \"\"\"
    print(\"\\n\" + \"=\"*80)
    print(\"🤖 TRAINING MODEL WITH PROPER CV (NO LEAKAGE)\")
    print(\"=\"*80)

    # Setup
    X = df_train[features].copy()
    y = df_train[target_col].copy()

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Storage
    oof_predictions = np.zeros(len(X))
    cv_scores = []
    feature_importance_list = []
    models = []

    # Improved hyperparameters
    lgbm_params = {
        'n_estimators': 2000,          # Increase karena ada early stopping
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,               # Let num_leaves control
        'min_child_samples': 20,       # Min data in leaf
        'subsample': 0.8,              # Row sampling (bagging)
        'subsample_freq': 1,
        'colsample_bytree': 0.8,       # Column sampling
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 0.1,             # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    print(f\"\\nModel Configuration:\")
    print(f\"  CV Folds: {cv_folds}\")
    print(f\"  Max estimators: {lgbm_params['n_estimators']}\")
    print(f\"  Learning rate: {lgbm_params['learning_rate']}\")
    print(f\"  Regularization: L1={lgbm_params['reg_alpha']}, L2={lgbm_params['reg_lambda']}\")
    print(f\"  Sampling: row={lgbm_params['subsample']}, col={lgbm_params['colsample_bytree']}\")

    print(f\"\\n\" + \"-\"*80)
    print(\"Starting Cross-Validation...\")
    print(\"-\"*80)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f\"\\n[Fold {fold}/{cv_folds}]\")

        # =====================================================================
        # IMPORTANT: Create features PER FOLD to avoid leakage
        # =====================================================================

        # Get fold data
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()

        # Create target-encoded features for THIS FOLD ONLY
        # Training fold: use only training data
        df_train_fold = create_fold_features(df_train_fold, df_train_fold, target_col)

        # Validation fold: use training fold stats (not validation!)
        df_val_fold = create_fold_features(df_val_fold, df_train_fold, target_col)

        # Get feature names (now includes target-encoded features)
        all_features = [f for f in df_train_fold.columns
                       if f not in [target_col, 'track_id', 'track_name', 'artists',
                                   'lyrics', 'release_year', 'track_genre']]

        # Prepare X, y for this fold
        X_train_fold = df_train_fold[all_features]
        y_train_fold = df_train_fold[target_col]
        X_val_fold = df_val_fold[all_features]
        y_val_fold = df_val_fold[target_col]

        print(f\"  Train: {len(X_train_fold)} samples\")
        print(f\"  Val:   {len(X_val_fold)} samples\")
        print(f\"  Features: {len(all_features)}\")

        # Train model
        model = LGBMRegressor(**lgbm_params)

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse',
            early_stopping_rounds=100,    # Stop jika tidak improve 100 rounds
            verbose=False
        )

        # Predict on validation
        y_pred = model.predict(X_val_fold)
        oof_predictions[val_idx] = y_pred

        # Calculate fold score
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        fold_mae = mean_absolute_error(y_val_fold, y_pred)
        fold_r2 = r2_score(y_val_fold, y_pred)

        cv_scores.append(fold_rmse)

        print(f\"  Best iteration: {model.best_iteration_}\")
        print(f\"  RMSE: {fold_rmse:.4f}\")
        print(f\"  MAE:  {fold_mae:.4f}\")
        print(f\"  R²:   {fold_r2:.4f}\")

        # Store feature importance
        fi_df = pd.DataFrame({
            'feature': all_features,
            'importance': model.feature_importances_,
            'fold': fold
        })
        feature_importance_list.append(fi_df)

        # Store model
        models.append(model)

    # =========================================================================
    # OVERALL RESULTS
    # =========================================================================

    cv_scores = np.array(cv_scores)
    overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    overall_mae = mean_absolute_error(y, oof_predictions)
    overall_r2 = r2_score(y, oof_predictions)

    print(f\"\\n\" + \"=\"*80)
    print(\"CROSS-VALIDATION RESULTS\")
    print(\"=\"*80)

    print(f\"\\nFold Scores:\")
    for i, score in enumerate(cv_scores, 1):
        print(f\"  Fold {i}: {score:.4f}\")

    print(f\"\\nCV Statistics:\")
    print(f\"  Mean RMSE: {cv_scores.mean():.4f}\")
    print(f\"  Std RMSE:  {cv_scores.std():.4f}\")
    print(f\"  Min RMSE:  {cv_scores.min():.4f}\")
    print(f\"  Max RMSE:  {cv_scores.max():.4f}\")

    print(f\"\\nOut-of-Fold Performance:\")
    print(f\"  OOF RMSE: {overall_rmse:.4f}\")
    print(f\"  OOF MAE:  {overall_mae:.4f}\")
    print(f\"  OOF R²:   {overall_r2:.4f}\")

    # Aggregate feature importance
    fi_all = pd.concat(feature_importance_list)
    fi_summary = fi_all.groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean', ascending=False)

    print(f\"\\nTop 10 Most Important Features:\")
    for idx, (feat, row) in enumerate(fi_summary.head(10).iterrows(), 1):
        print(f\"  {idx:2d}. {feat:30s}: {row['mean']:8.1f} (±{row['std']:.1f})\")

    # Return results
    results = {
        'models': models,
        'oof_predictions': oof_predictions,
        'cv_scores': cv_scores,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'feature_importance': fi_summary,
        'feature_importance_all_folds': fi_all,
        'all_features': all_features  # Important: return feature list used
    }

    return results


def train_final_model(df_train, features, target_col='popularity'):
    \"\"\"
    Train final model on FULL training data for test predictions

    Args:
        df_train: Full training DataFrame
        features: List of features to use
        target_col: Target column

    Returns:
        trained model
    \"\"\"
    print(f\"\\n\" + \"=\"*80)
    print(\"TRAINING FINAL MODEL (Full Training Data)\")
    print(\"=\"*80)

    # Create features for full training data
    df_train_with_features = create_fold_features(df_train, df_train, target_col)

    # Get all features
    all_features = [f for f in df_train_with_features.columns
                   if f not in [target_col, 'track_id', 'track_name', 'artists',
                               'lyrics', 'release_year', 'track_genre']]

    X = df_train_with_features[all_features]
    y = df_train_with_features[target_col]

    print(f\"\\n  Training samples: {len(X)}\")
    print(f\"  Features: {len(all_features)}\")

    # Same hyperparameters as CV
    lgbm_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = LGBMRegressor(**lgbm_params)

    # For final model, we don't have validation set
    # Use 80% of iterations from CV average
    # Or just fit without early stopping
    model.fit(X, y)

    print(f\"\\n✅ Final model trained!\")

    return model, all_features


def predict_test_set(model, df_test, features):
    \"\"\"
    Make predictions on test set

    Args:
        model: Trained model
        df_test: Test DataFrame (should already have features from create_fold_features)
        features: List of feature names

    Returns:
        predictions array
    \"\"\"
    print(f\"\\n\" + \"=\"*80)
    print(\"PREDICTING TEST SET\")
    print(\"=\"*80)

    X_test = df_test[features]

    print(f\"\\n  Test samples: {len(X_test)}\")
    print(f\"  Features: {len(features)}\")

    predictions = model.predict(X_test)

    # Clip to valid range
    predictions = np.clip(predictions, 0, 100)

    print(f\"\\n  Prediction statistics:\")
    print(f\"    Mean:   {predictions.mean():.2f}\")
    print(f\"    Median: {np.median(predictions):.2f}\")
    print(f\"    Std:    {predictions.std():.2f}\")
    print(f\"    Range:  [{predictions.min():.2f}, {predictions.max():.2f}]\")

    return predictions
