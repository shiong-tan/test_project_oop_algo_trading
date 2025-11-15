"""Model training and evaluation module with comprehensive metrics."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models with proper validation."""

    def __init__(self, model: BaseEstimator, model_name: str = "model"):
        """Initialize ModelTrainer.

        Args:
            model: Scikit-learn compatible model
            model_name: Name for saving/logging

        Raises:
            TypeError: If model is not a valid estimator
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError(f"Model must be sklearn estimator, got {type(model)}")

        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise ValueError("Model must implement fit() and predict() methods")

        self.model = model
        self.model_name = model_name
        self.is_fitted = False
        self.feature_importance = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weight: Optional[str] = "balanced",
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            class_weight: Strategy for handling class imbalance
        """
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train must have same length: "
                f"{len(X_train)} != {len(y_train)}"
            )

        if len(X_train) == 0:
            raise ValueError("Cannot train on empty dataset")

        logger.info(
            f"Training {self.model_name} on {len(X_train)} samples, "
            f"{X_train.shape[1]} features"
        )

        # Handle class imbalance if model supports it
        if class_weight and hasattr(self.model, "class_weight"):
            self.model.set_params(class_weight=class_weight)

        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Extract feature importance if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            logger.info("Feature importance extracted")

        logger.info(f"{self.model_name} training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Prediction probabilities

        Raises:
            ValueError: If model not fitted or doesn't support predict_proba
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"{self.model_name} does not support predict_proba()")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test",
    ) -> Dict[str, Any]:
        """Evaluate model with comprehensive metrics.

        Args:
            X: Features
            y: True labels
            set_name: Name of dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating {self.model_name} on {set_name} set")

        # Get predictions
        y_pred = self.predict(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }

        # Add ROC AUC if predict_proba available
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.predict_proba(X)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

        # Log metrics
        logger.info(f"{set_name.upper()} METRICS:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics["roc_auc"]:
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
        logger.info(f"    FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Perform time-series cross-validation.

        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {n_splits}-fold time-series cross-validation")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]

            # Train on this fold
            self.model.fit(X_train_cv, y_train_cv)

            # Evaluate on validation set
            y_pred_cv = self.model.predict(X_val_cv)
            fold_score = accuracy_score(y_val_cv, y_pred_cv)
            scores.append(fold_score)

            logger.info(f"  Fold {fold}: {fold_score:.4f}")

        cv_results = {
            "fold_scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
        }

        logger.info(
            f"CV Results: {cv_results['mean_score']:.4f} "
            f"(+/- {cv_results['std_score']:.4f})"
        )

        # Reset to untrained state
        self.is_fitted = False

        return cv_results

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        method: str = "grid",
        n_iter: int = 20,
        cv: int = 5,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Tune hyperparameters using GridSearch or RandomizedSearch.

        Args:
            X: Features
            y: Target
            param_grid: Parameter grid for search
            method: 'grid' or 'random'
            n_iter: Number of iterations for RandomizedSearch
            cv: Number of CV folds

        Returns:
            Tuple of (best_params, cv_results)
        """
        logger.info(f"Tuning hyperparameters using {method} search")

        tscv = TimeSeriesSplit(n_splits=cv)

        if method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
        elif method == "random":
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )
        else:
            raise ValueError(f"Invalid method: {method}. Use 'grid' or 'random'")

        # Perform search
        search.fit(X, y)

        best_params = search.best_params_
        best_score = search.best_score_

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")

        # Update model with best parameters
        self.model.set_params(**best_params)

        results = {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": search.cv_results_,
        }

        return best_params, results

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk.

        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_importance": self.feature_importance,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk.

        Args:
            filepath: Path to load model from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.model_name = model_data["model_name"]
        self.feature_importance = model_data.get("feature_importance")
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance.

        Args:
            top_n: Return only top N features (None for all)

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available for this model")

        if top_n:
            return self.feature_importance.head(top_n)

        return self.feature_importance
