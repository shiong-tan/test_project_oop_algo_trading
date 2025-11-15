"""Unit tests for model trainer module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from algo_trading.models.trainer import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.randint(0, 2, 100), name="target")
        return X, y

    def test_initialization(self):
        """Test ModelTrainer initialization."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        trainer = ModelTrainer(model, model_name="TestModel")

        assert trainer.model == model
        assert trainer.model_name == "TestModel"
        assert trainer.is_fitted is False
        assert trainer.feature_importance is None

    def test_initialization_invalid_model(self):
        """Test initialization with invalid model."""
        with pytest.raises(TypeError, match="Model must be sklearn estimator"):
            ModelTrainer("not_a_model")

    def test_train(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        trainer = ModelTrainer(model)

        trainer.train(X, y)

        assert trainer.is_fitted is True
        assert trainer.feature_importance is not None
        assert len(trainer.feature_importance) == 5

    def test_train_empty_dataset(self):
        """Test training with empty dataset raises error."""
        X = pd.DataFrame()
        y = pd.Series()
        trainer = ModelTrainer(RandomForestClassifier())

        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            trainer.train(X, y)

    def test_train_length_mismatch(self):
        """Test training with mismatched X and y lengths."""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 50))
        trainer = ModelTrainer(RandomForestClassifier())

        with pytest.raises(ValueError, match="must have same length"):
            trainer.train(X, y)

    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        trainer = ModelTrainer(RandomForestClassifier(n_estimators=10, random_state=42))
        trainer.train(X, y)

        predictions = trainer.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_before_training(self, sample_data):
        """Test prediction before training raises error."""
        X, y = sample_data
        trainer = ModelTrainer(RandomForestClassifier())

        with pytest.raises(ValueError, match="Model must be trained"):
            trainer.predict(X)

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        trainer = ModelTrainer(RandomForestClassifier(n_estimators=10, random_state=42))
        trainer.train(X, y)

        probabilities = trainer.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_not_supported(self, sample_data):
        """Test predict_proba with model that doesn't support it."""
        X, y = sample_data

        # Create a mock model without predict_proba
        class MockModel:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return np.zeros(len(X))

        from sklearn.base import BaseEstimator
        class SklearnMockModel(BaseEstimator, MockModel):
            pass

        trainer = ModelTrainer(SklearnMockModel())
        trainer.is_fitted = True

        with pytest.raises(ValueError, match="does not support predict_proba"):
            trainer.predict_proba(X)

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        trainer = ModelTrainer(RandomForestClassifier(n_estimators=10, random_state=42))
        trainer.train(X, y)

        metrics = trainer.evaluate(X, y, set_name="test")

        # Check all expected metrics are present
        expected_metrics = ["accuracy", "precision", "recall", "f1",
                           "roc_auc", "confusion_matrix",
                           "true_negatives", "false_positives",
                           "false_negatives", "true_positives"]

        for metric in expected_metrics:
            assert metric in metrics

        # Check metric values are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        trainer = ModelTrainer(RandomForestClassifier(n_estimators=10, random_state=42))

        cv_results = trainer.cross_validate(X, y, n_splits=3)

        assert "fold_scores" in cv_results
        assert "mean_score" in cv_results
        assert "std_score" in cv_results
        assert len(cv_results["fold_scores"]) == 3

        # After CV, model should not be fitted
        assert trainer.is_fitted is False

    def test_get_feature_importance(self, sample_data):
        """Test feature importance retrieval."""
        X, y = sample_data
        trainer = ModelTrainer(RandomForestClassifier(n_estimators=10, random_state=42))
        trainer.train(X, y)

        importance = trainer.get_feature_importance(top_n=3)

        assert len(importance) == 3
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_get_feature_importance_not_available(self):
        """Test feature importance when not available."""
        trainer = ModelTrainer(LogisticRegression())
        trainer.is_fitted = True

        with pytest.raises(ValueError, match="Feature importance not available"):
            trainer.get_feature_importance()
