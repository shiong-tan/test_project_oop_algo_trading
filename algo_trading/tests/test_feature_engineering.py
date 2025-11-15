"""Unit tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from algo_trading.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.DataFrame({"price": prices}, index=dates)

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(
            sma_windows=(20, 60),
            n_lags=5,
        )
        assert engineer.sma_windows == (20, 60)
        assert engineer.n_lags == 5
        assert engineer.exclude_features == ["d", "d_"]

    def test_create_returns(self, sample_data):
        """Test return creation."""
        engineer = FeatureEngineer()
        result = engineer.create_returns(sample_data, price_col="price")

        assert "r" in result.columns
        assert "direction" in result.columns
        assert result["direction"].isin([0, 1]).all()

    def test_create_technical_indicators(self, sample_data):
        """Test technical indicator creation."""
        engineer = FeatureEngineer(sma_windows=(20, 60))
        result = engineer.create_technical_indicators(sample_data, price_col="price")

        # Check all expected indicators are present
        expected_cols = ["SMA1", "SMA2", "SMA_diff", "EWMA1", "EWMA2",
                        "EWMA_diff", "V1", "V2", "RSI", "BB_width",
                        "BB_position", "momentum"]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation."""
        engineer = FeatureEngineer()
        rsi = engineer._calculate_rsi(sample_data["price"], period=14)

        # RSI should be between 0 and 100
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_add_lagged_features(self, sample_data):
        """Test lagged feature creation."""
        engineer = FeatureEngineer(n_lags=3)
        result = engineer.add_lagged_features(sample_data, ["price"])

        # Check lagged columns exist
        assert "price_lag_1" in result.columns
        assert "price_lag_2" in result.columns
        assert "price_lag_3" in result.columns

        # Check lag values are correct
        assert result["price_lag_1"].iloc[1] == sample_data["price"].iloc[0]

    def test_fit_transform(self, sample_data):
        """Test fit_transform on training data."""
        engineer = FeatureEngineer(sma_windows=(20, 60), n_lags=5)
        features, feature_cols = engineer.fit_transform(
            sample_data, price_col="price"
        )

        # Check output types
        assert isinstance(features, pd.DataFrame)
        assert isinstance(feature_cols, list)

        # Check direction column exists
        assert "direction" in features.columns

        # Check scaler was fitted
        assert engineer.scaler is not None
        assert len(engineer.feature_columns) > 0

        # Check all feature columns contain '_lag_'
        for col in feature_cols:
            assert "_lag_" in col

    def test_transform(self, sample_data):
        """Test transform on test data."""
        engineer = FeatureEngineer(sma_windows=(20, 60), n_lags=5)

        # Split data
        split_idx = int(len(sample_data) * 0.7)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]

        # Fit on training data
        train_features, feature_cols = engineer.fit_transform(
            train_data, price_col="price"
        )

        # Transform test data
        test_features = engineer.transform(test_data)

        # Check same columns exist
        for col in feature_cols:
            assert col in test_features.columns, f"Missing column in test: {col}"

        # Check direction column exists
        assert "direction" in test_features.columns

    def test_transform_before_fit_raises_error(self, sample_data):
        """Test that transform raises error if called before fit_transform."""
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="Must call fit_transform"):
            engineer.transform(sample_data)

    def test_exclude_features(self, sample_data):
        """Test that excluded features are not in feature list."""
        engineer = FeatureEngineer(exclude_features=["d", "d_"])

        # Add columns that should be excluded
        sample_data["d"] = 1
        sample_data["d_"] = 1

        features, feature_cols = engineer.fit_transform(
            sample_data, price_col="price"
        )

        # Check excluded features are not in feature columns
        for col in feature_cols:
            assert "d_lag_" not in col
            assert col != "d"

    def test_get_feature_names(self, sample_data):
        """Test get_feature_names method."""
        engineer = FeatureEngineer()
        features, feature_cols = engineer.fit_transform(
            sample_data, price_col="price"
        )

        feature_names = engineer.get_feature_names()
        assert feature_names == feature_cols
