"""Feature engineering module with proper handling to prevent data leakage."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates technical features for trading strategy.

    CRITICAL: This class is designed to prevent data leakage by:
    1. Separating feature calculation for train and test sets
    2. Fitting scaler ONLY on training data
    3. Excluding features that leak future information ('d', 'd_')
    """

    def __init__(
        self,
        sma_windows: Tuple[int, int] = (20, 60),
        ewma_halflife: Tuple[int, int] = (20, 60),
        volatility_windows: Tuple[int, int] = (20, 60),
        rsi_period: int = 14,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        n_lags: int = 5,
        exclude_features: Optional[List[str]] = None,
    ):
        """Initialize FeatureEngineer.

        Args:
            sma_windows: Short and long windows for SMA
            ewma_halflife: Short and long halflife for EWMA
            volatility_windows: Short and long windows for volatility
            rsi_period: Period for RSI calculation
            bollinger_period: Period for Bollinger Bands
            bollinger_std: Number of standard deviations for Bollinger Bands
            n_lags: Number of lagged features
            exclude_features: Features to exclude (default: ['d', 'd_'])
        """
        self.sma_windows = sma_windows
        self.ewma_halflife = ewma_halflife
        self.volatility_windows = volatility_windows
        self.rsi_period = rsi_period
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.n_lags = n_lags
        self.exclude_features = exclude_features or ["d", "d_"]

        self.scaler = StandardScaler()
        self.feature_columns = []
        self.price_column = None

    def create_returns(self, data: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Create return features.

        Args:
            data: Input DataFrame
            price_col: Name of price column

        Returns:
            DataFrame with return features
        """
        df = data.copy()

        # Log returns
        df["r"] = np.log(df[price_col] / df[price_col].shift(1))

        # Direction (target variable - do NOT use as feature!)
        df["direction"] = np.where(df["r"] > 0, 1, 0)

        logger.debug(f"Created return features. Shape: {df.shape}")

        return df

    def create_technical_indicators(
        self, data: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Create technical indicator features.

        Args:
            data: Input DataFrame
            price_col: Name of price column

        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()

        # Simple Moving Averages
        df["SMA1"] = df[price_col].rolling(window=self.sma_windows[0]).mean()
        df["SMA2"] = df[price_col].rolling(window=self.sma_windows[1]).mean()
        df["SMA_diff"] = df["SMA1"] - df["SMA2"]

        # Exponential Weighted Moving Averages
        df["EWMA1"] = df[price_col].ewm(halflife=self.ewma_halflife[0]).mean()
        df["EWMA2"] = df[price_col].ewm(halflife=self.ewma_halflife[1]).mean()
        df["EWMA_diff"] = df["EWMA1"] - df["EWMA2"]

        # Volatility (rolling standard deviation of returns)
        if "r" in df.columns:
            df["V1"] = df["r"].rolling(window=self.volatility_windows[0]).std()
            df["V2"] = df["r"].rolling(window=self.volatility_windows[1]).std()
        else:
            logger.warning("Returns not found, creating them first")
            df = self.create_returns(df, price_col)
            df["V1"] = df["r"].rolling(window=self.volatility_windows[0]).std()
            df["V2"] = df["r"].rolling(window=self.volatility_windows[1]).std()

        # RSI (Relative Strength Index)
        df["RSI"] = self._calculate_rsi(df[price_col], self.rsi_period)

        # Bollinger Bands
        df["BB_middle"] = df[price_col].rolling(window=self.bollinger_period).mean()
        bb_std = df[price_col].rolling(window=self.bollinger_period).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * self.bollinger_std)
        df["BB_lower"] = df["BB_middle"] - (bb_std * self.bollinger_std)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
        df["BB_position"] = (df[price_col] - df["BB_lower"]) / (
            df["BB_upper"] - df["BB_lower"]
        )

        # Momentum
        df["momentum"] = df[price_col] / df[price_col].shift(10) - 1

        logger.debug(f"Created technical indicators. Shape: {df.shape}")

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def add_lagged_features(
        self, data: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Add lagged features.

        Args:
            data: Input DataFrame
            feature_cols: Columns to create lags for

        Returns:
            DataFrame with lagged features
        """
        df = data.copy()
        lagged_cols = []

        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping lags")
                continue

            for lag in range(1, self.n_lags + 1):
                col_name = f"{col}_lag_{lag}"
                df[col_name] = df[col].shift(lag)
                lagged_cols.append(col_name)

        logger.debug(f"Created {len(lagged_cols)} lagged features")

        return df

    def fit_transform(
        self, data: pd.DataFrame, price_col: str, target_col: str = "direction"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fit feature engineering on training data and transform.

        CRITICAL: This method should ONLY be called on training data.
        The scaler is fitted here and should be used with transform() for test data.

        Args:
            data: Training data
            price_col: Name of price column
            target_col: Name of target column

        Returns:
            Tuple of (transformed DataFrame, list of feature columns)
        """
        self.price_column = price_col

        # Create returns
        df = self.create_returns(data, price_col)

        # Create technical indicators
        df = self.create_technical_indicators(df, price_col)

        # Define features to use (excluding price and target)
        base_features = [
            "r",
            "SMA1",
            "SMA2",
            "SMA_diff",
            "EWMA1",
            "EWMA2",
            "EWMA_diff",
            "V1",
            "V2",
            "RSI",
            "BB_width",
            "BB_position",
            "momentum",
        ]

        # Remove excluded features (prevent data leakage)
        base_features = [f for f in base_features if f not in self.exclude_features]

        # Normalize features BEFORE adding lags
        # This prevents data leakage from future normalization
        scaler_features = [
            f for f in base_features if f in df.columns and f != "r"
        ]

        if scaler_features:
            # FIT scaler on training data
            df[scaler_features] = self.scaler.fit_transform(df[scaler_features])
            logger.info(f"Fitted scaler on {len(scaler_features)} features")

        # Add lagged features
        df = self.add_lagged_features(df, base_features)

        # Create final feature list (all lagged features)
        self.feature_columns = [
            col for col in df.columns if "_lag_" in col
        ]

        # Drop NaN values created by rolling windows and lags
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        logger.info(
            f"Dropped {dropped} rows with NaN ({dropped/initial_len:.1%}). "
            f"Final training set: {len(df)} rows"
        )

        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        logger.info(
            f"Feature engineering complete. "
            f"Features: {len(self.feature_columns)}, Rows: {len(df)}"
        )

        return df, self.feature_columns

    def transform(
        self, data: pd.DataFrame, price_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform test data using fitted scaler.

        CRITICAL: This method should ONLY be called on test data.
        It uses the scaler fitted on training data to prevent data leakage.

        Args:
            data: Test data
            price_col: Name of price column (uses fitted value if None)

        Returns:
            Transformed DataFrame with same features as training

        Raises:
            ValueError: If transform called before fit_transform
        """
        if not self.feature_columns:
            raise ValueError("Must call fit_transform on training data first")

        if price_col is None:
            price_col = self.price_column

        # Create returns
        df = self.create_returns(data, price_col)

        # Create technical indicators
        df = self.create_technical_indicators(df, price_col)

        # Define base features (same as training)
        base_features = [
            "r",
            "SMA1",
            "SMA2",
            "SMA_diff",
            "EWMA1",
            "EWMA2",
            "EWMA_diff",
            "V1",
            "V2",
            "RSI",
            "BB_width",
            "BB_position",
            "momentum",
        ]

        # Remove excluded features
        base_features = [f for f in base_features if f not in self.exclude_features]

        # TRANSFORM (not fit!) using training scaler
        scaler_features = [
            f for f in base_features if f in df.columns and f != "r"
        ]

        if scaler_features:
            # TRANSFORM using fitted scaler (NO fitting on test data!)
            df[scaler_features] = self.scaler.transform(df[scaler_features])
            logger.info(f"Transformed test data using fitted scaler")

        # Add lagged features
        df = self.add_lagged_features(df, base_features)

        # Drop NaN values
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        logger.info(
            f"Dropped {dropped} rows with NaN ({dropped/initial_len:.1%}). "
            f"Final test set: {len(df)} rows"
        )

        # Verify all expected features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")

        logger.info(f"Test data transformation complete. Rows: {len(df)}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names.

        Returns:
            List of feature column names
        """
        return self.feature_columns.copy()
