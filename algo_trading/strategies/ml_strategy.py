"""Machine Learning-based trading strategy.

This module implements a prediction-based trading strategy that uses
ML model probabilities to generate trading signals with confidence scores.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Any, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """Generate trading signals from data.

        Args:
            data: Market data
            **kwargs: Additional parameters

        Returns:
            Series of signals (-1, 0, 1)
        """
        pass


class MLPredictionStrategy(TradingStrategy):
    """ML-based prediction strategy.

    Uses machine learning model predictions and confidence scores to generate
    trading signals. Only trades when model confidence exceeds threshold.

    Signals:
    - 1: BUY (model predicts up with confidence > threshold)
    - 0: HOLD (confidence below threshold)
    - -1: SELL (exit position)
    """

    def __init__(
        self,
        model: Any,
        confidence_threshold: float = 0.55,
        min_confidence_buy: Optional[float] = None,
        min_confidence_sell: Optional[float] = None,
        feature_columns: Optional[list] = None,
        enable_shorting: bool = False,
    ):
        """Initialize MLPredictionStrategy.

        Args:
            model: Trained ML model with predict_proba method
            confidence_threshold: Minimum confidence to generate buy signal (default: 0.55)
            min_confidence_buy: Override confidence threshold for buys (optional)
            min_confidence_sell: Confidence threshold for sells when holding (optional)
            feature_columns: List of feature column names to use (optional)
            enable_shorting: Enable short selling (default: False)

        Raises:
            ValueError: If parameters are invalid
            TypeError: If model doesn't support predict_proba
        """
        # Validate model
        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"Model {type(model).__name__} does not support predict_proba(). "
                f"Please use a probabilistic classifier."
            )

        if not 0 < confidence_threshold < 1:
            raise ValueError(
                f"Confidence threshold must be between 0 and 1, got {confidence_threshold}"
            )

        self.model = model
        self.confidence_threshold = confidence_threshold
        self.min_confidence_buy = min_confidence_buy or confidence_threshold
        self.min_confidence_sell = min_confidence_sell or 0.5  # Default to neutral
        self.feature_columns = feature_columns
        self.enable_shorting = enable_shorting

        logger.info(
            f"MLPredictionStrategy initialized: confidence_threshold={confidence_threshold:.2%}, "
            f"shorting={'enabled' if enable_shorting else 'disabled'}"
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Market data

        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Cannot generate signals from empty data")

        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing required feature columns: {missing_cols}. "
                    f"Available: {data.columns.tolist()}"
                )

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for model prediction.

        Args:
            data: Market data with features

        Returns:
            Feature matrix ready for prediction

        Raises:
            ValueError: If features cannot be prepared
        """
        if self.feature_columns:
            # Use specified feature columns
            try:
                features = data[self.feature_columns].copy()
            except KeyError as e:
                raise ValueError(f"Feature column not found: {e}")
        else:
            # Use all numeric columns
            features = data.select_dtypes(include=[np.number]).copy()

        # Check for missing values
        if features.isnull().any().any():
            missing_counts = features.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            logger.warning(
                f"Features contain missing values:\n{missing_cols}\n"
                f"Filling with forward fill then 0"
            )
            features = features.fillna(method="ffill").fillna(0)

        # Check for infinite values
        if np.isinf(features.values).any():
            logger.warning("Features contain infinite values, replacing with 0")
            features = features.replace([np.inf, -np.inf], 0)

        return features

    def predict_proba(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate probability predictions from model.

        Args:
            data: Market data with features

        Returns:
            Tuple of (predictions, probabilities)
            - predictions: Array of class predictions (0 or 1)
            - probabilities: Array of positive class probabilities

        Raises:
            ValueError: If prediction fails
        """
        self._validate_data(data)
        features = self._prepare_features(data)

        try:
            # Get probability predictions
            proba = self.model.predict_proba(features)

            # Extract probability of positive class (class 1)
            if proba.shape[1] == 2:
                # Binary classification: use probability of class 1
                probabilities = proba[:, 1]
            else:
                raise ValueError(
                    f"Expected 2 classes, got {proba.shape[1]}. "
                    f"This strategy only supports binary classification."
                )

            # Get class predictions
            predictions = self.model.predict(features)

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise ValueError(f"Failed to generate predictions: {e}") from e

    def generate_signals(
        self,
        data: pd.DataFrame,
        return_confidence: bool = False,
        **kwargs,
    ) -> pd.Series:
        """Generate trading signals based on ML predictions.

        Args:
            data: Market data with features
            return_confidence: If True, return (signals, confidences) tuple
            **kwargs: Additional parameters (ignored)

        Returns:
            Series of trading signals (-1=sell, 0=hold, 1=buy)
            If return_confidence=True, returns tuple (signals, confidences)

        Raises:
            ValueError: If signal generation fails
        """
        logger.info(f"Generating signals for {len(data)} periods")

        # Get predictions and probabilities
        predictions, probabilities = self.predict_proba(data)

        # Initialize signals (0 = hold)
        signals = pd.Series(0, index=data.index, name="signal")
        confidences = pd.Series(probabilities, index=data.index, name="confidence")

        # Track position state
        in_position = False

        for i, (idx, prob) in enumerate(confidences.items()):
            pred = predictions[i]

            if not in_position:
                # Look for entry signal
                if pred == 1 and prob >= self.min_confidence_buy:
                    # Strong bullish signal -> BUY
                    signals[idx] = 1
                    in_position = True
                    logger.debug(
                        f"{idx}: BUY signal (confidence={prob:.2%})"
                    )
                elif self.enable_shorting and pred == 0 and prob <= (1 - self.min_confidence_buy):
                    # Strong bearish signal -> SHORT (represented as -1)
                    signals[idx] = -1
                    in_position = True
                    logger.debug(
                        f"{idx}: SHORT signal (confidence={1-prob:.2%})"
                    )
            else:
                # Already in position, look for exit signal
                if pred == 0 or prob < self.min_confidence_sell:
                    # Confidence dropped or model predicts down -> SELL/CLOSE
                    signals[idx] = -1
                    in_position = False
                    logger.debug(
                        f"{idx}: SELL signal (confidence={prob:.2%})"
                    )

        # Log summary
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        hold_signals = (signals == 0).sum()

        logger.info(
            f"Generated {buy_signals} BUY, {sell_signals} SELL, "
            f"{hold_signals} HOLD signals"
        )

        if return_confidence:
            return signals, confidences
        else:
            return signals

    def generate_signals_simple(
        self,
        data: pd.DataFrame,
        return_confidence: bool = False,
        **kwargs,
    ) -> pd.Series:
        """Generate simple signals without position tracking.

        Simpler logic:
        - Signal 1 (BUY) when model predicts up with high confidence
        - Signal 0 (HOLD) when confidence is low
        - Signal -1 (SELL) when model predicts down or confidence drops

        This is useful for backtesting where position tracking is done separately.

        Args:
            data: Market data with features
            return_confidence: If True, return (signals, confidences) tuple
            **kwargs: Additional parameters (ignored)

        Returns:
            Series of trading signals (-1=sell, 0=hold, 1=buy)
            If return_confidence=True, returns tuple (signals, confidences)
        """
        logger.info(f"Generating simple signals for {len(data)} periods")

        # Get predictions and probabilities
        predictions, probabilities = self.predict_proba(data)

        # Initialize signals
        signals = pd.Series(0, index=data.index, name="signal")
        confidences = pd.Series(probabilities, index=data.index, name="confidence")

        # Generate signals based on predictions and confidence
        for i, (idx, prob) in enumerate(confidences.items()):
            pred = predictions[i]

            if pred == 1 and prob >= self.confidence_threshold:
                # Bullish with high confidence
                signals[idx] = 1
            elif pred == 0:
                # Bearish prediction
                signals[idx] = -1

            # Note: Signal 0 (hold) is the default

        # Log summary
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        hold_signals = (signals == 0).sum()

        logger.info(
            f"Generated {buy_signals} BUY, {sell_signals} SELL, "
            f"{hold_signals} HOLD signals (simple mode)"
        )

        if return_confidence:
            return signals, confidences
        else:
            return signals

    def get_signal_statistics(self, signals: pd.Series) -> Dict[str, Any]:
        """Calculate signal statistics.

        Args:
            signals: Series of trading signals

        Returns:
            Dictionary of signal statistics
        """
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        hold_count = (signals == 0).sum()
        total = len(signals)

        stats = {
            "total_periods": total,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "buy_pct": buy_count / total if total > 0 else 0,
            "sell_pct": sell_count / total if total > 0 else 0,
            "hold_pct": hold_count / total if total > 0 else 0,
            "total_trades": buy_count + sell_count,
            "trade_frequency": (buy_count + sell_count) / total if total > 0 else 0,
        }

        return stats

    def optimize_threshold(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        threshold_range: Tuple[float, float] = (0.5, 0.9),
        step: float = 0.05,
    ) -> Tuple[float, float]:
        """Optimize confidence threshold based on historical returns.

        Args:
            data: Market data with features
            returns: Series of actual returns
            threshold_range: Tuple of (min_threshold, max_threshold)
            step: Step size for threshold search

        Returns:
            Tuple of (best_threshold, best_return)

        Raises:
            ValueError: If inputs are invalid
        """
        if len(data) != len(returns):
            raise ValueError(
                f"Data and returns length mismatch: {len(data)} != {len(returns)}"
            )

        min_thresh, max_thresh = threshold_range

        if not 0 < min_thresh < max_thresh < 1:
            raise ValueError(
                f"Invalid threshold range: ({min_thresh}, {max_thresh}). "
                f"Must be between 0 and 1 with min < max."
            )

        logger.info(
            f"Optimizing threshold in range [{min_thresh}, {max_thresh}] "
            f"with step={step}"
        )

        best_threshold = self.confidence_threshold
        best_return = float("-inf")

        # Try different thresholds
        thresholds = np.arange(min_thresh, max_thresh + step, step)

        for thresh in thresholds:
            # Temporarily set threshold
            original_thresh = self.confidence_threshold
            self.confidence_threshold = thresh

            try:
                # Generate signals with this threshold
                signals = self.generate_signals_simple(data)

                # Calculate strategy return
                strategy_returns = signals.shift(1) * returns
                total_return = (1 + strategy_returns).prod() - 1

                logger.debug(
                    f"Threshold {thresh:.2f}: return={total_return:.2%}"
                )

                if total_return > best_return:
                    best_return = total_return
                    best_threshold = thresh

            except Exception as e:
                logger.warning(f"Failed to test threshold {thresh}: {e}")
            finally:
                # Restore original threshold
                self.confidence_threshold = original_thresh

        # Set optimal threshold
        self.confidence_threshold = best_threshold

        logger.info(
            f"Optimal threshold: {best_threshold:.2f} with return={best_return:.2%}"
        )

        return best_threshold, best_return

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy configuration information.

        Returns:
            Dictionary with strategy configuration
        """
        info = {
            "strategy_name": "MLPredictionStrategy",
            "model_type": type(self.model).__name__,
            "confidence_threshold": self.confidence_threshold,
            "min_confidence_buy": self.min_confidence_buy,
            "min_confidence_sell": self.min_confidence_sell,
            "enable_shorting": self.enable_shorting,
            "feature_columns": self.feature_columns,
            "has_predict_proba": hasattr(self.model, "predict_proba"),
        }

        return info

    def __repr__(self) -> str:
        """String representation of strategy."""
        return (
            f"MLPredictionStrategy("
            f"model={type(self.model).__name__}, "
            f"threshold={self.confidence_threshold:.2%}, "
            f"shorting={'enabled' if self.enable_shorting else 'disabled'}"
            f")"
        )


class ThresholdOptimizer:
    """Utility class for optimizing strategy thresholds."""

    @staticmethod
    def grid_search(
        strategy: MLPredictionStrategy,
        data: pd.DataFrame,
        returns: pd.Series,
        threshold_range: Tuple[float, float] = (0.5, 0.9),
        step: float = 0.05,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """Perform grid search over confidence thresholds.

        Args:
            strategy: MLPredictionStrategy instance
            data: Market data with features
            returns: Series of actual returns
            threshold_range: Tuple of (min_threshold, max_threshold)
            step: Step size for threshold search
            metric: Optimization metric ('sharpe', 'return', 'calmar')

        Returns:
            DataFrame with threshold search results

        Raises:
            ValueError: If inputs are invalid
        """
        min_thresh, max_thresh = threshold_range
        thresholds = np.arange(min_thresh, max_thresh + step, step)

        results = []

        original_thresh = strategy.confidence_threshold

        for thresh in thresholds:
            strategy.confidence_threshold = thresh

            try:
                # Generate signals
                signals = strategy.generate_signals_simple(data)

                # Calculate metrics
                strategy_returns = signals.shift(1) * returns
                total_return = (1 + strategy_returns).prod() - 1

                # Sharpe ratio
                if strategy_returns.std() > 0:
                    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                else:
                    sharpe = 0

                # Max drawdown
                cumulative = (1 + strategy_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(drawdown.min())

                # Calmar ratio
                calmar = total_return / max_dd if max_dd > 0 else 0

                results.append({
                    "threshold": thresh,
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "calmar_ratio": calmar,
                    "num_trades": (signals != 0).sum(),
                })

            except Exception as e:
                logger.warning(f"Failed to evaluate threshold {thresh}: {e}")

        # Restore original threshold
        strategy.confidence_threshold = original_thresh

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Sort by selected metric
            metric_col = {
                "sharpe": "sharpe_ratio",
                "return": "total_return",
                "calmar": "calmar_ratio",
            }.get(metric, "sharpe_ratio")

            results_df = results_df.sort_values(metric_col, ascending=False)

        return results_df
