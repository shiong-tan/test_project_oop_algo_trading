"""Position sizing strategies for risk management.

This module implements various position sizing algorithms to determine
optimal position sizes based on different risk management approaches.
"""

import numpy as np
import logging
from typing import Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PositionSizingStrategy(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def calculate_size(
        self,
        capital: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float,
        **kwargs,
    ) -> float:
        """Calculate position size.

        Args:
            capital: Available trading capital
            risk_per_trade: Maximum risk amount for this trade
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            **kwargs: Additional parameters specific to strategy

        Returns:
            Position size (number of shares/units)
        """
        pass

    def _validate_inputs(
        self,
        capital: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> None:
        """Validate common input parameters.

        Args:
            capital: Available trading capital
            risk_per_trade: Maximum risk amount for this trade
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price

        Raises:
            ValueError: If any input is invalid
        """
        if capital <= 0:
            raise ValueError(f"Capital must be positive, got {capital}")

        if risk_per_trade <= 0:
            raise ValueError(f"Risk per trade must be positive, got {risk_per_trade}")

        if risk_per_trade > capital:
            raise ValueError(
                f"Risk per trade ({risk_per_trade}) cannot exceed capital ({capital})"
            )

        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        if stop_loss_price <= 0:
            raise ValueError(f"Stop loss price must be positive, got {stop_loss_price}")

        if entry_price == stop_loss_price:
            raise ValueError("Entry price and stop loss price cannot be equal")


class FixedFractionalSizing(PositionSizingStrategy):
    """Fixed fractional position sizing (risk-based).

    This strategy risks a fixed percentage of capital on each trade.
    The position size is calculated based on the distance to stop loss.

    Default: Risk 1% of capital per trade.
    """

    def __init__(self, risk_fraction: float = 0.01, max_position_fraction: float = 0.25):
        """Initialize FixedFractionalSizing.

        Args:
            risk_fraction: Fraction of capital to risk per trade (default: 0.01 = 1%)
            max_position_fraction: Maximum fraction of capital in single position (default: 0.25 = 25%)

        Raises:
            ValueError: If fractions are invalid
        """
        if not 0 < risk_fraction <= 1:
            raise ValueError(
                f"Risk fraction must be between 0 and 1, got {risk_fraction}"
            )

        if not 0 < max_position_fraction <= 1:
            raise ValueError(
                f"Max position fraction must be between 0 and 1, got {max_position_fraction}"
            )

        self.risk_fraction = risk_fraction
        self.max_position_fraction = max_position_fraction

        logger.info(
            f"Initialized FixedFractionalSizing: risk={risk_fraction:.1%}, "
            f"max_position={max_position_fraction:.1%}"
        )

    def calculate_size(
        self,
        capital: float,
        risk_per_trade: Optional[float] = None,
        entry_price: float = 0.0,
        stop_loss_price: float = 0.0,
        **kwargs,
    ) -> float:
        """Calculate position size based on fixed fractional risk.

        Args:
            capital: Available trading capital
            risk_per_trade: Override risk amount (if None, uses risk_fraction * capital)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            **kwargs: Additional parameters (ignored)

        Returns:
            Position size (number of shares/units)

        Raises:
            ValueError: If inputs are invalid
        """
        # Calculate risk amount if not provided
        if risk_per_trade is None:
            risk_per_trade = capital * self.risk_fraction

        # Validate inputs
        self._validate_inputs(capital, risk_per_trade, entry_price, stop_loss_price)

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            raise ValueError("Risk per share cannot be zero")

        # Calculate position size
        position_size = risk_per_trade / risk_per_share

        # Apply maximum position size constraint
        max_position_size = (capital * self.max_position_fraction) / entry_price
        position_size = min(position_size, max_position_size)

        # Ensure non-negative
        position_size = max(0, position_size)

        logger.debug(
            f"FixedFractional sizing: capital={capital:.2f}, risk={risk_per_trade:.2f}, "
            f"entry={entry_price:.2f}, stop={stop_loss_price:.2f}, size={position_size:.2f}"
        )

        return position_size


class KellyCriterion(PositionSizingStrategy):
    """Kelly Criterion position sizing.

    Uses the Kelly formula to calculate optimal position size based on
    win rate and reward/risk ratio. Includes fractional Kelly for safety.

    Formula: f = (p * b - q) / b
    where:
        f = fraction of capital to bet
        p = probability of winning
        q = probability of losing (1 - p)
        b = reward/risk ratio (win amount / loss amount)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_fraction: float = 0.25,
        min_win_rate: float = 0.40,
    ):
        """Initialize KellyCriterion.

        Args:
            kelly_fraction: Fraction of Kelly to use (default: 0.25 = conservative)
            max_position_fraction: Maximum fraction of capital in single position
            min_win_rate: Minimum win rate required to take position (default: 0.40)

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < kelly_fraction <= 1:
            raise ValueError(
                f"Kelly fraction must be between 0 and 1, got {kelly_fraction}"
            )

        if not 0 < max_position_fraction <= 1:
            raise ValueError(
                f"Max position fraction must be between 0 and 1, got {max_position_fraction}"
            )

        if not 0 < min_win_rate < 1:
            raise ValueError(
                f"Min win rate must be between 0 and 1, got {min_win_rate}"
            )

        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.min_win_rate = min_win_rate

        logger.info(
            f"Initialized KellyCriterion: kelly_fraction={kelly_fraction:.1%}, "
            f"max_position={max_position_fraction:.1%}, min_win_rate={min_win_rate:.1%}"
        )

    def calculate_size(
        self,
        capital: float,
        risk_per_trade: float = 0.0,
        entry_price: float = 0.0,
        stop_loss_price: float = 0.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        reward_risk_ratio: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Calculate position size using Kelly Criterion.

        Args:
            capital: Available trading capital
            risk_per_trade: Not used (for interface compatibility)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            win_rate: Historical win rate (probability of winning)
            avg_win: Average win amount
            avg_loss: Average loss amount
            reward_risk_ratio: Reward/risk ratio (alternative to avg_win/avg_loss)
            **kwargs: Additional parameters (ignored)

        Returns:
            Position size (number of shares/units)

        Raises:
            ValueError: If inputs are invalid or insufficient
        """
        # Validate basic inputs
        if capital <= 0:
            raise ValueError(f"Capital must be positive, got {capital}")

        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        if stop_loss_price <= 0:
            raise ValueError(f"Stop loss price must be positive, got {stop_loss_price}")

        # Validate Kelly-specific inputs
        if win_rate is None:
            raise ValueError("win_rate is required for Kelly Criterion")

        if not 0 < win_rate < 1:
            raise ValueError(f"Win rate must be between 0 and 1, got {win_rate}")

        # Calculate reward/risk ratio
        if reward_risk_ratio is None:
            if avg_win is None or avg_loss is None:
                raise ValueError(
                    "Either reward_risk_ratio or (avg_win and avg_loss) must be provided"
                )

            if avg_loss <= 0:
                raise ValueError(f"Average loss must be positive, got {avg_loss}")

            if avg_win <= 0:
                raise ValueError(f"Average win must be positive, got {avg_win}")

            reward_risk_ratio = avg_win / avg_loss

        if reward_risk_ratio <= 0:
            raise ValueError(
                f"Reward/risk ratio must be positive, got {reward_risk_ratio}"
            )

        # Check minimum win rate
        if win_rate < self.min_win_rate:
            logger.warning(
                f"Win rate {win_rate:.1%} below minimum {self.min_win_rate:.1%}, "
                f"returning zero position"
            )
            return 0.0

        # Calculate Kelly fraction
        loss_rate = 1 - win_rate
        kelly_f = (win_rate * reward_risk_ratio - loss_rate) / reward_risk_ratio

        # Kelly can be negative if edge is negative
        if kelly_f <= 0:
            logger.warning(
                f"Kelly fraction is negative ({kelly_f:.4f}), no edge detected, "
                f"returning zero position"
            )
            return 0.0

        # Apply fractional Kelly for safety
        kelly_f = kelly_f * self.kelly_fraction

        # Cap at maximum position fraction
        kelly_f = min(kelly_f, self.max_position_fraction)

        # Calculate position value and size
        position_value = capital * kelly_f
        position_size = position_value / entry_price

        logger.debug(
            f"Kelly sizing: capital={capital:.2f}, win_rate={win_rate:.2%}, "
            f"RR_ratio={reward_risk_ratio:.2f}, kelly_f={kelly_f:.2%}, "
            f"size={position_size:.2f}"
        )

        return position_size


class ConfidenceBasedSizing(PositionSizingStrategy):
    """Confidence-based position sizing using ML model probabilities.

    Adjusts position size based on model prediction confidence.
    Higher confidence = larger position (within risk limits).
    """

    def __init__(
        self,
        base_risk_fraction: float = 0.01,
        max_risk_fraction: float = 0.02,
        min_confidence: float = 0.55,
        max_position_fraction: float = 0.25,
    ):
        """Initialize ConfidenceBasedSizing.

        Args:
            base_risk_fraction: Base risk fraction at min_confidence (default: 0.01 = 1%)
            max_risk_fraction: Maximum risk fraction at 100% confidence (default: 0.02 = 2%)
            min_confidence: Minimum confidence to take position (default: 0.55)
            max_position_fraction: Maximum fraction of capital in single position

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < base_risk_fraction <= 1:
            raise ValueError(
                f"Base risk fraction must be between 0 and 1, got {base_risk_fraction}"
            )

        if not 0 < max_risk_fraction <= 1:
            raise ValueError(
                f"Max risk fraction must be between 0 and 1, got {max_risk_fraction}"
            )

        if base_risk_fraction > max_risk_fraction:
            raise ValueError(
                f"Base risk ({base_risk_fraction}) cannot exceed max risk ({max_risk_fraction})"
            )

        if not 0 < min_confidence < 1:
            raise ValueError(
                f"Min confidence must be between 0 and 1, got {min_confidence}"
            )

        if not 0 < max_position_fraction <= 1:
            raise ValueError(
                f"Max position fraction must be between 0 and 1, got {max_position_fraction}"
            )

        self.base_risk_fraction = base_risk_fraction
        self.max_risk_fraction = max_risk_fraction
        self.min_confidence = min_confidence
        self.max_position_fraction = max_position_fraction

        logger.info(
            f"Initialized ConfidenceBasedSizing: base_risk={base_risk_fraction:.1%}, "
            f"max_risk={max_risk_fraction:.1%}, min_confidence={min_confidence:.1%}"
        )

    def calculate_size(
        self,
        capital: float,
        risk_per_trade: Optional[float] = None,
        entry_price: float = 0.0,
        stop_loss_price: float = 0.0,
        confidence: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Calculate position size based on prediction confidence.

        Args:
            capital: Available trading capital
            risk_per_trade: Override risk amount (calculated from confidence if None)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            confidence: Model prediction confidence (probability)
            **kwargs: Additional parameters (ignored)

        Returns:
            Position size (number of shares/units)

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate confidence
        if confidence is None:
            raise ValueError("confidence is required for ConfidenceBasedSizing")

        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        # Check minimum confidence threshold
        if confidence < self.min_confidence:
            logger.debug(
                f"Confidence {confidence:.2%} below minimum {self.min_confidence:.2%}, "
                f"returning zero position"
            )
            return 0.0

        # Calculate risk fraction based on confidence
        # Linear interpolation between min_confidence and 1.0
        if confidence >= 1.0:
            risk_fraction = self.max_risk_fraction
        else:
            # Map confidence range [min_confidence, 1.0] to [base_risk, max_risk]
            confidence_range = 1.0 - self.min_confidence
            risk_range = self.max_risk_fraction - self.base_risk_fraction
            normalized_confidence = (confidence - self.min_confidence) / confidence_range
            risk_fraction = self.base_risk_fraction + (normalized_confidence * risk_range)

        # Calculate risk amount
        if risk_per_trade is None:
            risk_per_trade = capital * risk_fraction

        # Validate inputs
        self._validate_inputs(capital, risk_per_trade, entry_price, stop_loss_price)

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            raise ValueError("Risk per share cannot be zero")

        # Calculate position size
        position_size = risk_per_trade / risk_per_share

        # Apply maximum position size constraint
        max_position_size = (capital * self.max_position_fraction) / entry_price
        position_size = min(position_size, max_position_size)

        # Ensure non-negative
        position_size = max(0, position_size)

        logger.debug(
            f"ConfidenceBased sizing: capital={capital:.2f}, confidence={confidence:.2%}, "
            f"risk_fraction={risk_fraction:.2%}, risk={risk_per_trade:.2f}, "
            f"entry={entry_price:.2f}, stop={stop_loss_price:.2f}, size={position_size:.2f}"
        )

        return position_size


def create_position_sizer(
    method: str,
    **kwargs,
) -> PositionSizingStrategy:
    """Factory function to create position sizing strategy.

    Args:
        method: Position sizing method ('fixed_fractional', 'kelly', 'confidence_based')
        **kwargs: Parameters for the specific strategy

    Returns:
        PositionSizingStrategy instance

    Raises:
        ValueError: If method is unknown

    Examples:
        >>> sizer = create_position_sizer('fixed_fractional', risk_fraction=0.01)
        >>> sizer = create_position_sizer('kelly', kelly_fraction=0.25)
        >>> sizer = create_position_sizer('confidence_based', min_confidence=0.55)
    """
    methods = {
        "fixed_fractional": FixedFractionalSizing,
        "kelly": KellyCriterion,
        "confidence_based": ConfidenceBasedSizing,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown position sizing method: {method}. "
            f"Available: {list(methods.keys())}"
        )

    strategy_class = methods[method]

    try:
        return strategy_class(**kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for {method} strategy: {e}"
        ) from e
