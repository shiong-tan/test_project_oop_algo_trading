"""Unit tests for position sizing module."""

import pytest
import numpy as np
from algo_trading.risk.position_sizing import (
    FixedFractionalSizing,
    KellyCriterion,
    ConfidenceBasedSizing,
    create_position_sizer,
)


class TestFixedFractionalSizing:
    """Test suite for FixedFractionalSizing."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        sizer = FixedFractionalSizing()
        assert sizer.risk_fraction == 0.01
        assert sizer.max_position_fraction == 0.25

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        sizer = FixedFractionalSizing(risk_fraction=0.02, max_position_fraction=0.30)
        assert sizer.risk_fraction == 0.02
        assert sizer.max_position_fraction == 0.30

    def test_initialization_invalid_risk_fraction(self):
        """Test initialization with invalid risk fraction."""
        with pytest.raises(ValueError, match="Risk fraction must be between 0 and 1"):
            FixedFractionalSizing(risk_fraction=1.5)

    def test_calculate_size(self):
        """Test position size calculation."""
        sizer = FixedFractionalSizing(risk_fraction=0.01)

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
        )

        # Risk per trade = $100 (1% of $10000)
        # Risk per share = $3 (100 - 97)
        # Position size = $100 / $3 = 33.33 shares
        assert abs(size - 33.33) < 0.01

    def test_calculate_size_max_position_limit(self):
        """Test that maximum position size is enforced."""
        sizer = FixedFractionalSizing(risk_fraction=0.10, max_position_fraction=0.25)

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=99.0,  # Very tight stop = large position
        )

        # Max position = $2500 (25% of $10000) = 25 shares at $100
        assert size == 25.0

    def test_calculate_size_invalid_capital(self):
        """Test with invalid capital."""
        sizer = FixedFractionalSizing()

        with pytest.raises(ValueError, match="Capital must be positive"):
            sizer.calculate_size(capital=-1000, entry_price=100.0, stop_loss_price=97.0)


class TestKellyCriterion:
    """Test suite for KellyCriterion."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        sizer = KellyCriterion()
        assert sizer.kelly_fraction == 0.25
        assert sizer.max_position_fraction == 0.25
        assert sizer.min_win_rate == 0.40

    def test_calculate_size_positive_edge(self):
        """Test position sizing with positive edge."""
        sizer = KellyCriterion(kelly_fraction=0.25)

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
            win_rate=0.55,
            avg_win=2.0,
            avg_loss=1.0,
        )

        # Kelly% = (0.55 * 2.0 - 0.45) / 2.0 = 0.325
        # Fractional Kelly = 0.325 * 0.25 = 0.08125
        # Position value = $10000 * 0.08125 = $812.50
        # Position size = $812.50 / $100 = 8.125 shares
        assert size > 0
        assert size < 100  # Reasonable upper bound

    def test_calculate_size_negative_edge(self):
        """Test position sizing with negative edge returns zero."""
        sizer = KellyCriterion()

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
            win_rate=0.40,
            avg_win=1.0,
            avg_loss=2.0,
        )

        # Negative edge should return zero position
        assert size == 0.0

    def test_calculate_size_low_win_rate(self):
        """Test that low win rate returns zero position."""
        sizer = KellyCriterion(min_win_rate=0.50)

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
            win_rate=0.45,  # Below minimum
            avg_win=2.0,
            avg_loss=1.0,
        )

        assert size == 0.0

    def test_calculate_size_missing_params(self):
        """Test that missing required parameters raises error."""
        sizer = KellyCriterion()

        with pytest.raises(ValueError, match="win_rate is required"):
            sizer.calculate_size(
                capital=10000,
                entry_price=100.0,
                stop_loss_price=97.0,
            )


class TestConfidenceBasedSizing:
    """Test suite for ConfidenceBasedSizing."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        sizer = ConfidenceBasedSizing()
        assert sizer.base_risk_fraction == 0.01
        assert sizer.max_risk_fraction == 0.02
        assert sizer.min_confidence == 0.55

    def test_calculate_size_high_confidence(self):
        """Test position sizing with high confidence."""
        sizer = ConfidenceBasedSizing(base_risk_fraction=0.01, max_risk_fraction=0.02)

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
            confidence=0.80,  # High confidence
        )

        # At confidence=0.80, risk should be scaled up from base
        # Higher than base (1%), lower than max (2%)
        assert size > 33.33  # Base size
        assert size < 66.67  # Max size

    def test_calculate_size_min_confidence(self):
        """Test position sizing at minimum confidence."""
        sizer = ConfidenceBasedSizing(
            base_risk_fraction=0.01,
            max_risk_fraction=0.02,
            min_confidence=0.55,
        )

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
            confidence=0.55,  # At minimum
        )

        # At min confidence, should use base risk
        assert abs(size - 33.33) < 0.5

    def test_calculate_size_below_threshold(self):
        """Test that confidence below threshold returns zero."""
        sizer = ConfidenceBasedSizing(min_confidence=0.55)

        size = sizer.calculate_size(
            capital=10000,
            entry_price=100.0,
            stop_loss_price=97.0,
            confidence=0.50,  # Below threshold
        )

        assert size == 0.0

    def test_calculate_size_missing_confidence(self):
        """Test that missing confidence parameter raises error."""
        sizer = ConfidenceBasedSizing()

        with pytest.raises(ValueError, match="confidence is required"):
            sizer.calculate_size(
                capital=10000,
                entry_price=100.0,
                stop_loss_price=97.0,
            )

    def test_calculate_size_invalid_confidence(self):
        """Test with invalid confidence value."""
        sizer = ConfidenceBasedSizing()

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            sizer.calculate_size(
                capital=10000,
                entry_price=100.0,
                stop_loss_price=97.0,
                confidence=1.5,
            )


class TestFactoryFunction:
    """Test suite for create_position_sizer factory function."""

    def test_create_fixed_fractional(self):
        """Test creating FixedFractionalSizing."""
        sizer = create_position_sizer("fixed_fractional", risk_fraction=0.02)

        assert isinstance(sizer, FixedFractionalSizing)
        assert sizer.risk_fraction == 0.02

    def test_create_kelly(self):
        """Test creating KellyCriterion."""
        sizer = create_position_sizer("kelly", kelly_fraction=0.5)

        assert isinstance(sizer, KellyCriterion)
        assert sizer.kelly_fraction == 0.5

    def test_create_confidence_based(self):
        """Test creating ConfidenceBasedSizing."""
        sizer = create_position_sizer("confidence_based", min_confidence=0.60)

        assert isinstance(sizer, ConfidenceBasedSizing)
        assert sizer.min_confidence == 0.60

    def test_create_unknown_method(self):
        """Test creating unknown position sizer."""
        with pytest.raises(ValueError, match="Unknown position sizing method"):
            create_position_sizer("unknown_method")

    def test_create_invalid_params(self):
        """Test creating with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid parameters"):
            create_position_sizer("fixed_fractional", invalid_param=123)
