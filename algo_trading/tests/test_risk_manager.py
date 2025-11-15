"""Unit tests for risk manager module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from algo_trading.risk.manager import RiskManager, Position, DrawdownZone


class TestPosition:
    """Test suite for Position dataclass."""

    def test_position_creation(self):
        """Test Position creation."""
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
        )

        assert position.symbol == "AAPL"
        assert position.entry_price == 100.0
        assert position.quantity == 10.0
        assert position.stop_loss == 97.0
        assert position.highest_price == 100.0

    def test_position_value(self):
        """Test position value calculation."""
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
        )

        assert position.position_value == 1000.0

    def test_risk_amount(self):
        """Test risk amount calculation."""
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
        )

        # Risk = (100 - 97) * 10 = 30
        assert position.risk_amount == 30.0

    def test_update_trailing_stop(self):
        """Test trailing stop update."""
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
            trailing_stop_pct=0.02,
        )

        # Price moves up
        updated = position.update_trailing_stop(105.0)

        assert updated is True
        assert position.highest_price == 105.0
        assert position.stop_loss == 105.0 * 0.98  # 2% trailing stop

    def test_should_exit_stop_loss(self):
        """Test exit condition - stop loss."""
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
        )

        should_exit, reason = position.should_exit(96.0, datetime(2023, 1, 2))

        assert should_exit is True
        assert reason == "stop_loss"

    def test_should_exit_take_profit(self):
        """Test exit condition - take profit."""
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
            take_profit=105.0,
        )

        should_exit, reason = position.should_exit(106.0, datetime(2023, 1, 2))

        assert should_exit is True
        assert reason == "take_profit"

    def test_should_exit_time_stop(self):
        """Test exit condition - time stop."""
        entry_date = datetime(2023, 1, 1)
        position = Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=entry_date,
            stop_loss=97.0,
            time_stop_days=30,
        )

        exit_date = entry_date + timedelta(days=31)
        should_exit, reason = position.should_exit(102.0, exit_date)

        assert should_exit is True
        assert reason == "time_stop"


class TestRiskManager:
    """Test suite for RiskManager class."""

    def test_initialization(self):
        """Test RiskManager initialization."""
        risk_mgr = RiskManager(initial_capital=10000)

        assert risk_mgr.initial_capital == 10000
        assert risk_mgr.current_capital == 10000
        assert risk_mgr.peak_capital == 10000
        assert len(risk_mgr.open_positions) == 0

    def test_initialization_invalid_capital(self):
        """Test initialization with invalid capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            RiskManager(initial_capital=-1000)

    def test_get_drawdown_zone(self):
        """Test drawdown zone determination."""
        risk_mgr = RiskManager(initial_capital=10000)

        # Normal zone
        assert risk_mgr.get_drawdown_zone() == DrawdownZone.NORMAL

        # Simulate 8% drawdown -> Caution zone
        risk_mgr.current_capital = 9200
        risk_mgr.peak_capital = 10000
        assert risk_mgr.get_drawdown_zone() == DrawdownZone.CAUTION

        # Simulate 12% drawdown -> Alert zone
        risk_mgr.current_capital = 8800
        assert risk_mgr.get_drawdown_zone() == DrawdownZone.ALERT

        # Simulate 18% drawdown -> Emergency zone
        risk_mgr.current_capital = 8200
        assert risk_mgr.get_drawdown_zone() == DrawdownZone.EMERGENCY

    def test_get_current_drawdown(self):
        """Test current drawdown calculation."""
        risk_mgr = RiskManager(initial_capital=10000)

        # No drawdown initially
        assert risk_mgr.get_current_drawdown() == 0.0

        # Simulate 10% drawdown
        risk_mgr.current_capital = 9000
        risk_mgr.peak_capital = 10000
        assert abs(risk_mgr.get_current_drawdown() - 0.10) < 0.001

    def test_calculate_stop_loss(self):
        """Test stop loss calculation."""
        risk_mgr = RiskManager(initial_capital=10000, hard_stop_pct=0.03)

        # Long position: stop below entry
        stop = risk_mgr.calculate_stop_loss(entry_price=100.0, direction=1)
        assert stop == 97.0

        # Short position: stop above entry
        stop = risk_mgr.calculate_stop_loss(entry_price=100.0, direction=-1)
        assert stop == 103.0

    def test_calculate_take_profit(self):
        """Test take profit calculation."""
        risk_mgr = RiskManager(initial_capital=10000, take_profit_pct=0.05)

        # Long position: target above entry
        target = risk_mgr.calculate_take_profit(entry_price=100.0, direction=1)
        assert target == 105.0

        # Short position: target below entry
        target = risk_mgr.calculate_take_profit(entry_price=100.0, direction=-1)
        assert target == 95.0

    def test_validate_trade_success(self):
        """Test successful trade validation."""
        risk_mgr = RiskManager(initial_capital=10000)

        is_valid, issues = risk_mgr.validate_trade(
            symbol="AAPL",
            quantity=10.0,
            entry_price=100.0,
            stop_loss=97.0,
            confidence=0.60,
        )

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_trade_insufficient_capital(self):
        """Test trade validation with insufficient capital."""
        risk_mgr = RiskManager(initial_capital=1000)

        is_valid, issues = risk_mgr.validate_trade(
            symbol="AAPL",
            quantity=20.0,
            entry_price=100.0,
            stop_loss=97.0,
        )

        assert is_valid is False
        assert any("Insufficient capital" in issue for issue in issues)

    def test_validate_trade_position_too_large(self):
        """Test trade validation with position too large."""
        risk_mgr = RiskManager(initial_capital=10000)

        is_valid, issues = risk_mgr.validate_trade(
            symbol="AAPL",
            quantity=30.0,  # $3000 position on $10000 capital = 30% > 25% limit
            entry_price=100.0,
            stop_loss=97.0,
        )

        assert is_valid is False
        assert any("Position too large" in issue for issue in issues)

    def test_validate_trade_low_confidence(self):
        """Test trade validation with low confidence."""
        risk_mgr = RiskManager(initial_capital=10000)

        is_valid, issues = risk_mgr.validate_trade(
            symbol="AAPL",
            quantity=10.0,
            entry_price=100.0,
            stop_loss=97.0,
            confidence=0.50,
            min_confidence=0.55,
        )

        assert is_valid is False
        assert any("Confidence" in issue for issue in issues)

    def test_open_position(self):
        """Test opening a position."""
        risk_mgr = RiskManager(initial_capital=10000)

        result = risk_mgr.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
        )

        assert result is True
        assert "AAPL" in risk_mgr.open_positions
        assert risk_mgr.open_positions["AAPL"].quantity == 10.0

    def test_open_position_duplicate(self):
        """Test opening duplicate position raises error."""
        risk_mgr = RiskManager(initial_capital=10000)

        risk_mgr.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
        )

        with pytest.raises(ValueError, match="Position already exists"):
            risk_mgr.open_position(
                symbol="AAPL",
                entry_price=105.0,
                quantity=5.0,
                entry_date=datetime(2023, 1, 2),
            )

    def test_close_position(self):
        """Test closing a position."""
        risk_mgr = RiskManager(initial_capital=10000)

        # Open position
        risk_mgr.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
        )

        # Close position at profit
        trade_record = risk_mgr.close_position(
            symbol="AAPL",
            exit_price=105.0,
            exit_date=datetime(2023, 1, 10),
            reason="take_profit",
        )

        assert trade_record is not None
        assert trade_record["pnl"] == 50.0  # (105 - 100) * 10
        assert "AAPL" not in risk_mgr.open_positions
        assert len(risk_mgr.trade_history) == 1

    def test_update_positions(self):
        """Test updating positions and checking exits."""
        risk_mgr = RiskManager(initial_capital=10000)

        # Open position
        risk_mgr.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,
        )

        # Update with price that triggers stop loss
        current_prices = {"AAPL": 96.0}
        closed_trades = risk_mgr.update_positions(
            current_prices, datetime(2023, 1, 2)
        )

        assert len(closed_trades) == 1
        assert closed_trades[0]["reason"] == "stop_loss"
        assert "AAPL" not in risk_mgr.open_positions

    def test_calculate_var(self):
        """Test VaR calculation."""
        risk_mgr = RiskManager(initial_capital=10000)

        # Add some trade history
        risk_mgr.trade_history = [
            {"pnl_pct": 0.02},
            {"pnl_pct": -0.01},
            {"pnl_pct": 0.03},
            {"pnl_pct": -0.02},
            {"pnl_pct": 0.01},
        ]

        var = risk_mgr.calculate_var(confidence_level=0.95)

        assert var >= 0  # VaR should be positive (loss amount)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        risk_mgr = RiskManager(initial_capital=10000)

        # Simulate equity curve
        risk_mgr.equity_curve = [10000, 11000, 10500, 9500, 9000, 10000, 11500]

        max_dd, max_dd_duration = risk_mgr.calculate_max_drawdown()

        # Max drawdown from 11000 to 9000 = 18.18%
        assert max_dd > 0.15
        assert max_dd < 0.20
        assert max_dd_duration > 0

    def test_get_portfolio_heat(self):
        """Test portfolio heat calculation."""
        risk_mgr = RiskManager(initial_capital=10000)

        # Open positions with known risk
        risk_mgr.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=datetime(2023, 1, 1),
            stop_loss=97.0,  # Risk = $30
        )

        heat = risk_mgr.get_portfolio_heat()

        # Heat = $30 / $10000 = 0.003 (0.3%)
        assert abs(heat - 0.003) < 0.0001

    def test_reset_daily_tracking(self):
        """Test resetting daily tracking."""
        risk_mgr = RiskManager(initial_capital=10000)
        risk_mgr.current_capital = 9500

        risk_mgr.reset_daily_tracking()

        assert risk_mgr.daily_start_capital == 9500
