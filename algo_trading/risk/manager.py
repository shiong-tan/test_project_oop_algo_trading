"""Risk management system for algorithmic trading.

This module implements comprehensive risk management including:
- Stop-loss management (hard stop, trailing stop, take-profit, time stop)
- Drawdown control (4-zone system)
- Portfolio heat tracking
- Trade validation
- Risk metrics calculation (VaR, CVaR, max drawdown)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class DrawdownZone(Enum):
    """Drawdown risk zones."""

    NORMAL = "normal"  # 0-5% drawdown
    CAUTION = "caution"  # 5-10% drawdown
    ALERT = "alert"  # 10-15% drawdown
    EMERGENCY = "emergency"  # >15% drawdown (halt trading)


@dataclass
class Position:
    """Represents an open trading position."""

    symbol: str
    entry_price: float
    quantity: float
    entry_date: datetime
    stop_loss: float
    trailing_stop_pct: Optional[float] = None
    take_profit: Optional[float] = None
    time_stop_days: Optional[int] = None
    highest_price: float = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields."""
        self.highest_price = self.entry_price

        # Validate inputs
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {self.entry_price}")

        if self.stop_loss <= 0:
            raise ValueError(f"Stop loss must be positive, got {self.stop_loss}")

    @property
    def position_value(self) -> float:
        """Current position value at entry price."""
        return self.entry_price * self.quantity

    @property
    def risk_amount(self) -> float:
        """Amount at risk (distance to stop loss)."""
        return abs(self.entry_price - self.stop_loss) * self.quantity

    def update_trailing_stop(self, current_price: float) -> bool:
        """Update trailing stop loss if price has moved favorably.

        Args:
            current_price: Current market price

        Returns:
            True if trailing stop was updated
        """
        if self.trailing_stop_pct is None:
            return False

        # Update highest price
        if current_price > self.highest_price:
            self.highest_price = current_price

            # Calculate new trailing stop
            new_stop = self.highest_price * (1 - self.trailing_stop_pct)

            # Only update if new stop is higher than current stop
            if new_stop > self.stop_loss:
                old_stop = self.stop_loss
                self.stop_loss = new_stop
                logger.debug(
                    f"Updated trailing stop for {self.symbol}: "
                    f"{old_stop:.2f} -> {new_stop:.2f}"
                )
                return True

        return False

    def should_exit(self, current_price: float, current_date: datetime) -> Tuple[bool, str]:
        """Check if position should be exited based on stop conditions.

        Args:
            current_price: Current market price
            current_date: Current date

        Returns:
            Tuple of (should_exit, reason)
        """
        # Check hard stop loss
        if current_price <= self.stop_loss:
            return True, "stop_loss"

        # Check take profit
        if self.take_profit and current_price >= self.take_profit:
            return True, "take_profit"

        # Check time stop
        if self.time_stop_days:
            days_held = (current_date - self.entry_date).days
            if days_held >= self.time_stop_days:
                return True, "time_stop"

        return False, ""


class RiskManager:
    """Comprehensive risk management system.

    Implements:
    - Position-level risk management (stops, trailing stops, etc.)
    - Portfolio-level risk management (drawdown, heat, daily limits)
    - Trade validation
    - Risk metrics calculation
    """

    def __init__(
        self,
        initial_capital: float,
        max_drawdown_pct: float = 0.15,
        daily_loss_limit_pct: float = 0.03,
        max_portfolio_heat: float = 0.03,
        drawdown_zones: Optional[Dict[str, float]] = None,
        hard_stop_pct: float = 0.03,
        trailing_stop_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        time_stop_days: int = 30,
    ):
        """Initialize RiskManager.

        Args:
            initial_capital: Starting capital
            max_drawdown_pct: Maximum allowable drawdown (default: 15%)
            daily_loss_limit_pct: Maximum daily loss as fraction of capital (default: 3%)
            max_portfolio_heat: Maximum total portfolio risk exposure (default: 3%)
            drawdown_zones: Drawdown zone thresholds
            hard_stop_pct: Hard stop loss percentage (default: 3%)
            trailing_stop_pct: Trailing stop percentage (default: 2%)
            take_profit_pct: Take profit percentage (default: 5%)
            time_stop_days: Maximum days to hold position (default: 30)

        Raises:
            ValueError: If parameters are invalid
        """
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")

        if not 0 < max_drawdown_pct <= 1:
            raise ValueError(
                f"Max drawdown must be between 0 and 1, got {max_drawdown_pct}"
            )

        if not 0 < daily_loss_limit_pct <= 1:
            raise ValueError(
                f"Daily loss limit must be between 0 and 1, got {daily_loss_limit_pct}"
            )

        if not 0 < max_portfolio_heat <= 1:
            raise ValueError(
                f"Max portfolio heat must be between 0 and 1, got {max_portfolio_heat}"
            )

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_portfolio_heat = max_portfolio_heat

        # Stop loss configuration
        self.hard_stop_pct = hard_stop_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.time_stop_days = time_stop_days

        # Drawdown zones
        if drawdown_zones is None:
            drawdown_zones = {
                DrawdownZone.NORMAL.value: 0.05,
                DrawdownZone.CAUTION.value: 0.10,
                DrawdownZone.ALERT.value: 0.15,
                DrawdownZone.EMERGENCY.value: 0.20,
            }
        self.drawdown_zones = drawdown_zones

        # Tracking
        self.open_positions: Dict[str, Position] = {}
        self.daily_start_capital: Optional[float] = None
        self.daily_pnl: float = 0.0
        self.equity_curve: List[float] = [initial_capital]
        self.trade_history: List[Dict[str, Any]] = []

        logger.info(
            f"RiskManager initialized: capital={initial_capital:.2f}, "
            f"max_drawdown={max_drawdown_pct:.1%}, max_heat={max_portfolio_heat:.1%}"
        )

    def get_drawdown_zone(self) -> DrawdownZone:
        """Get current drawdown zone.

        Returns:
            Current DrawdownZone
        """
        drawdown_pct = self.get_current_drawdown()

        if drawdown_pct >= self.drawdown_zones[DrawdownZone.EMERGENCY.value]:
            return DrawdownZone.EMERGENCY
        elif drawdown_pct >= self.drawdown_zones[DrawdownZone.ALERT.value]:
            return DrawdownZone.ALERT
        elif drawdown_pct >= self.drawdown_zones[DrawdownZone.CAUTION.value]:
            return DrawdownZone.CAUTION
        else:
            return DrawdownZone.NORMAL

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak.

        Returns:
            Current drawdown as decimal (e.g., 0.10 = 10% drawdown)
        """
        if self.peak_capital == 0:
            return 0.0

        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return max(0.0, drawdown)

    def get_portfolio_heat(self) -> float:
        """Calculate total portfolio heat (risk exposure).

        Portfolio heat is the sum of all position risk amounts divided by capital.

        Returns:
            Portfolio heat as decimal (e.g., 0.03 = 3% of capital at risk)
        """
        if not self.open_positions:
            return 0.0

        total_risk = sum(pos.risk_amount for pos in self.open_positions.values())
        return total_risk / self.current_capital if self.current_capital > 0 else 0.0

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int = 1,
        stop_pct: Optional[float] = None,
    ) -> float:
        """Calculate stop loss price.

        Args:
            entry_price: Entry price for position
            direction: Trade direction (1 for long, -1 for short)
            stop_pct: Stop loss percentage (uses hard_stop_pct if None)

        Returns:
            Stop loss price

        Raises:
            ValueError: If inputs are invalid
        """
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        if direction not in [1, -1]:
            raise ValueError(f"Direction must be 1 (long) or -1 (short), got {direction}")

        if stop_pct is None:
            stop_pct = self.hard_stop_pct

        if not 0 < stop_pct < 1:
            raise ValueError(f"Stop percentage must be between 0 and 1, got {stop_pct}")

        # For long positions, stop is below entry
        # For short positions, stop is above entry
        stop_loss = entry_price * (1 - direction * stop_pct)

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int = 1,
        profit_pct: Optional[float] = None,
    ) -> float:
        """Calculate take profit price.

        Args:
            entry_price: Entry price for position
            direction: Trade direction (1 for long, -1 for short)
            profit_pct: Take profit percentage (uses take_profit_pct if None)

        Returns:
            Take profit price

        Raises:
            ValueError: If inputs are invalid
        """
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        if direction not in [1, -1]:
            raise ValueError(f"Direction must be 1 (long) or -1 (short), got {direction}")

        if profit_pct is None:
            profit_pct = self.take_profit_pct

        if not 0 < profit_pct < 1:
            raise ValueError(f"Profit percentage must be between 0 and 1, got {profit_pct}")

        # For long positions, take profit is above entry
        # For short positions, take profit is below entry
        take_profit = entry_price * (1 + direction * profit_pct)

        return take_profit

    def validate_trade(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        confidence: Optional[float] = None,
        min_confidence: float = 0.55,
    ) -> Tuple[bool, List[str]]:
        """Validate if trade meets all risk management criteria.

        7-point validation checklist:
        1. Sufficient balance
        2. Position limit check
        3. Confidence threshold
        4. Drawdown zone check
        5. Portfolio heat check
        6. Daily loss limit check
        7. Valid prices

        Args:
            symbol: Trading symbol
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Model confidence (if applicable)
            min_confidence: Minimum required confidence

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # 1. Sufficient balance check
        position_value = quantity * entry_price
        if position_value > self.current_capital:
            issues.append(
                f"Insufficient capital: need {position_value:.2f}, "
                f"have {self.current_capital:.2f}"
            )

        # 2. Position limit check (max 25% of capital per position)
        max_position_value = self.current_capital * 0.25
        if position_value > max_position_value:
            issues.append(
                f"Position too large: {position_value:.2f} exceeds "
                f"max {max_position_value:.2f} (25% of capital)"
            )

        # 3. Confidence threshold check
        if confidence is not None:
            if confidence < min_confidence:
                issues.append(
                    f"Confidence {confidence:.2%} below minimum {min_confidence:.2%}"
                )

        # 4. Drawdown zone check
        zone = self.get_drawdown_zone()
        if zone == DrawdownZone.EMERGENCY:
            issues.append(
                f"Trading halted: in EMERGENCY drawdown zone "
                f"({self.get_current_drawdown():.1%})"
            )
        elif zone == DrawdownZone.ALERT:
            # Reduce position sizing in alert zone (checked elsewhere)
            logger.warning(f"In ALERT drawdown zone ({self.get_current_drawdown():.1%})")

        # 5. Portfolio heat check
        risk_amount = abs(entry_price - stop_loss) * quantity
        new_heat = (self.get_portfolio_heat() * self.current_capital + risk_amount) / self.current_capital
        if new_heat > self.max_portfolio_heat:
            issues.append(
                f"Exceeds portfolio heat limit: {new_heat:.2%} > {self.max_portfolio_heat:.2%}"
            )

        # 6. Daily loss limit check
        if self.daily_start_capital is not None:
            daily_loss = self.daily_start_capital - self.current_capital
            daily_loss_pct = daily_loss / self.daily_start_capital if self.daily_start_capital > 0 else 0
            if daily_loss_pct >= self.daily_loss_limit_pct:
                issues.append(
                    f"Daily loss limit reached: {daily_loss_pct:.2%} >= "
                    f"{self.daily_loss_limit_pct:.2%}"
                )

        # 7. Valid prices check
        if entry_price <= 0 or stop_loss <= 0:
            issues.append(f"Invalid prices: entry={entry_price}, stop={stop_loss}")

        if quantity <= 0:
            issues.append(f"Invalid quantity: {quantity}")

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Trade validation failed for {symbol}: {issues}")

        return is_valid, issues

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        entry_date: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Open a new position.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position size
            entry_date: Entry date
            stop_loss: Stop loss price (calculated if None)
            take_profit: Take profit price (calculated if None)
            metadata: Additional position metadata

        Returns:
            True if position opened successfully

        Raises:
            ValueError: If position already exists for symbol
        """
        if symbol in self.open_positions:
            raise ValueError(f"Position already exists for {symbol}")

        # Calculate stops if not provided
        if stop_loss is None:
            stop_loss = self.calculate_stop_loss(entry_price, direction=1)

        if take_profit is None:
            take_profit = self.calculate_take_profit(entry_price, direction=1)

        # Create position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_date=entry_date,
            stop_loss=stop_loss,
            trailing_stop_pct=self.trailing_stop_pct,
            take_profit=take_profit,
            time_stop_days=self.time_stop_days,
            metadata=metadata or {},
        )

        self.open_positions[symbol] = position

        logger.info(
            f"Opened position: {symbol} @ {entry_price:.2f}, qty={quantity:.2f}, "
            f"stop={stop_loss:.2f}, target={take_profit:.2f}"
        )

        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: datetime,
        reason: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        """Close an existing position.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_date: Exit date
            reason: Exit reason

        Returns:
            Trade record dictionary, or None if position not found
        """
        if symbol not in self.open_positions:
            logger.warning(f"No open position found for {symbol}")
            return None

        position = self.open_positions.pop(symbol)

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price - position.entry_price) / position.entry_price

        # Update capital
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.equity_curve.append(self.current_capital)

        # Record trade
        trade_record = {
            "symbol": symbol,
            "entry_date": position.entry_date,
            "exit_date": exit_date,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "days_held": (exit_date - position.entry_date).days,
            "metadata": position.metadata,
        }

        self.trade_history.append(trade_record)

        logger.info(
            f"Closed position: {symbol} @ {exit_price:.2f}, "
            f"P&L={pnl:.2f} ({pnl_pct:+.2%}), reason={reason}"
        )

        return trade_record

    def update_positions(
        self,
        current_prices: Dict[str, float],
        current_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Update all open positions and check for exits.

        Args:
            current_prices: Dictionary of symbol -> current_price
            current_date: Current date

        Returns:
            List of closed trade records
        """
        closed_trades = []

        for symbol in list(self.open_positions.keys()):
            if symbol not in current_prices:
                logger.warning(f"No price data for {symbol}, skipping update")
                continue

            position = self.open_positions[symbol]
            current_price = current_prices[symbol]

            # Update trailing stop
            position.update_trailing_stop(current_price)

            # Check exit conditions
            should_exit, reason = position.should_exit(current_price, current_date)

            if should_exit:
                trade_record = self.close_position(
                    symbol=symbol,
                    exit_price=current_price,
                    exit_date=current_date,
                    reason=reason,
                )
                if trade_record:
                    closed_trades.append(trade_record)

        return closed_trades

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics (call at start of each trading day)."""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0.0

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        returns: Optional[pd.Series] = None,
    ) -> float:
        """Calculate Value at Risk (VaR).

        Args:
            confidence_level: Confidence level (default: 0.95 for 95% VaR)
            returns: Return series (uses trade history if None)

        Returns:
            VaR as positive value (potential loss amount)
        """
        if returns is None:
            if not self.trade_history:
                return 0.0
            returns = pd.Series([t["pnl_pct"] for t in self.trade_history])

        if len(returns) == 0:
            return 0.0

        # VaR is the negative of the (1 - confidence_level) percentile
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)

        # Return as positive value (potential loss)
        var = abs(min(0, var_percentile)) * self.current_capital

        return var

    def calculate_cvar(
        self,
        confidence_level: float = 0.95,
        returns: Optional[pd.Series] = None,
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that loss exceeds VaR.

        Args:
            confidence_level: Confidence level (default: 0.95)
            returns: Return series (uses trade history if None)

        Returns:
            CVaR as positive value (expected loss amount)
        """
        if returns is None:
            if not self.trade_history:
                return 0.0
            returns = pd.Series([t["pnl_pct"] for t in self.trade_history])

        if len(returns) == 0:
            return 0.0

        # Find VaR threshold
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)

        # CVaR is the mean of all returns below VaR threshold
        tail_returns = returns[returns <= var_percentile]

        if len(tail_returns) == 0:
            return 0.0

        cvar = abs(tail_returns.mean()) * self.current_capital

        return cvar

    def calculate_max_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown from equity curve.

        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_duration_days)
        """
        if len(self.equity_curve) < 2:
            return 0.0, 0

        equity = pd.Series(self.equity_curve)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        max_dd = abs(drawdown.min())

        # Calculate duration
        drawdown_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, drawdown_duration

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        max_dd, max_dd_duration = self.calculate_max_drawdown()
        current_dd = self.get_current_drawdown()
        portfolio_heat = self.get_portfolio_heat()
        var_95 = self.calculate_var(confidence_level=0.95)
        cvar_95 = self.calculate_cvar(confidence_level=0.95)

        metrics = {
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "drawdown_zone": self.get_drawdown_zone().value,
            "portfolio_heat": portfolio_heat,
            "open_positions": len(self.open_positions),
            "total_trades": len(self.trade_history),
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

        return metrics

    def get_summary(self) -> str:
        """Get formatted risk management summary.

        Returns:
            Formatted summary string
        """
        metrics = self.get_risk_metrics()

        summary = f"""
Risk Management Summary
{'=' * 50}
Capital:
  Current:     ${metrics['current_capital']:,.2f}
  Peak:        ${metrics['peak_capital']:,.2f}
  Initial:     ${self.initial_capital:,.2f}
  Return:      {(metrics['current_capital'] / self.initial_capital - 1):+.2%}

Drawdown:
  Current:     {metrics['current_drawdown']:.2%}
  Maximum:     {metrics['max_drawdown']:.2%}
  Max Duration: {metrics['max_drawdown_duration']} periods
  Zone:        {metrics['drawdown_zone'].upper()}

Risk Exposure:
  Portfolio Heat: {metrics['portfolio_heat']:.2%} (max: {self.max_portfolio_heat:.2%})
  Open Positions: {metrics['open_positions']}
  VaR (95%):      ${metrics['var_95']:,.2f}
  CVaR (95%):     ${metrics['cvar_95']:,.2f}

Trading:
  Total Trades: {metrics['total_trades']}
"""

        return summary
