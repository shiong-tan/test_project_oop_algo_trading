"""
Risk Management System Implementation for ML-Based Trading Strategy

This module provides concrete implementations of position sizing, stop-loss mechanics,
drawdown tracking, portfolio heat management, and risk metrics calculation.

To integrate into backtesting engine:
1. Import these classes into your BacktestingBase class
2. Create instances in __init__
3. Call validation methods before placing trades
4. Update metrics after each bar
5. Monitor for rule violations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


# ============================================================================
# 1. POSITION SIZING CLASSES
# ============================================================================

class PositionSizer:
    """Base class for position sizing strategies"""

    def __init__(self, account_value: float):
        self.account_value = account_value

    def calculate_position_size(self, **kwargs) -> int:
        """Override in subclass"""
        raise NotImplementedError


class FixedFractionalSizer(PositionSizer):
    """
    Fixed Fractional Position Sizing

    Risk per trade: 1-2% of account
    Position size determined by: (Account Risk) / (Risk Per Unit)
    """

    def __init__(self, account_value: float, risk_per_trade_pct: float = 0.01):
        super().__init__(account_value)
        self.risk_per_trade_pct = risk_per_trade_pct  # Default 1%

    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss_price: float,
                                account_value: Optional[float] = None) -> int:
        """
        Calculate position size for fixed fractional sizing

        Args:
            entry_price: Price at which to enter
            stop_loss_price: Hard stop-loss price
            account_value: Optional - use if account value has changed

        Returns:
            Number of units to trade
        """
        if account_value is None:
            account_value = self.account_value

        risk_per_unit = entry_price - stop_loss_price
        account_risk = account_value * self.risk_per_trade_pct

        if risk_per_unit <= 0:
            return 0

        units = account_risk / risk_per_unit
        return int(units)

    def get_risk_amount(self, account_value: Optional[float] = None) -> float:
        """Get dollar amount at risk per trade"""
        if account_value is None:
            account_value = self.account_value
        return account_value * self.risk_per_trade_pct


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion Position Sizing

    f* = (p * b - q) / b
    where:
        p = probability of win
        q = probability of loss
        b = ratio of win size to loss size
        f* = optimal fraction to wager

    Applies safety factor (25-50%) to reduce volatility
    """

    def __init__(self, account_value: float, safety_factor: float = 0.25):
        super().__init__(account_value)
        self.safety_factor = safety_factor  # 25% default
        self.min_trades_required = 50

    def calculate_optimal_fraction(self,
                                   win_rate: float,
                                   avg_win: float,
                                   avg_loss: float) -> float:
        """
        Calculate optimal Kelly fraction

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount in dollars
            avg_loss: Average loss amount in dollars (positive value)

        Returns:
            Optimal fraction to risk (before safety factor)
        """
        if avg_loss <= 0:
            return 0

        loss_rate = 1 - win_rate
        b = avg_win / avg_loss

        if b == 0:
            return 0

        f_star = (win_rate * b - loss_rate) / b

        # Ensure between 0 and 1
        f_star = max(0, min(1, f_star))

        return f_star

    def calculate_position_size(self,
                                win_rate: float,
                                avg_win: float,
                                avg_loss: float,
                                entry_price: float,
                                num_trades: int = 0) -> float:
        """
        Calculate position size using Kelly with safety factor

        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            entry_price: Current entry price
            num_trades: Number of completed trades (to validate sufficiency)

        Returns:
            Fraction of capital to deploy
        """
        # Don't use Kelly until sufficient trade history
        if num_trades < self.min_trades_required:
            return 0

        f_star = self.calculate_optimal_fraction(win_rate, avg_win, avg_loss)
        adjusted_f = f_star * self.safety_factor

        return adjusted_f


class ConfidenceBasedSizer(PositionSizer):
    """
    Confidence-Based Position Sizing

    Adjusts position size based on model prediction confidence

    Confidence Tiers:
    - 50-55%: 0.25x (weak signal)
    - 55-60%: 0.50x (moderate signal)
    - 60-70%: 0.75x (strong signal)
    - 70-80%: 1.00x (very strong signal)
    - >80%:   0.50x (potential overfit, reduce)
    """

    def __init__(self, base_sizer: PositionSizer):
        super().__init__(base_sizer.account_value)
        self.base_sizer = base_sizer

    def get_confidence_multiplier(self, confidence: float) -> float:
        """
        Get position size multiplier based on confidence level

        Args:
            confidence: Model prediction probability (0-1)

        Returns:
            Multiplier to apply to base position size (0-1)
        """
        if confidence <= 0.55:
            return 0.25
        elif confidence <= 0.60:
            return 0.50
        elif confidence <= 0.70:
            return 0.75
        elif confidence <= 0.80:
            return 1.00
        else:  # > 0.80 - suspicious overfit
            return 0.50

    def should_skip_trade(self, confidence: float) -> Tuple[bool, str]:
        """Check if trade should be skipped due to confidence"""
        if confidence < 0.52:
            return True, "Confidence below 52% minimum"
        if confidence > 0.80:
            return False, "High confidence but suspicious (>80%)"
        return False, "Confidence OK"

    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss_price: float,
                                confidence: float,
                                account_value: Optional[float] = None) -> int:
        """
        Calculate position size with confidence adjustment

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Model prediction confidence (0-1)
            account_value: Optional updated account value

        Returns:
            Number of units to trade
        """
        # Base position size
        base_units = self.base_sizer.calculate_position_size(
            entry_price, stop_loss_price, account_value
        )

        # Confidence adjustment
        multiplier = self.get_confidence_multiplier(confidence)
        adjusted_units = int(base_units * multiplier)

        return adjusted_units


# ============================================================================
# 2. STOP-LOSS AND TAKE-PROFIT CLASSES
# ============================================================================

class StopLossManager:
    """Manages stop-loss and take-profit levels for trades"""

    def __init__(self,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.05,
                 min_risk_reward_ratio: float = 2.0,
                 trailing_stop_pct: float = 0.02):
        """
        Args:
            stop_loss_pct: Default stop-loss distance (3% = 0.03)
            take_profit_pct: Default take-profit distance (5% = 0.05)
            min_risk_reward_ratio: Minimum R:R ratio required (1:2 = 2.0)
            trailing_stop_pct: Trailing stop distance once profitable
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.trailing_stop_pct = trailing_stop_pct

    def calculate_stop_and_target(self,
                                  entry_price: float
                                  ) -> Dict[str, float]:
        """
        Calculate stop-loss and take-profit levels

        Args:
            entry_price: Entry price

        Returns:
            Dictionary with 'stop_loss' and 'take_profit' prices
        """
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)

        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def validate_risk_reward(self,
                            entry_price: float,
                            stop_loss_price: float,
                            take_profit_price: float) -> Tuple[bool, float, str]:
        """
        Validate that trade meets minimum risk/reward ratio

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            take_profit_price: Take-profit price

        Returns:
            (is_valid, risk_reward_ratio, message)
        """
        risk = entry_price - stop_loss_price
        reward = take_profit_price - entry_price

        if risk <= 0:
            return False, 0, "Invalid stop-loss (above entry)"

        ratio = reward / risk

        if ratio < self.min_risk_reward_ratio:
            return False, ratio, \
                f"Risk/Reward {ratio:.2f} below minimum {self.min_risk_reward_ratio}"

        return True, ratio, f"Risk/Reward OK: {ratio:.2f}"

    def get_trailing_stop(self,
                         entry_price: float,
                         current_price: float,
                         profit_threshold: float = 0.02) -> Optional[float]:
        """
        Calculate trailing stop once trade is profitable

        Args:
            entry_price: Entry price
            current_price: Current price
            profit_threshold: Minimum profit before activating trail (2% = 0.02)

        Returns:
            Trailing stop price, or None if not yet profitable enough
        """
        profit_pct = (current_price - entry_price) / entry_price

        # Activate only if profitable
        if profit_pct < profit_threshold:
            return None

        trailing_stop = current_price * (1 - self.trailing_stop_pct)
        return trailing_stop


class TradeExecution:
    """Represents a single trade with all relevant data"""

    def __init__(self,
                 trade_id: int,
                 symbol: str,
                 entry_price: float,
                 units: int,
                 entry_date: datetime,
                 stop_loss: float,
                 take_profit: float,
                 confidence: float):
        """
        Args:
            trade_id: Unique trade identifier
            symbol: Stock symbol
            entry_price: Entry price
            units: Number of units
            entry_date: Entry datetime
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            confidence: Model confidence (0-1)
        """
        self.trade_id = trade_id
        self.symbol = symbol
        self.entry_price = entry_price
        self.units = units
        self.entry_date = entry_date
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence

        self.risk_per_unit = entry_price - stop_loss
        self.one_r = self.risk_per_unit * units
        self.risk_reward_ratio = (take_profit - entry_price) / self.risk_per_unit

        self.exit_price: Optional[float] = None
        self.exit_date: Optional[datetime] = None
        self.exit_type: Optional[str] = None  # 'stop', 'target', 'time', 'manual'
        self.pnl: Optional[float] = None
        self.r_multiple: Optional[float] = None

    def exit_trade(self,
                   exit_price: float,
                   exit_date: datetime,
                   exit_type: str):
        """
        Close the trade

        Args:
            exit_price: Exit price
            exit_date: Exit datetime
            exit_type: Type of exit ('stop', 'target', 'time', 'manual')
        """
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_type = exit_type

        pnl = (exit_price - self.entry_price) * self.units
        self.pnl = pnl

        # Calculate R-multiple
        self.r_multiple = pnl / self.one_r

    def days_open(self) -> int:
        """Get number of trading days trade has been open"""
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        return 0

    def to_dict(self) -> Dict:
        """Convert trade to dictionary for reporting"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'units': self.units,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'exit_type': self.exit_type,
            'pnl': self.pnl,
            'r_multiple': self.r_multiple,
            'days_open': self.days_open(),
            'confidence': self.confidence
        }


# ============================================================================
# 3. DRAWDOWN AND PORTFOLIO HEAT MANAGEMENT
# ============================================================================

class DrawdownTracker:
    """Tracks drawdown and enforces maximum drawdown limits"""

    def __init__(self, initial_capital: float, max_drawdown_pct: float = 0.10):
        """
        Args:
            initial_capital: Starting capital
            max_drawdown_pct: Maximum allowed drawdown (10% = 0.10)
        """
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.max_drawdown = 0
        self.max_allowed_dd = max_drawdown_pct
        self.max_allowed_dd_amount = initial_capital * max_drawdown_pct

        self.drawdown_history = []

    def update_equity(self, new_equity: float) -> Tuple[bool, Dict]:
        """
        Update current equity and check drawdown limits

        Args:
            new_equity: Current account equity

        Returns:
            (is_within_limits, status_dict)
        """
        self.current_equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        # Calculate drawdowns
        current_dd = (self.peak_equity - new_equity) / self.peak_equity
        current_dd_amount = self.peak_equity - new_equity

        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'equity': new_equity,
            'peak': self.peak_equity,
            'dd_pct': current_dd,
            'dd_amount': current_dd_amount
        })

        # Check limit
        within_limit = current_dd_amount <= self.max_allowed_dd_amount

        status = {
            'current_dd_pct': current_dd,
            'current_dd_amount': current_dd_amount,
            'max_dd_pct': self.max_drawdown,
            'max_allowed_pct': self.max_allowed_dd,
            'within_limits': within_limit,
            'zone': self.get_drawdown_zone(current_dd)
        }

        return within_limit, status

    def get_drawdown_zone(self, dd_pct: float) -> str:
        """Determine drawdown management zone"""
        if dd_pct <= 0.05:
            return "NORMAL (0-5%)"
        elif dd_pct <= 0.075:
            return "CAUTION (5-7.5%)"
        elif dd_pct <= 0.10:
            return "ALERT (7.5-10%)"
        else:
            return "EMERGENCY (>10%)"

    def get_position_size_multiplier(self) -> float:
        """Get position sizing multiplier based on drawdown zone"""
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity

        if current_dd <= 0.05:
            return 1.0  # Normal
        elif current_dd <= 0.075:
            return 0.5  # Caution: 50% position sizes
        elif current_dd <= 0.10:
            return 0.25  # Alert: 25% position sizes
        else:
            return 0.0  # Emergency: no new trades

    def get_stats(self) -> Dict:
        """Get drawdown statistics"""
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_dd_pct': (self.peak_equity - self.current_equity) / self.peak_equity,
            'current_dd_amount': self.peak_equity - self.current_equity,
            'max_dd_pct': self.max_drawdown,
            'max_allowed_pct': self.max_allowed_dd,
            'within_limits': ((self.peak_equity - self.current_equity) <= self.max_allowed_dd_amount)
        }


class PortfolioHeat:
    """Manages total portfolio risk exposure across all open positions"""

    def __init__(self, account_value: float, max_heat_pct: float = 0.03):
        """
        Args:
            account_value: Current account value
            max_heat_pct: Maximum heat as % of account (3% = 0.03)
        """
        self.account_value = account_value
        self.max_heat_pct = max_heat_pct
        self.max_heat_amount = account_value * max_heat_pct
        self.positions = {}  # symbol: {'units': X, 'entry_price': Y, 'risk': Z}
        self.total_heat = 0

    def can_add_position(self,
                        symbol: str,
                        units: int,
                        risk_per_unit: float) -> Tuple[bool, str]:
        """
        Check if new position would exceed heat limits

        Args:
            symbol: Stock symbol
            units: Number of units
            risk_per_unit: Risk per unit (entry - stop)

        Returns:
            (can_add, message)
        """
        potential_heat = units * risk_per_unit
        new_total_heat = self.total_heat + potential_heat

        if new_total_heat > self.max_heat_amount:
            available_heat = self.max_heat_amount - self.total_heat
            max_units = int(available_heat / risk_per_unit) if risk_per_unit > 0 else 0
            return False, \
                f"Heat limit exceeded. Current {self.total_heat:.2f}, " \
                f"Max {self.max_heat_amount:.2f}. Can add max {max_units} units"

        return True, f"Position OK. Total heat after: {new_total_heat:.2f}"

    def add_position(self,
                    symbol: str,
                    units: int,
                    entry_price: float,
                    risk_per_unit: float) -> Dict:
        """
        Add position to heat tracking

        Args:
            symbol: Stock symbol
            units: Number of units
            entry_price: Entry price
            risk_per_unit: Risk per unit

        Returns:
            Heat status dictionary
        """
        heat = units * risk_per_unit
        self.positions[symbol] = {
            'units': units,
            'entry_price': entry_price,
            'risk': heat
        }
        self.total_heat += heat

        return {
            'position_heat': heat,
            'total_heat': self.total_heat,
            'heat_pct': self.total_heat / self.account_value,
            'remaining_heat': self.max_heat_amount - self.total_heat,
            'positions_count': len(self.positions)
        }

    def close_position(self, symbol: str) -> Optional[Dict]:
        """
        Remove position from heat tracking

        Args:
            symbol: Stock symbol

        Returns:
            Updated heat status or None if symbol not found
        """
        if symbol not in self.positions:
            return None

        self.total_heat -= self.positions[symbol]['risk']
        del self.positions[symbol]

        return {
            'total_heat': self.total_heat,
            'heat_pct': self.total_heat / self.account_value,
            'remaining_heat': self.max_heat_amount - self.total_heat,
            'positions_count': len(self.positions)
        }

    def update_account_value(self, new_account_value: float):
        """Update account value (adjust max heat if account grows/shrinks)"""
        self.account_value = new_account_value
        self.max_heat_amount = new_account_value * self.max_heat_pct

    def get_stats(self) -> Dict:
        """Get current heat statistics"""
        return {
            'total_heat': self.total_heat,
            'max_heat': self.max_heat_amount,
            'heat_pct': self.total_heat / self.account_value,
            'remaining_heat': self.max_heat_amount - self.total_heat,
            'positions': self.positions,
            'num_positions': len(self.positions)
        }


# ============================================================================
# 4. RISK METRICS CALCULATION
# ============================================================================

class RiskMetrics:
    """Calculate risk metrics for portfolio performance analysis"""

    @staticmethod
    def calculate_var_95(daily_returns: np.ndarray, account_value: float) -> float:
        """
        Calculate Value at Risk at 95% confidence

        Args:
            daily_returns: Array of daily returns (as decimals, e.g., 0.02 = 2%)
            account_value: Current account value

        Returns:
            VaR in dollars (negative number representing max loss)
        """
        var_pct = np.percentile(daily_returns, 5)  # 5th percentile
        return var_pct * account_value

    @staticmethod
    def calculate_cvar_95(daily_returns: np.ndarray, account_value: float) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall) at 95% confidence

        Args:
            daily_returns: Array of daily returns
            account_value: Current account value

        Returns:
            CVaR in dollars
        """
        cutoff = np.percentile(daily_returns, 5)
        cvar_pct = daily_returns[daily_returns <= cutoff].mean()
        return cvar_pct * account_value

    @staticmethod
    def calculate_sharpe_ratio(daily_returns: np.ndarray,
                               risk_free_rate: float = 0.0005) -> float:
        """
        Calculate Sharpe Ratio

        Args:
            daily_returns: Array of daily returns
            risk_free_rate: Daily risk-free rate (0.0005 = ~5% annually)

        Returns:
            Annualized Sharpe ratio
        """
        excess_return = daily_returns.mean() - risk_free_rate
        volatility = daily_returns.std()

        if volatility == 0:
            return 0

        return (excess_return / volatility) * np.sqrt(252)

    @staticmethod
    def calculate_sortino_ratio(daily_returns: np.ndarray,
                                risk_free_rate: float = 0.0005) -> float:
        """
        Calculate Sortino Ratio (downside risk only)

        Args:
            daily_returns: Array of daily returns
            risk_free_rate: Daily risk-free rate

        Returns:
            Annualized Sortino ratio
        """
        excess_return = daily_returns.mean() - risk_free_rate
        downside_returns = daily_returns[daily_returns < 0]

        if len(downside_returns) == 0:
            return 0

        downside_volatility = downside_returns.std()

        if downside_volatility == 0:
            return 0

        return (excess_return / downside_volatility) * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve

        Args:
            equity_curve: Array of account equity values over time

        Returns:
            Maximum drawdown as decimal (e.g., 0.15 = 15%)
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max
        return np.max(drawdown)

    @staticmethod
    def calculate_r_expectancy(r_multiples: List[float]) -> Dict:
        """
        Calculate R-multiple expectancy and related metrics

        Args:
            r_multiples: List of R-multiple outcomes from trades

        Returns:
            Dictionary with expectancy and related stats
        """
        if not r_multiples:
            return {
                'expectancy': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_trades': 0
            }

        r_multiples = np.array(r_multiples)
        wins = r_multiples[r_multiples > 0]
        losses = r_multiples[r_multiples <= 0]

        win_rate = len(wins) / len(r_multiples) if len(r_multiples) > 0 else 0
        loss_rate = len(losses) / len(r_multiples) if len(r_multiples) > 0 else 0

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

        profit_factor = 0
        if len(losses) > 0 and losses.sum() != 0:
            profit_factor = abs(wins.sum() / losses.sum())

        return {
            'expectancy': expectancy,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(r_multiples),
            'winners': len(wins),
            'losers': len(losses)
        }


# ============================================================================
# 5. TRADE VALIDATION AND RISK RULES
# ============================================================================

class TradeValidator:
    """Validates trades against risk rules before execution"""

    def __init__(self,
                 min_confidence: float = 0.52,
                 max_confidence: float = 0.70,
                 min_risk_reward: float = 2.0,
                 max_position_pct: float = 0.05,
                 max_heat_pct: float = 0.03):
        """
        Args:
            min_confidence: Minimum model confidence threshold
            max_confidence: Maximum confidence before flagging overfit
            min_risk_reward: Minimum risk/reward ratio
            max_position_pct: Maximum position as % of account
            max_heat_pct: Maximum portfolio heat as % of account
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.max_heat_pct = max_heat_pct

    def validate_trade(self,
                      entry_price: float,
                      stop_loss_price: float,
                      take_profit_price: float,
                      units: int,
                      confidence: float,
                      account_value: float,
                      current_equity: float,
                      current_heat: float,
                      drawdown_zone: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive pre-trade validation

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            take_profit_price: Take-profit price
            units: Proposed position size
            confidence: Model confidence
            account_value: Account value
            current_equity: Current equity
            current_heat: Current portfolio heat
            drawdown_zone: Current drawdown zone

        Returns:
            (is_valid, list_of_rejection_reasons)
        """
        rejections = []

        # 1. Signal Quality Check
        if confidence < self.min_confidence:
            rejections.append(f"Confidence {confidence:.2%} below minimum {self.min_confidence:.2%}")

        if confidence > self.max_confidence:
            rejections.append(f"Confidence {confidence:.2%} above max (potential overfit)")

        # 2. Position Sizing Check
        position_value = units * entry_price
        position_pct = position_value / account_value

        if position_pct > self.max_position_pct:
            rejections.append(
                f"Position {position_pct:.1%} exceeds max {self.max_position_pct:.1%}"
            )

        # 3. Risk/Reward Check
        risk = entry_price - stop_loss_price
        reward = take_profit_price - entry_price

        if risk <= 0:
            rejections.append("Stop-loss above entry price (invalid)")

        elif (reward / risk) < self.min_risk_reward:
            ratio = reward / risk
            rejections.append(
                f"Risk/Reward {ratio:.2f} below minimum {self.min_risk_reward}"
            )

        # 4. Portfolio Heat Check
        new_heat = current_heat + (units * risk)
        max_heat = account_value * self.max_heat_pct

        if new_heat > max_heat:
            rejections.append(
                f"Heat {new_heat:.2f} would exceed limit {max_heat:.2f}"
            )

        # 5. Drawdown Zone Check
        if "EMERGENCY" in drawdown_zone:
            rejections.append(f"Emergency mode: {drawdown_zone}")

        return len(rejections) == 0, rejections


# ============================================================================
# 6. INTEGRATION EXAMPLE WITH BACKTESTER
# ============================================================================

class RiskManagedBacktester:
    """
    Example of integrating risk management into backtesting engine

    This shows how to use all the risk management components together
    """

    def __init__(self,
                 initial_capital: float = 10000,
                 risk_per_trade_pct: float = 0.01,
                 max_drawdown_pct: float = 0.10):
        """
        Initialize risk-managed backtester

        Args:
            initial_capital: Starting capital
            risk_per_trade_pct: Risk per trade as % of account
            max_drawdown_pct: Maximum drawdown limit
        """
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.units_held = 0
        self.trades = []
        self.equity_curve = [initial_capital]
        self.daily_returns = []

        # Risk management components
        self.position_sizer = FixedFractionalSizer(
            initial_capital, risk_per_trade_pct
        )
        self.confidence_sizer = ConfidenceBasedSizer(self.position_sizer)
        self.stop_loss_mgr = StopLossManager()
        self.drawdown_tracker = DrawdownTracker(initial_capital, max_drawdown_pct)
        self.portfolio_heat = PortfolioHeat(initial_capital, max_heat_pct=0.03)
        self.validator = TradeValidator()

        self.trade_id_counter = 0

    def attempt_trade(self,
                     symbol: str,
                     entry_price: float,
                     current_date: datetime,
                     model_signal: int,
                     confidence: float) -> Tuple[bool, str, Optional[TradeExecution]]:
        """
        Attempt to execute a trade with full risk validation

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            current_date: Current date
            model_signal: 1 for buy, 0 for sell
            confidence: Model confidence

        Returns:
            (trade_executed, message, trade_object_or_none)
        """
        # Calculate stop-loss and take-profit
        levels = self.stop_loss_mgr.calculate_stop_and_target(entry_price)
        is_valid, rr_msg = self.stop_loss_mgr.validate_risk_reward(
            entry_price, levels['stop_loss'], levels['take_profit']
        )

        if not is_valid:
            return False, f"Risk/Reward validation failed: {rr_msg}", None

        # Calculate position size with confidence adjustment
        units = self.confidence_sizer.calculate_position_size(
            entry_price,
            levels['stop_loss'],
            confidence,
            self.current_balance
        )

        if units == 0:
            return False, "Position size calculated to zero units", None

        # Get current drawdown zone
        drawdown_zone = self.drawdown_tracker.get_drawdown_zone(
            (self.drawdown_tracker.peak_equity - self.current_balance) /
            self.drawdown_tracker.peak_equity
        )

        # Comprehensive validation
        is_valid, rejections = self.validator.validate_trade(
            entry_price=entry_price,
            stop_loss_price=levels['stop_loss'],
            take_profit_price=levels['take_profit'],
            units=units,
            confidence=confidence,
            account_value=self.initial_capital,
            current_equity=self.current_balance,
            current_heat=self.portfolio_heat.total_heat,
            drawdown_zone=drawdown_zone
        )

        if not is_valid:
            return False, f"Validation failed: {'; '.join(rejections)}", None

        # Check portfolio heat
        can_add, heat_msg = self.portfolio_heat.can_add_position(
            symbol, units,
            entry_price - levels['stop_loss']
        )

        if not can_add:
            return False, f"Portfolio heat check failed: {heat_msg}", None

        # Create trade
        self.trade_id_counter += 1
        trade = TradeExecution(
            trade_id=self.trade_id_counter,
            symbol=symbol,
            entry_price=entry_price,
            units=units,
            entry_date=current_date,
            stop_loss=levels['stop_loss'],
            take_profit=levels['take_profit'],
            confidence=confidence
        )

        # Execute trade
        self.units_held = units
        trade_cost = units * entry_price
        self.current_balance -= trade_cost

        # Add to tracking
        self.trades.append(trade)
        self.portfolio_heat.add_position(
            symbol, units, entry_price,
            entry_price - levels['stop_loss']
        )

        return True, f"Trade executed: {units} units at {entry_price:.2f}", trade

    def close_trade(self,
                   exit_price: float,
                   exit_date: datetime,
                   exit_type: str = 'manual'):
        """
        Close current trade

        Args:
            exit_price: Exit price
            exit_date: Exit date
            exit_type: Type of exit
        """
        if not self.trades or not self.units_held:
            return

        trade = self.trades[-1]
        trade.exit_trade(exit_price, exit_date, exit_type)

        # Update balance
        proceeds = self.units_held * exit_price
        self.current_balance += proceeds
        self.units_held = 0

        # Update heat
        self.portfolio_heat.close_position(trade.symbol)

        # Update equity curve
        self.equity_curve.append(self.current_balance)

        # Calculate daily return
        if len(self.equity_curve) > 1:
            daily_return = (self.current_balance - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)

        # Update drawdown tracker
        self.drawdown_tracker.update_equity(self.current_balance)

    def get_performance_metrics(self) -> Dict:
        """Get overall performance metrics"""
        if not self.daily_returns:
            return {}

        daily_returns = np.array(self.daily_returns)
        equity_curve = np.array(self.equity_curve)

        r_multiples = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        expectancy = RiskMetrics.calculate_r_expectancy(r_multiples)

        return {
            'total_return_pct': (self.current_balance - self.initial_capital) / self.initial_capital * 100,
            'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(daily_returns),
            'sortino_ratio': RiskMetrics.calculate_sortino_ratio(daily_returns),
            'max_drawdown_pct': RiskMetrics.calculate_max_drawdown(equity_curve) * 100,
            'var_95': RiskMetrics.calculate_var_95(daily_returns, self.current_balance),
            'cvar_95': RiskMetrics.calculate_cvar_95(daily_returns, self.current_balance),
            'expectancy': expectancy,
            'num_trades': len(self.trades),
            'final_balance': self.current_balance
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a risk-managed backtester
    backtester = RiskManagedBacktester(
        initial_capital=10000,
        risk_per_trade_pct=0.01,  # 1% risk per trade
        max_drawdown_pct=0.10  # 10% max drawdown
    )

    # Example trade attempt
    success, message, trade = backtester.attempt_trade(
        symbol='AAPL',
        entry_price=150.00,
        current_date=datetime.now(),
        model_signal=1,
        confidence=0.65  # 65% confidence
    )

    print(f"Trade Execution: {success}")
    print(f"Message: {message}")

    if trade:
        print(f"Trade ID: {trade.trade_id}")
        print(f"Entry: {trade.entry_price}, Stop: {trade.stop_loss}, Target: {trade.take_profit}")
        print(f"Risk/Reward Ratio: {trade.risk_reward_ratio:.2f}")

        # Close trade
        backtester.close_trade(
            exit_price=154.50,
            exit_date=datetime.now(),
            exit_type='target'
        )

        print(f"Trade P&L: {trade.pnl:.2f}")
        print(f"Trade R-Multiple: {trade.r_multiple:.2f}R")

    # Get performance metrics
    metrics = backtester.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
