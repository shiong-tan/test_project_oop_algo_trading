"""Backtesting engine for algorithmic trading strategies.

This module implements an event-driven backtesting engine with:
- Integration with RiskManager
- Order execution with validation
- Portfolio tracking
- Trade logging
- Slippage modeling
- Realistic transaction costs
- Comprehensive error handling
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..risk.manager import RiskManager
from ..risk.position_sizing import PositionSizingStrategy

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")

        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")


@dataclass
class Fill:
    """Represents an order fill."""

    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        base_cost = self.quantity * self.price
        if self.side == OrderSide.BUY:
            return base_cost + self.commission + self.slippage
        else:
            return base_cost - self.commission - self.slippage


class BacktestEngine:
    """Event-driven backtesting engine.

    Features:
    - Event-driven architecture for realistic simulation
    - Integration with RiskManager for risk controls
    - Realistic order execution with slippage and commissions
    - Position tracking and portfolio management
    - Comprehensive trade logging
    - Performance metrics calculation
    """

    def __init__(
        self,
        initial_capital: float,
        risk_manager: Optional[RiskManager] = None,
        position_sizer: Optional[PositionSizingStrategy] = None,
        commission_pct: float = 0.0015,
        slippage_bps: float = 5.0,
        verbose: bool = False,
    ):
        """Initialize BacktestEngine.

        Args:
            initial_capital: Starting capital
            risk_manager: RiskManager instance (created if None)
            position_sizer: Position sizing strategy (optional)
            commission_pct: Commission as percentage of trade value (default: 0.15%)
            slippage_bps: Slippage in basis points (default: 5 bps)
            verbose: Enable verbose logging

        Raises:
            ValueError: If parameters are invalid
        """
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")

        if not 0 <= commission_pct < 1:
            raise ValueError(
                f"Commission must be between 0 and 1, got {commission_pct}"
            )

        if slippage_bps < 0:
            raise ValueError(f"Slippage must be non-negative, got {slippage_bps}")

        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps
        self.verbose = verbose

        # Create risk manager if not provided
        if risk_manager is None:
            risk_manager = RiskManager(initial_capital=initial_capital)
        self.risk_manager = risk_manager

        self.position_sizer = position_sizer

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.fills: List[Fill] = []

        # Current state
        self.current_date: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}

        logger.info(
            f"BacktestEngine initialized: capital=${initial_capital:,.2f}, "
            f"commission={commission_pct:.2%}, slippage={slippage_bps} bps"
        )

    @property
    def portfolio_value(self) -> float:
        """Calculate current portfolio value.

        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = sum(
            qty * self.current_prices.get(symbol, 0)
            for symbol, qty in self.positions.items()
        )
        return self.cash + positions_value

    def calculate_slippage(self, price: float, quantity: float, side: OrderSide) -> float:
        """Calculate slippage for an order.

        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (BUY/SELL)

        Returns:
            Slippage amount
        """
        # Slippage is typically against the trader
        # For buys: pay more, for sells: receive less
        slippage_pct = self.slippage_bps / 10000  # Convert bps to percentage

        base_value = price * quantity
        slippage = base_value * slippage_pct

        return slippage

    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate commission for an order.

        Args:
            price: Order price
            quantity: Order quantity

        Returns:
            Commission amount
        """
        trade_value = price * quantity
        commission = trade_value * self.commission_pct
        return commission

    def execute_order(
        self,
        order: Order,
        current_price: float,
        timestamp: datetime,
    ) -> Optional[Fill]:
        """Execute an order and return fill.

        Args:
            order: Order to execute
            current_price: Current market price
            timestamp: Execution timestamp

        Returns:
            Fill object if executed, None if execution failed

        Raises:
            ValueError: If order cannot be executed
        """
        # Validate order
        if order.quantity <= 0:
            raise ValueError(f"Invalid order quantity: {order.quantity}")

        # Determine execution price
        if order.order_type == OrderType.MARKET:
            execution_price = current_price
        elif order.order_type == OrderType.LIMIT:
            # Limit orders only fill if price is favorable
            if order.side == OrderSide.BUY and current_price > order.price:
                logger.debug(
                    f"Limit buy not filled: price {current_price} > limit {order.price}"
                )
                return None
            elif order.side == OrderSide.SELL and current_price < order.price:
                logger.debug(
                    f"Limit sell not filled: price {current_price} < limit {order.price}"
                )
                return None
            execution_price = order.price
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        # Calculate costs
        commission = self.calculate_commission(execution_price, order.quantity)
        slippage = self.calculate_slippage(execution_price, order.quantity, order.side)

        # Calculate total cost
        if order.side == OrderSide.BUY:
            total_cost = (execution_price * order.quantity) + commission + slippage
        else:
            total_cost = (execution_price * order.quantity) - commission - slippage

        # Check if sufficient cash for buys
        if order.side == OrderSide.BUY:
            if total_cost > self.cash:
                logger.warning(
                    f"Insufficient cash for order: need ${total_cost:.2f}, "
                    f"have ${self.cash:.2f}"
                )
                return None

        # Check if sufficient position for sells
        if order.side == OrderSide.SELL:
            current_position = self.positions.get(order.symbol, 0)
            if order.quantity > current_position:
                logger.warning(
                    f"Insufficient position for sell: need {order.quantity}, "
                    f"have {current_position}"
                )
                return None

        # Execute order
        if order.side == OrderSide.BUY:
            self.cash -= total_cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:  # SELL
            self.cash += total_cost
            self.positions[order.symbol] -= order.quantity
            # Remove position if completely closed
            if self.positions[order.symbol] == 0:
                del self.positions[order.symbol]

        # Prevent negative cash (safety check)
        if self.cash < 0:
            logger.error(
                f"Negative cash balance: ${self.cash:.2f}. This should not happen!"
            )
            self.cash = 0

        # Create fill
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp,
            metadata=order.metadata,
        )

        self.fills.append(fill)

        if self.verbose:
            logger.info(
                f"Filled {order.side.value} {order.quantity:.2f} {order.symbol} "
                f"@ ${execution_price:.2f}, cost=${total_cost:.2f}"
            )

        return fill

    def process_signal(
        self,
        symbol: str,
        signal: int,
        timestamp: datetime,
        price: float,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Fill]:
        """Process a trading signal.

        Args:
            symbol: Trading symbol
            signal: Signal value (1=buy, -1=sell, 0=hold)
            timestamp: Signal timestamp
            price: Current price
            confidence: Signal confidence (for position sizing)
            metadata: Additional metadata

        Returns:
            Fill object if order executed, None otherwise
        """
        self.current_date = timestamp
        self.current_prices[symbol] = price

        # Update risk manager's positions
        self.risk_manager.update_positions({symbol: price}, timestamp)

        # No action for hold signal
        if signal == 0:
            return None

        # Handle buy signal
        if signal == 1:
            # Check if already have position
            if symbol in self.positions:
                logger.debug(f"Already have position in {symbol}, skipping buy")
                return None

            # Calculate position size
            if self.position_sizer:
                # Calculate stop loss for position sizing
                stop_loss = self.risk_manager.calculate_stop_loss(price, direction=1)

                # Calculate position size
                try:
                    if confidence is not None:
                        quantity = self.position_sizer.calculate_size(
                            capital=self.cash,
                            entry_price=price,
                            stop_loss_price=stop_loss,
                            confidence=confidence,
                        )
                    else:
                        quantity = self.position_sizer.calculate_size(
                            capital=self.cash,
                            entry_price=price,
                            stop_loss_price=stop_loss,
                        )
                except Exception as e:
                    logger.error(f"Position sizing failed: {e}")
                    return None
            else:
                # Default to 1% risk if no position sizer
                risk_amount = self.cash * 0.01
                stop_loss = self.risk_manager.calculate_stop_loss(price, direction=1)
                quantity = risk_amount / abs(price - stop_loss)

            # Validate quantity
            if quantity <= 0:
                logger.debug(f"Invalid position size: {quantity}, skipping trade")
                return None

            # Validate trade with risk manager
            stop_loss = self.risk_manager.calculate_stop_loss(price, direction=1)
            is_valid, issues = self.risk_manager.validate_trade(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                stop_loss=stop_loss,
                confidence=confidence,
            )

            if not is_valid:
                logger.warning(f"Trade validation failed: {issues}")
                return None

            # Create and execute buy order
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp,
                metadata=metadata,
            )

            fill = self.execute_order(order, price, timestamp)

            if fill:
                # Open position in risk manager
                take_profit = self.risk_manager.calculate_take_profit(price, direction=1)
                self.risk_manager.open_position(
                    symbol=symbol,
                    entry_price=price,
                    quantity=quantity,
                    entry_date=timestamp,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata=metadata,
                )

            return fill

        # Handle sell signal
        elif signal == -1:
            # Check if have position to sell
            if symbol not in self.positions:
                logger.debug(f"No position in {symbol} to sell")
                return None

            quantity = self.positions[symbol]

            # Create and execute sell order
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp,
                metadata=metadata,
            )

            fill = self.execute_order(order, price, timestamp)

            if fill:
                # Close position in risk manager
                self.risk_manager.close_position(
                    symbol=symbol,
                    exit_price=price,
                    exit_date=timestamp,
                    reason="signal",
                )

            return fill

        return None

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        confidences: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Run backtest on historical data.

        Args:
            data: DataFrame with price data (requires 'Close' column or single price column)
            signals: Series of trading signals (1=buy, -1=sell, 0=hold)
            confidences: Series of signal confidences (optional)

        Returns:
            Dictionary with backtest results

        Raises:
            ValueError: If data is invalid
        """
        if len(data) != len(signals):
            raise ValueError(
                f"Data and signals length mismatch: {len(data)} != {len(signals)}"
            )

        if data.empty:
            raise ValueError("Cannot backtest on empty data")

        # Determine price column
        if "Close" in data.columns:
            price_col = "Close"
        elif len(data.columns) == 1:
            price_col = data.columns[0]
        else:
            raise ValueError(
                f"Cannot determine price column. Columns: {data.columns.tolist()}"
            )

        logger.info(
            f"Starting backtest: {len(data)} periods, "
            f"initial capital=${self.initial_capital:,.2f}"
        )

        # Reset daily tracking at start
        self.risk_manager.reset_daily_tracking()

        # Process each time period
        for idx, (timestamp, row) in enumerate(data.iterrows()):
            price = row[price_col]
            signal = signals.iloc[idx]
            confidence = confidences.iloc[idx] if confidences is not None else None

            # Process signal
            self.process_signal(
                symbol=price_col,
                signal=signal,
                timestamp=timestamp,
                price=price,
                confidence=confidence,
            )

            # Record equity
            portfolio_val = self.portfolio_value
            self.equity_curve.append((timestamp, portfolio_val))

            # Update risk manager capital
            self.risk_manager.current_capital = portfolio_val
            self.risk_manager.peak_capital = max(
                self.risk_manager.peak_capital, portfolio_val
            )

        # Calculate performance metrics
        results = self._calculate_performance_metrics()

        logger.info(
            f"Backtest complete: Final capital=${results['final_capital']:,.2f}, "
            f"Return={results['total_return']:.2%}, Sharpe={results['sharpe_ratio']:.2f}"
        )

        return results

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.equity_curve:
            return {
                "final_capital": self.initial_capital,
                "total_return": 0.0,
                "error": "No equity curve data",
            }

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=["date", "equity"])
        equity_df.set_index("date", inplace=True)

        final_capital = equity_df["equity"].iloc[-1]
        total_return = (final_capital / self.initial_capital) - 1

        # Calculate returns
        equity_df["returns"] = equity_df["equity"].pct_change()

        # Basic metrics
        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_trades": len(self.risk_manager.trade_history),
            "total_fills": len(self.fills),
        }

        # Calculate additional metrics if we have returns
        if len(equity_df) > 1:
            returns = equity_df["returns"].dropna()

            # Annualized metrics (assuming daily data)
            trading_days = len(equity_df)
            years = trading_days / 252

            if years > 0:
                cagr = (final_capital / self.initial_capital) ** (1 / years) - 1
                metrics["cagr"] = cagr
            else:
                metrics["cagr"] = 0.0

            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

            if returns.std() > 0:
                sharpe_ratio = (returns.mean() - daily_rf) / returns.std() * np.sqrt(252)
                metrics["sharpe_ratio"] = sharpe_ratio
            else:
                metrics["sharpe_ratio"] = 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (
                    (returns.mean() - daily_rf) / downside_returns.std() * np.sqrt(252)
                )
                metrics["sortino_ratio"] = sortino_ratio
            else:
                metrics["sortino_ratio"] = 0.0

            # Win rate and profit metrics
            if self.risk_manager.trade_history:
                trades = pd.DataFrame(self.risk_manager.trade_history)
                winning_trades = trades[trades["pnl"] > 0]
                losing_trades = trades[trades["pnl"] < 0]

                metrics["win_rate"] = (
                    len(winning_trades) / len(trades) if len(trades) > 0 else 0
                )

                if len(winning_trades) > 0:
                    metrics["avg_win"] = winning_trades["pnl"].mean()
                    metrics["avg_win_pct"] = winning_trades["pnl_pct"].mean()
                else:
                    metrics["avg_win"] = 0.0
                    metrics["avg_win_pct"] = 0.0

                if len(losing_trades) > 0:
                    metrics["avg_loss"] = losing_trades["pnl"].mean()
                    metrics["avg_loss_pct"] = losing_trades["pnl_pct"].mean()
                else:
                    metrics["avg_loss"] = 0.0
                    metrics["avg_loss_pct"] = 0.0

                # Profit factor
                total_wins = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
                total_losses = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0

                if total_losses > 0:
                    metrics["profit_factor"] = total_wins / total_losses
                else:
                    metrics["profit_factor"] = float("inf") if total_wins > 0 else 0

                # R-expectancy
                if len(losing_trades) > 0 and metrics["avg_loss"] != 0:
                    avg_r = metrics["avg_win"] / abs(metrics["avg_loss"])
                    r_expectancy = (
                        metrics["win_rate"] * avg_r - (1 - metrics["win_rate"])
                    )
                    metrics["r_expectancy"] = r_expectancy
                else:
                    metrics["r_expectancy"] = 0.0

        # Get risk metrics from risk manager
        risk_metrics = self.risk_manager.get_risk_metrics()
        metrics.update({
            "max_drawdown": risk_metrics["max_drawdown"],
            "max_drawdown_duration": risk_metrics["max_drawdown_duration"],
            "var_95": risk_metrics["var_95"],
            "cvar_95": risk_metrics["cvar_95"],
        })

        # Calmar ratio (return / max drawdown)
        if metrics.get("max_drawdown", 0) > 0 and "cagr" in metrics:
            metrics["calmar_ratio"] = metrics["cagr"] / metrics["max_drawdown"]
        else:
            metrics["calmar_ratio"] = 0.0

        return metrics

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame.

        Returns:
            DataFrame with date and equity columns
        """
        if not self.equity_curve:
            return pd.DataFrame(columns=["date", "equity"])

        df = pd.DataFrame(self.equity_curve, columns=["date", "equity"])
        df.set_index("date", inplace=True)
        return df

    def get_trades(self) -> pd.DataFrame:
        """Get trade history as DataFrame.

        Returns:
            DataFrame with trade records
        """
        if not self.risk_manager.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.risk_manager.trade_history)

    def get_fills(self) -> pd.DataFrame:
        """Get fill history as DataFrame.

        Returns:
            DataFrame with fill records
        """
        if not self.fills:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": f.timestamp,
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "commission": f.commission,
                "slippage": f.slippage,
                "total_cost": f.total_cost,
            }
            for f in self.fills
        ])
