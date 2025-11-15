# Risk Management Integration Guide

## How to Integrate Risk Management into the Existing Backtesting Engine

This guide provides step-by-step instructions to integrate the risk management framework into your existing `BacktestingBase` class.

---

## Current State Analysis

### Issues in Current Implementation

```python
# CURRENT: BacktestingBase.place_buy_order()
def place_buy_order(self, bar, units=None, amount=None):
    date, price = self.get_date_price(bar)
    if units is None:
        units = int(amount / (price * (1 + self.ptc)))
    self.units += units
    self.current_balance -= units * price * (1 + self.ptc)
    self.trades += 1
```

**Problems:**
1. No position sizing validation (allows 50% of capital in one trade)
2. No stop-loss enforcement (position held until model changes signal)
3. No take-profit targets
4. No position size limits (concentration risk)
5. No portfolio risk tracking (heat)
6. No risk metrics (VaR, Sharpe, etc.)
7. All trades get same size regardless of confidence
8. No validation before entry
9. No maximum trade duration
10. No drawdown tracking

### Result
All models show negative returns, but the root cause is not just the models—it's the lack of risk control allowing large losses.

---

## Step-by-Step Integration

### Step 1: Import Risk Management Classes

Add these imports to the top of your notebook/script:

```python
from risk_management_implementation import (
    FixedFractionalSizer,
    ConfidenceBasedSizer,
    StopLossManager,
    TradeExecution,
    DrawdownTracker,
    PortfolioHeat,
    TradeValidator,
    RiskMetrics
)
```

---

### Step 2: Modify BacktestingBase.__init__()

**Before:**
```python
class BacktestingBase(FinancialData):
    def __init__(self, url, symbol, model, amount, ptc, verbose=False):
        super().__init__(url, symbol, model)
        self.initial_amount = amount
        self.current_balance = amount
        self.ptc = ptc
        self.units = 0
        self.trades = 0
        self.wealth_over_time = []
        self.verbose = verbose
        # ... rest of initialization
```

**After:**
```python
class BacktestingBase(FinancialData):
    def __init__(self, url, symbol, model, amount, ptc, verbose=False):
        super().__init__(url, symbol, model)
        self.initial_amount = amount
        self.current_balance = amount
        self.ptc = ptc
        self.units = 0
        self.trades = 0
        self.wealth_over_time = []
        self.verbose = verbose

        # ADD RISK MANAGEMENT COMPONENTS
        self.position_sizer = FixedFractionalSizer(amount, risk_per_trade_pct=0.01)
        self.confidence_sizer = ConfidenceBasedSizer(self.position_sizer)
        self.stop_loss_manager = StopLossManager(
            stop_loss_pct=0.03,      # 3% below entry
            take_profit_pct=0.05,    # 5% above entry
            min_risk_reward_ratio=2.0
        )
        self.drawdown_tracker = DrawdownTracker(amount, max_drawdown_pct=0.10)
        self.portfolio_heat = PortfolioHeat(amount, max_heat_pct=0.03)
        self.trade_validator = TradeValidator()

        # Trade tracking
        self.trade_objects = []  # List of TradeExecution objects
        self.trade_id_counter = 0
        self.open_position = None  # Current open TradeExecution

        # Performance tracking
        self.daily_returns = []
        self.equity_history = [amount]
        self.risk_metrics_history = []

        # split_data, normalize_data, add_lags, train_model, apply_model
        self.split_data()
        self.normalize_data()
        self.add_lags(self.lags)
        self.train_model()
        self.apply_model()
```

---

### Step 3: Add Prediction Probability Support

Modify `apply_model()` to get prediction probabilities:

**Before:**
```python
def apply_model(self):
    self.train['prediction'] = self.model.predict(self.train[self.cols_])
    self.test['prediction'] = self.model.predict(self.test[self.cols_])
```

**After:**
```python
def apply_model(self):
    # Get class predictions
    self.train['prediction'] = self.model.predict(self.train[self.cols_])
    self.test['prediction'] = self.model.predict(self.test[self.cols_])

    # Get prediction probabilities for confidence-based sizing
    try:
        train_proba = self.model.predict_proba(self.train[self.cols_])
        test_proba = self.model.predict_proba(self.test[self.cols_])

        # Store max probability as confidence for each prediction
        self.train['confidence'] = train_proba.max(axis=1)
        self.test['confidence'] = test_proba.max(axis=1)
    except AttributeError:
        # Model doesn't have predict_proba (e.g., SVM)
        # Use decision_function if available, otherwise default to 0.6
        if hasattr(self.model, 'decision_function'):
            train_scores = self.model.decision_function(self.train[self.cols_])
            test_scores = self.model.decision_function(self.test[self.cols_])
            # Convert scores to probabilities (0-1 range)
            self.train['confidence'] = 1 / (1 + np.exp(-train_scores))
            self.test['confidence'] = 1 / (1 + np.exp(-test_scores))
        else:
            # Default: assume 60% confidence for all predictions
            self.train['confidence'] = 0.60
            self.test['confidence'] = 0.60
```

---

### Step 4: Replace place_buy_order() and place_sell_order()

**Before:**
```python
def place_buy_order(self, bar, units=None, amount=None):
    date, price = self.get_date_price(bar)
    if units is None:
        units = int(amount / (price * (1 + self.ptc)))
    self.units += units
    self.current_balance -= units * price * (1 + self.ptc)
    self.trades += 1
    if self.verbose:
        print(f'{date} | bought {units} units for {price:.2f}')
        self.print_balance(bar)
        self.print_net_wealth(bar)
```

**After:**
```python
def place_buy_order(self, bar, units=None, amount=None, stop_loss=None,
                   take_profit=None, confidence=0.60):
    """
    Place buy order with risk management

    Args:
        bar: Bar index
        units: Number of units (calculated if not provided)
        amount: Dollar amount (used if units not provided)
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        confidence: Model confidence (0-1)
    """
    date, price = self.get_date_price(bar)

    # Close any existing position first
    if self.open_position is not None:
        self.place_sell_order(bar)

    # Calculate stop-loss and take-profit if not provided
    if stop_loss is None or take_profit is None:
        levels = self.stop_loss_manager.calculate_stop_and_target(price)
        stop_loss = levels['stop_loss']
        take_profit = levels['take_profit']

    # Validate risk/reward
    is_valid, rr_ratio, rr_msg = self.stop_loss_manager.validate_risk_reward(
        price, stop_loss, take_profit
    )
    if not is_valid:
        if self.verbose:
            print(f'{date} | BUY REJECTED: {rr_msg}')
        return False

    # Calculate position size (with confidence adjustment)
    if units is None:
        if amount is None:
            amount = self.current_balance  # Use all available cash

        # Fixed fractional sizing
        risk_per_unit = price - stop_loss
        base_units = int(amount / (price * (1 + self.ptc)))

        # Adjust for confidence
        confidence_multiplier = self.confidence_sizer.get_confidence_multiplier(confidence)
        units = int(base_units * confidence_multiplier)

    if units <= 0:
        if self.verbose:
            print(f'{date} | BUY REJECTED: Position size zero after risk adjustment')
        return False

    # Validate position limits
    position_value = units * price
    position_pct = position_value / self.initial_amount

    if position_pct > 0.05:  # Max 5% per position
        if self.verbose:
            print(f'{date} | BUY REJECTED: Position {position_pct:.1%} exceeds 5% limit')
        return False

    # Validate portfolio heat
    can_add, heat_msg = self.portfolio_heat.can_add_position(
        self.symbol, units, price - stop_loss
    )
    if not can_add:
        if self.verbose:
            print(f'{date} | BUY REJECTED: {heat_msg}')
        return False

    # Validate drawdown zone
    drawdown_zone = self.drawdown_tracker.get_drawdown_zone(
        (self.drawdown_tracker.peak_equity - self.current_balance) /
        self.drawdown_tracker.peak_equity
    )

    if "EMERGENCY" in drawdown_zone:
        if self.verbose:
            print(f'{date} | BUY REJECTED: Emergency mode - {drawdown_zone}')
        return False

    # Validate signal quality
    if confidence < 0.52:
        if self.verbose:
            print(f'{date} | BUY REJECTED: Confidence {confidence:.2%} below 52% minimum')
        return False

    # Execute trade
    self.trade_id_counter += 1
    trade = TradeExecution(
        trade_id=self.trade_id_counter,
        symbol=self.symbol,
        entry_price=price,
        units=units,
        entry_date=pd.Timestamp(self.data.index[bar]),
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence
    )

    self.open_position = trade
    self.units = units
    self.current_balance -= units * price * (1 + self.ptc)
    self.trades += 1
    self.trade_objects.append(trade)
    self.portfolio_heat.add_position(self.symbol, units, price, price - stop_loss)

    if self.verbose:
        print(f'{date} | BOUGHT {units} units at {price:.2f}')
        print(f'       | Stop: {stop_loss:.2f}, Target: {take_profit:.2f}')
        print(f'       | Risk/Reward: {rr_ratio:.2f}, Confidence: {confidence:.2%}')
        self.print_balance(bar)
        self.print_net_wealth(bar)

    return True


def place_sell_order(self, bar, units=None, exit_type='manual'):
    """
    Close position (sell order)

    Args:
        bar: Bar index
        units: Number of units to sell (all if not specified)
        exit_type: Type of exit ('stop', 'target', 'time', 'manual')
    """
    if self.units == 0:
        return False

    date, price = self.get_date_price(bar)

    if units is None:
        units = self.units

    # Close trade
    self.units -= units
    self.current_balance += units * price * (1 - self.ptc)
    self.trades += 1

    # Record trade outcome
    if self.open_position is not None:
        self.open_position.exit_trade(
            exit_price=price,
            exit_date=pd.Timestamp(self.data.index[bar]),
            exit_type=exit_type
        )
        self.portfolio_heat.close_position(self.open_position.symbol)

        if self.verbose:
            print(f'{date} | SOLD {units} units at {price:.2f}')
            print(f'       | Exit Type: {exit_type}')
            print(f'       | P&L: {self.open_position.pnl:.2f} ({self.open_position.r_multiple:.2f}R)')
            self.print_balance(bar)
            self.print_net_wealth(bar)

        self.open_position = None

    return True
```

---

### Step 5: Implement Stop-Loss and Take-Profit Checking

Add a new method to check stop-loss/take-profit at each bar:

```python
def check_stops_and_targets(self, bar):
    """
    Check if position should be closed due to stop-loss or take-profit

    Args:
        bar: Current bar index
    """
    if self.open_position is None or self.units == 0:
        return

    date, price = self.get_date_price(bar)

    # Check stop-loss
    if price <= self.open_position.stop_loss:
        if self.verbose:
            print(f'{date} | STOP-LOSS HIT at {price:.2f}')
        self.place_sell_order(bar, exit_type='stop')
        return

    # Check take-profit
    if price >= self.open_position.take_profit:
        if self.verbose:
            print(f'{date} | TAKE-PROFIT HIT at {price:.2f}')
        self.place_sell_order(bar, exit_type='target')
        return

    # Check trailing stop (once in profit)
    if price > self.open_position.entry_price:
        trailing_stop = self.stop_loss_manager.get_trailing_stop(
            self.open_position.entry_price,
            price,
            profit_threshold=0.02
        )

        if trailing_stop is not None and price <= trailing_stop:
            if self.verbose:
                print(f'{date} | TRAILING STOP HIT at {price:.2f}')
            self.place_sell_order(bar, exit_type='manual')
            return

    # Check time-based stop (max 10 days)
    if self.open_position.days_open() >= 10:
        if self.verbose:
            print(f'{date} | TIME STOP (10 days) - closing position')
        self.place_sell_order(bar, exit_type='time')
        return
```

---

### Step 6: Update run_strategy() to Use Risk Management

**Before:**
```python
def run_strategy(self):
    print(92 * '=')
    print(f'*** BACKTESTING STRATEGY ***')
    print(f'{self.symbol} | lags={self.lags} | model={model}')
    print(92 * '=')

    for test_bar in range(len(self.test)):
        bar = test_bar + len(self.train)
        trade = False
        prediction = self.test['prediction'].iloc[test_bar]

        if self.units == 0:
            if prediction == 1:
                self.place_buy_order(bar, amount=self.current_balance/2)
                trade = True
        else:
            if prediction == 0:
                self.place_sell_order(bar, units=self.units)
                trade = True

        if trade and self.verbose:
            self.print_balance(bar)
            self.print_net_wealth(bar)
            print(92 * '-')

    self.close_out(bar)
```

**After:**
```python
def run_strategy(self):
    print(92 * '=')
    print(f'*** BACKTESTING STRATEGY WITH RISK MANAGEMENT ***')
    print(f'{self.symbol} | Model: {self.model.__class__.__name__}')
    print(f'Risk per Trade: 1% | Max Drawdown: 10% | Max Heat: 3%')
    print(92 * '=')

    for test_bar in range(len(self.test)):
        bar = test_bar + len(self.train)
        date, price = self.get_date_price(bar)

        # Get prediction and confidence
        prediction = self.test['prediction'].iloc[test_bar]
        confidence = self.test.get('confidence', pd.Series([0.60] * len(self.test))).iloc[test_bar]

        # Update drawdown tracker
        current_equity = self.current_balance + (self.units * price)
        self.equity_history.append(current_equity)
        within_limits, dd_status = self.drawdown_tracker.update_equity(current_equity)

        if not within_limits and self.verbose:
            print(f'{date} | WARNING: Drawdown exceeded limits! {dd_status}')

        # Check stops and targets on open position
        self.check_stops_and_targets(bar)

        # Entry logic
        if self.units == 0:  # No open position
            if prediction == 1:  # Buy signal
                success = self.place_buy_order(
                    bar,
                    amount=None,  # Will be calculated
                    confidence=confidence
                )
                if success and self.verbose:
                    print(92 * '-')

        else:  # Position is open
            if prediction == 0:  # Sell signal (from model)
                success = self.place_sell_order(bar, exit_type='signal')
                if success and self.verbose:
                    print(92 * '-')

        # Calculate daily return if position exists
        if self.units > 0:
            daily_equity = self.current_balance + (self.units * price)
            if len(self.equity_history) > 1:
                daily_return = (daily_equity - self.equity_history[-2]) / self.equity_history[-2]
                self.daily_returns.append(daily_return)

    # Close out any remaining position
    self.close_out(len(self.data) - 1)

    # Print risk metrics
    self.print_risk_metrics()
```

---

### Step 7: Add Methods to Track and Report Risk Metrics

```python
def print_risk_metrics(self):
    """Print comprehensive risk metrics at end of backtest"""
    print(92 * '=')
    print('*** RISK METRICS AND PERFORMANCE ***')
    print(92 * '=')

    # Basic performance
    total_return = (self.current_balance - self.initial_amount) / self.initial_amount
    print(f'\nPERFORMANCE:')
    print(f'  Initial Capital:      ${self.initial_amount:,.2f}')
    print(f'  Final Balance:        ${self.current_balance:,.2f}')
    print(f'  Total Return:         {total_return:,.2%}')
    print(f'  Total Trades:         {self.trades}')

    # Trade statistics
    if self.trade_objects:
        closed_trades = [t for t in self.trade_objects if t.pnl is not None]

        if closed_trades:
            r_multiples = [t.r_multiple for t in closed_trades]

            wins = [t for t in closed_trades if t.pnl > 0]
            losses = [t for t in closed_trades if t.pnl <= 0]

            print(f'\nTRADE STATISTICS:')
            print(f'  Total Closed Trades:  {len(closed_trades)}')
            print(f'  Winners:              {len(wins)} ({len(wins)/len(closed_trades):.1%})')
            print(f'  Losers:               {len(losses)} ({len(losses)/len(closed_trades):.1%})')

            if wins:
                avg_win = sum([t.pnl for t in wins]) / len(wins)
                print(f'  Avg Winner:           ${avg_win:.2f}')

            if losses:
                avg_loss = sum([t.pnl for t in losses]) / len(losses)
                print(f'  Avg Loser:            ${avg_loss:.2f}')

            expectancy = RiskMetrics.calculate_r_expectancy(r_multiples)
            print(f'\nR-MULTIPLE ANALYSIS:')
            print(f'  R Expectancy:         {expectancy["expectancy"]:+.2f}R per trade')
            print(f'  Win Rate:             {expectancy["win_rate"]:.1%}')
            print(f'  Avg Win (R):          {expectancy["avg_win"]:+.2f}R')
            print(f'  Avg Loss (R):         {expectancy["avg_loss"]:+.2f}R')
            print(f'  Profit Factor:        {expectancy["profit_factor"]:.2f}')

    # Risk metrics
    if self.daily_returns:
        daily_returns = np.array(self.daily_returns)
        equity_curve = np.array(self.equity_history)

        print(f'\nRISK METRICS:')
        print(f'  Max Drawdown:         {RiskMetrics.calculate_max_drawdown(equity_curve):.2%}')
        print(f'  Sharpe Ratio:         {RiskMetrics.calculate_sharpe_ratio(daily_returns):.2f}')
        print(f'  Sortino Ratio:        {RiskMetrics.calculate_sortino_ratio(daily_returns):.2f}')
        print(f'  VaR (95%):            ${RiskMetrics.calculate_var_95(daily_returns, self.current_balance):.2f}')
        print(f'  CVaR (95%):           ${RiskMetrics.calculate_cvar_95(daily_returns, self.current_balance):.2f}')

    print(f'\nDRAWDOWN TRACKING:')
    dd_stats = self.drawdown_tracker.get_stats()
    print(f'  Current Drawdown:     {dd_stats["current_dd_pct"]:.2%}')
    print(f'  Max Drawdown:         {dd_stats["max_dd_pct"]:.2%}')
    print(f'  Drawdown Limit:       {dd_stats["max_allowed_pct"]:.2%}')
    print(f'  Within Limits:        {dd_stats["within_limits"]}')

    print(92 * '=')

def get_trade_log(self) -> pd.DataFrame:
    """Get detailed trade log as DataFrame"""
    trades_data = [t.to_dict() for t in self.trade_objects]
    return pd.DataFrame(trades_data)
```

---

## Verification: Expected Changes

### Before Integration
```
Model: gauss
- Final balance: $10,000
- Trades: 1
- No drawdown protection
- No position limits
- Max loss: 50% of account potential
```

### After Integration
```
Model: gauss
- Final balance: Likely same (if model doesn't predict)
- Trades: 0-1 (more selective)
- Max drawdown: < 10%
- Max per position: 5% of account
- Max loss per trade: ~1%

Benefits:
- Capital protection
- Measurable risk metrics
- Confidence-based sizing
- Stop-loss enforcement
- Opportunity to improve through data and model refinement
```

---

## Configuration Parameters to Tune

Once integrated, adjust these based on live backtest results:

```python
# In BacktestingBase.__init__()

# Position Sizing
self.position_sizer = FixedFractionalSizer(amount, risk_per_trade_pct=0.01)  # Try 0.5%, 1%, 1.5%

# Stop Loss
self.stop_loss_manager = StopLossManager(
    stop_loss_pct=0.03,      # Try 0.02, 0.03, 0.05
    take_profit_pct=0.05,    # Try 0.03, 0.05, 0.10
    min_risk_reward_ratio=2.0  # Try 1.5, 2.0, 3.0
)

# Drawdown Limits
self.drawdown_tracker = DrawdownTracker(amount, max_drawdown_pct=0.10)  # Try 0.05, 0.10, 0.15

# Portfolio Heat
self.portfolio_heat = PortfolioHeat(amount, max_heat_pct=0.03)  # Try 0.02, 0.03, 0.05
```

---

## Testing Checklist

After integration, verify:

- [ ] No trades execute if confidence < 52%
- [ ] Positions close at hard stop-loss
- [ ] Positions close at take-profit target
- [ ] Positions close after 10 days if still open
- [ ] Max drawdown never exceeds 10%
- [ ] Max position size never exceeds 5% of account
- [ ] Portfolio heat never exceeds 3%
- [ ] Daily equity curve shows proper tracking
- [ ] Risk metrics calculate correctly
- [ ] Trade log shows proper entry/exit with R-multiples

---

## Next Steps for Model Improvement

With risk management in place, improve returns by:

1. **Feature Engineering**: Add technical indicators (RSI, MACD, Bollinger Bands)
2. **Hyperparameter Tuning**: Use GridSearchCV for optimal model parameters
3. **Ensemble Methods**: Combine multiple model predictions
4. **Threshold Optimization**: Find optimal confidence threshold > 52%
5. **Regime Detection**: Identify market conditions and adjust strategy
6. **Position Sizing Optimization**: Use Kelly Criterion with confirmed edge
7. **Entry/Exit Optimization**: Find optimal stop and target percentages
8. **Cross-Validation**: Use proper time-series cross-validation

The risk management system now provides safe framework for these improvements.
