# Production Trading System Modules

This document describes the production-ready modules created for the algorithmic trading system.

## Overview

The system consists of four main modules:

1. **Position Sizing** (`algo_trading/risk/position_sizing.py`)
2. **Risk Manager** (`algo_trading/risk/manager.py`)
3. **Backtest Engine** (`algo_trading/backtesting/engine.py`)
4. **ML Strategy** (`algo_trading/strategies/ml_strategy.py`)

## Module Details

### 1. Position Sizing (`algo_trading/risk/position_sizing.py`)

Implements three position sizing strategies based on quantitative finance best practices.

#### Classes

**FixedFractionalSizing**
- Risks a fixed percentage of capital per trade (default: 1%)
- Position size calculated based on distance to stop loss
- Includes maximum position size constraint (default: 25%)

```python
from algo_trading.risk import FixedFractionalSizing

sizer = FixedFractionalSizing(risk_fraction=0.01, max_position_fraction=0.25)
position_size = sizer.calculate_size(
    capital=10000,
    entry_price=100,
    stop_loss_price=97,
)
```

**KellyCriterion**
- Optimal position sizing based on Kelly formula
- Uses win rate and reward/risk ratio
- Includes fractional Kelly for safety (default: 25%)
- Minimum win rate requirement (default: 40%)

```python
from algo_trading.risk import KellyCriterion

kelly = KellyCriterion(kelly_fraction=0.25, min_win_rate=0.40)
position_size = kelly.calculate_size(
    capital=10000,
    entry_price=100,
    stop_loss_price=97,
    win_rate=0.60,
    avg_win=300,
    avg_loss=100,
)
```

**ConfidenceBasedSizing**
- Scales position size based on ML model confidence
- Higher confidence = larger position (within limits)
- Linear interpolation between base and max risk

```python
from algo_trading.risk import ConfidenceBasedSizing

conf_sizer = ConfidenceBasedSizing(
    base_risk_fraction=0.01,
    max_risk_fraction=0.02,
    min_confidence=0.55,
)
position_size = conf_sizer.calculate_size(
    capital=10000,
    entry_price=100,
    stop_loss_price=97,
    confidence=0.75,
)
```

#### Factory Function

```python
from algo_trading.risk import create_position_sizer

sizer = create_position_sizer(
    method='confidence_based',
    base_risk_fraction=0.01,
    max_risk_fraction=0.02,
)
```

---

### 2. Risk Manager (`algo_trading/risk/manager.py`)

Comprehensive risk management system with multi-layered controls.

#### Features

**Stop Loss Management**
- Hard stop loss (default: 3%)
- Trailing stop loss (default: 2%)
- Take profit targets (default: 5%)
- Time-based stops (default: 30 days)

**Drawdown Control - 4 Zone System**
- NORMAL: 0-5% drawdown (full trading)
- CAUTION: 5-10% drawdown (reduced risk)
- ALERT: 10-15% drawdown (minimal trading)
- EMERGENCY: >15% drawdown (trading halted)

**Portfolio Heat Tracking**
- Tracks total portfolio risk exposure
- Default maximum: 3% of capital at risk
- Prevents over-leveraging

**Trade Validation - 7 Point Checklist**
1. Sufficient balance check
2. Position limit check (max 25% per position)
3. Confidence threshold check
4. Drawdown zone check
5. Portfolio heat check
6. Daily loss limit check (default: 3%)
7. Valid prices check

**Risk Metrics**
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum drawdown
- Drawdown duration

#### Usage Example

```python
from algo_trading.risk import RiskManager

# Initialize risk manager
risk_manager = RiskManager(
    initial_capital=10000,
    max_drawdown_pct=0.15,
    daily_loss_limit_pct=0.03,
    max_portfolio_heat=0.03,
    hard_stop_pct=0.03,
    trailing_stop_pct=0.02,
    take_profit_pct=0.05,
)

# Validate trade
is_valid, issues = risk_manager.validate_trade(
    symbol="AAPL",
    quantity=10,
    entry_price=100,
    stop_loss=97,
    confidence=0.65,
)

if is_valid:
    # Open position
    risk_manager.open_position(
        symbol="AAPL",
        entry_price=100,
        quantity=10,
        entry_date=datetime.now(),
    )

# Update positions with current prices
closed_trades = risk_manager.update_positions(
    current_prices={"AAPL": 102},
    current_date=datetime.now(),
)

# Get risk metrics
metrics = risk_manager.get_risk_metrics()
print(f"Current drawdown: {metrics['current_drawdown']:.2%}")
print(f"Portfolio heat: {metrics['portfolio_heat']:.2%}")

# Get summary
print(risk_manager.get_summary())
```

---

### 3. Backtest Engine (`algo_trading/backtesting/engine.py`)

Event-driven backtesting engine with realistic execution simulation.

#### Features

**Realistic Execution**
- Market orders with slippage (default: 5 bps)
- Transaction costs (default: 0.15%)
- Order validation
- Prevents negative balances

**Risk Integration**
- Full integration with RiskManager
- Position sizing strategy support
- Trade validation
- Stop loss management

**Performance Tracking**
- Equity curve generation
- Comprehensive metrics calculation
- Trade-by-trade logging
- Fill history

**Performance Metrics**
- Total return, CAGR
- Sharpe ratio, Sortino ratio
- Maximum drawdown, Calmar ratio
- Win rate, profit factor
- R-expectancy
- VaR, CVaR

#### Usage Example

```python
from algo_trading.backtesting import BacktestEngine
from algo_trading.risk import RiskManager, create_position_sizer

# Setup components
risk_manager = RiskManager(initial_capital=10000)
position_sizer = create_position_sizer('fixed_fractional', risk_fraction=0.01)

# Create engine
engine = BacktestEngine(
    initial_capital=10000,
    risk_manager=risk_manager,
    position_sizer=position_sizer,
    commission_pct=0.0015,
    slippage_bps=5,
)

# Run backtest
results = engine.run_backtest(
    data=price_data,  # DataFrame with price column
    signals=signals,  # Series of -1, 0, 1
    confidences=confidences,  # Series of confidence scores
)

# Analyze results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")

# Get detailed data
equity_curve = engine.get_equity_curve()
trades = engine.get_trades()
fills = engine.get_fills()
```

---

### 4. ML Strategy (`algo_trading/strategies/ml_strategy.py`)

Machine learning-based trading strategy using model predictions and confidence scores.

#### Features

**Prediction-Based Signals**
- Uses `predict_proba` for confidence scores
- Configurable confidence thresholds
- Prevents trading when confidence is low
- Optional short selling support

**Signal Generation**
- BUY (1): Model predicts up with high confidence
- HOLD (0): Confidence below threshold
- SELL (-1): Exit position or short signal

**Threshold Optimization**
- Grid search over thresholds
- Multiple optimization metrics (Sharpe, return, Calmar)
- Walk-forward optimization support

#### Usage Example

```python
from algo_trading.strategies import MLPredictionStrategy
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create strategy
strategy = MLPredictionStrategy(
    model=model,
    confidence_threshold=0.55,
    feature_columns=feature_cols,
)

# Generate signals
signals, confidences = strategy.generate_signals_simple(
    data=X_test,
    return_confidence=True,
)

# Get statistics
stats = strategy.get_signal_statistics(signals)
print(f"Buy signals: {stats['buy_signals']}")
print(f"Trade frequency: {stats['trade_frequency']:.2%}")

# Optimize threshold
from algo_trading.strategies import ThresholdOptimizer

results_df = ThresholdOptimizer.grid_search(
    strategy=strategy,
    data=X_test,
    returns=actual_returns,
    threshold_range=(0.5, 0.9),
    step=0.05,
    metric='sharpe',
)
```

---

## Complete Integration Example

See `example_usage.py` for a complete end-to-end example showing:

1. Data loading and feature engineering
2. Model training
3. Strategy creation
4. Risk management setup
5. Backtesting
6. Results analysis

Run the example:

```bash
python example_usage.py
```

---

## Testing

Run the module tests:

```bash
python test_modules.py
```

This verifies:
- All modules import correctly
- Position sizing strategies work
- Risk manager functions properly
- Backtest engine executes

---

## Configuration

All parameters are configurable via `config.yaml`:

```yaml
risk_management:
  position_sizing:
    method: "confidence_based"  # or "fixed_fractional", "kelly"
    risk_per_trade: 0.01
    max_position_size: 0.25

  stop_loss:
    enabled: true
    hard_stop_pct: 0.03
    trailing_stop_pct: 0.02
    take_profit_pct: 0.05
    time_stop_days: 30

  portfolio:
    max_drawdown_pct: 0.15
    daily_loss_limit_pct: 0.03
    max_portfolio_heat: 0.03

  drawdown_zones:
    normal: 0.05
    caution: 0.10
    alert: 0.15
    emergency: 0.20

strategy:
  type: "MLPredictionStrategy"
  confidence_threshold: 0.55

backtesting:
  initial_capital: 10000
  transaction_cost: 0.0015
  slippage_bps: 5
```

---

## Best Practices

### Position Sizing
- Start conservative (1% risk per trade)
- Use fractional Kelly (25% or less)
- Never risk more than 2% on a single trade
- Adjust based on confidence when applicable

### Risk Management
- Always validate trades before execution
- Monitor drawdown zones actively
- Respect daily loss limits
- Use trailing stops to protect profits

### Backtesting
- Include realistic transaction costs
- Model slippage appropriately
- Prevent data leakage (no future information)
- Test out-of-sample (walk-forward)

### Strategy
- Require minimum confidence threshold (55%+)
- Monitor signal frequency
- Avoid overtrading
- Optimize thresholds on validation set

---

## Error Handling

All modules include comprehensive error handling:

- Input validation with descriptive errors
- Graceful handling of edge cases
- Detailed logging for debugging
- Safe defaults to prevent losses

---

## Logging

Configure logging level in code or config:

```python
from algo_trading.utils.logger import setup_logger

logger = setup_logger(
    name="trading",
    log_file="./logs/trading.log",
    level="INFO",
    console=True,
)
```

Logging levels:
- DEBUG: Detailed execution information
- INFO: High-level progress and results
- WARNING: Potential issues (e.g., low confidence)
- ERROR: Failures and exceptions

---

## Performance Considerations

**Position Sizing**: O(1) - constant time
**Risk Manager**: O(n) where n = number of open positions
**Backtest Engine**: O(m) where m = number of time periods
**ML Strategy**: O(m × f) where f = number of features

For large datasets:
- Use vectorized operations where possible
- Consider batching for very long backtests
- Cache model predictions if reusing

---

## Future Enhancements

Potential improvements:

1. **Multi-asset support** - Track multiple symbols
2. **Advanced order types** - Limit orders, stop-limit
3. **Slippage models** - Volume-based, spread-based
4. **Transaction cost models** - Dynamic based on size
5. **Walk-forward optimization** - Automated parameter tuning
6. **Portfolio optimization** - Modern portfolio theory
7. **Risk parity** - Equal risk contribution
8. **Regime detection** - Adapt to market conditions

---

## Support

For issues or questions:
1. Check the code documentation (comprehensive docstrings)
2. Review example files (`example_usage.py`, `test_modules.py`)
3. Examine unit tests for usage patterns
4. Check configuration in `config.yaml`

---

## License

See LICENSE file in project root.
