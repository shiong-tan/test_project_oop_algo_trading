# Production-Ready Algorithmic Trading System

A comprehensive, modular Python package for ML-based algorithmic trading with enterprise-grade risk management, backtesting, and performance analysis.

## 🎯 Overview

This system transforms a basic prediction-based trading strategy into a production-ready trading platform with:

- **Leakage-free feature engineering** (fixed critical data leakage bugs)
- **Comprehensive risk management** (position sizing, stops, drawdown control)
- **Event-driven backtesting** (realistic execution with costs)
- **ML-based trading strategies** (confidence-based signal generation)
- **Professional visualization** (equity curves, drawdown analysis, performance reports)
- **Extensive testing** (unit tests with pytest)

## 📊 Features

### Core Capabilities

- **Data Management**: Robust data loading with error handling and validation
- **Feature Engineering**: 65+ technical features with proper train/test separation
- **ML Models**: Support for RandomForest, XGBoost, LightGBM, Logistic Regression
- **Risk Management**:
  - 3 position sizing strategies (Fixed Fractional, Kelly, Confidence-Based)
  - 4-zone drawdown system
  - VaR/CVaR calculation
  - Portfolio heat tracking
- **Backtesting**: Event-driven engine with slippage and commission modeling
- **Visualization**: Comprehensive performance charts and reports

### Technical Highlights

✅ **Fixed Critical Bugs**:
- Test set normalization (scaler fitted only on training data)
- Look-ahead bias (features calculated separately for train/test)
- Information leakage (excluded current direction from features)

✅ **Production Features**:
- SOLID design principles
- Comprehensive error handling
- Type hints throughout
- Structured logging
- Configuration management via YAML

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shiong-tan/test_project_oop_algo_trading.git
cd test_project_oop_algo_trading

# Install package
pip install -e .

# Or install with all dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from algo_trading.data.loader import DataLoader
from algo_trading.features.engineering import FeatureEngineer
from algo_trading.models.trainer import ModelTrainer
from algo_trading.backtesting.engine import BacktestEngine
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
loader = DataLoader()
data = loader.load_from_url(
    "http://hilpisch.com/ref_eikon_eod_data.csv",
    symbol="AAPL.O"
)

# 2. Engineer features (leakage-free!)
engineer = FeatureEngineer()
train_features, feature_cols = engineer.fit_transform(train_data, price_col="AAPL.O")
test_features = engineer.transform(test_data)  # No fitting on test!

# 3. Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
trainer = ModelTrainer(model)
trainer.train(train_features[feature_cols], train_features['direction'])

# 4. Backtest
engine = BacktestEngine(initial_capital=10000)
results = engine.run_backtest(test_data, signals, confidences)
```

### Command-Line Usage

```bash
# Run full backtest with Random Forest
python main.py --model rf

# Run with XGBoost and verbose logging
python main.py --model xgb --verbose

# Run simple example
python examples/simple_backtest.py

# Run integration tests
python test_integration.py
```

## 📁 Project Structure

```
algo_trading/
├── data/
│   └── loader.py                 # Data loading with error handling
├── features/
│   └── engineering.py            # Feature engineering (leakage-free)
├── models/
│   └── trainer.py                # Model training and evaluation
├── backtesting/
│   └── engine.py                 # Event-driven backtesting
├── risk/
│   ├── position_sizing.py        # Position sizing strategies
│   └── manager.py                # Risk management system
├── strategies/
│   └── ml_strategy.py            # ML-based trading strategy
├── visualization/
│   └── plots.py                  # Performance visualization
├── utils/
│   ├── config.py                 # Configuration management
│   └── logger.py                 # Logging utilities
└── tests/                        # Unit tests (pytest)

config.yaml                       # System configuration
main.py                           # Main execution script
examples/simple_backtest.py       # Simple example
```

## ⚙️ Configuration

All parameters are configurable via `config.yaml`:

```yaml
risk_management:
  position_sizing:
    method: "confidence_based"    # or "fixed_fractional", "kelly"
    risk_per_trade: 0.01         # 1% risk per trade
    max_position_size: 0.25      # Max 25% of capital

  stop_loss:
    hard_stop_pct: 0.03          # 3% stop loss
    trailing_stop_pct: 0.02      # 2% trailing stop
    take_profit_pct: 0.05        # 5% take profit

  portfolio:
    max_drawdown_pct: 0.15       # 15% maximum drawdown
    daily_loss_limit_pct: 0.03   # 3% daily loss limit

backtesting:
  initial_capital: 10000
  transaction_cost: 0.0015       # 0.15% per trade
  slippage_bps: 5                # 5 basis points

strategy:
  confidence_threshold: 0.55     # Minimum confidence to trade
```

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

**Capital Metrics:**
- Total Return
- CAGR (Compound Annual Growth Rate)

**Risk Metrics:**
- Maximum Drawdown
- VaR (Value at Risk)
- CVaR (Conditional VaR)

**Risk-Adjusted Returns:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Trading Statistics:**
- Win Rate
- Profit Factor
- R-Expectancy
- Average Win/Loss

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=algo_trading --cov-report=html

# Run specific test file
pytest algo_trading/tests/test_risk_manager.py

# Run integration tests
python test_integration.py
```

**Test Coverage:**
- Data loader: Input validation, error handling
- Feature engineering: Leakage prevention, transformations
- Model trainer: Training, evaluation, cross-validation
- Risk manager: Position management, drawdown control
- Position sizing: All three strategies
- Backtesting: Order execution, portfolio tracking

## 📊 Visualization

Create comprehensive performance reports:

```python
from algo_trading.visualization.plots import create_visualizer

visualizer = create_visualizer()

# Equity curve
visualizer.plot_equity_curve(equity_df, save_path="equity_curve.png")

# Drawdown analysis
visualizer.plot_drawdown(equity_df, save_path="drawdown.png")

# Trade analysis
visualizer.plot_trade_analysis(trades_df, save_path="trades.png")

# Complete summary report
visualizer.create_summary_report(
    equity_curve=equity_df,
    trades=trades_df,
    metrics=results,
    feature_importance=importance_df,
    save_path="summary_report.png"
)
```

## 🎓 Examples

### Example 1: Basic Backtest

See `examples/simple_backtest.py` for a minimal working example.

### Example 2: Custom Strategy

```python
from algo_trading.strategies.ml_strategy import MLPredictionStrategy

# Create custom strategy
strategy = MLPredictionStrategy(
    model=your_trained_model,
    confidence_threshold=0.60,  # Higher threshold = fewer trades
    feature_columns=feature_cols,
)

# Generate signals with confidence scores
signals, confidences = strategy.generate_signals(
    test_data,
    return_confidence=True
)
```

### Example 3: Custom Position Sizing

```python
from algo_trading.risk.position_sizing import create_position_sizer

# Kelly Criterion sizing
position_sizer = create_position_sizer(
    method="kelly",
    kelly_fraction=0.25,  # Use 25% of Kelly (conservative)
)

# Confidence-based sizing
position_sizer = create_position_sizer(
    method="confidence_based",
    base_risk_fraction=0.01,  # 1% base risk
    max_risk_fraction=0.02,   # 2% max risk
    min_confidence=0.55,
)
```

## 🔧 Advanced Features

### Walk-Forward Optimization

```python
from algo_trading.models.trainer import ModelTrainer

trainer = ModelTrainer(model)

# Time-series cross-validation
cv_results = trainer.cross_validate(X, y, n_splits=5)

# Hyperparameter tuning
best_params, results = trainer.tune_hyperparameters(
    X, y,
    param_grid={'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
    method='grid'
)
```

### Threshold Optimization

```python
from algo_trading.strategies.ml_strategy import MLPredictionStrategy

strategy = MLPredictionStrategy(model, confidence_threshold=0.50)

# Optimize threshold on historical data
best_threshold, best_return = strategy.optimize_threshold(
    data=validation_data,
    returns=validation_returns,
    threshold_range=(0.50, 0.90),
    step=0.05
)
```

## 📚 Documentation

- `MODULE_DELIVERY.md`: Comprehensive module delivery report
- `INTEGRATION_SUMMARY.md`: Integration documentation
- `IMPROVEMENT_STATUS.md`: Improvement tracking
- Individual module docstrings: Google-style docstrings throughout

## 🐛 Known Issues & Limitations

**Resolved:**
- ✅ Data leakage in test set normalization
- ✅ Look-ahead bias in feature calculation
- ✅ Missing error handling
- ✅ Hardcoded magic numbers
- ✅ Missing classification metrics

**Current Limitations:**
- Single-asset backtesting only (multi-asset support planned)
- Long-only strategies (shorting available but not fully tested)
- Daily data only (intraday support planned)

## 🔄 Version History

### v2.0.0 (2025-11-15) - Production Release
- Modular package structure
- Fixed critical data leakage bugs
- Comprehensive risk management system
- Event-driven backtesting engine
- ML-based trading strategies
- Professional visualization suite
- Extensive unit tests
- Configuration management
- Full documentation

### v1.0.0 (Original)
- Monolithic Jupyter notebook
- Basic event-driven backtesting
- 5 ML models
- Limited error handling

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📧 Contact

For issues or questions:
- GitHub Issues: https://github.com/shiong-tan/test_project_oop_algo_trading/issues
- Review documentation in `/docs` folder

## 🙏 Acknowledgments

- Original requirements from PyAlgo course
- Data source: [Yves Hilpisch](http://hilpisch.com)
- Built with scikit-learn, pandas, matplotlib, and other open-source libraries

---

**Status**: Production Ready ✅

**Last Updated**: 2025-11-15
