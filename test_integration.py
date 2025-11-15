"""Integration test to verify all modules can be imported and instantiated."""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

print("Testing module imports...")

# Test imports
try:
    from algo_trading.data.loader import DataLoader
    from algo_trading.features.engineering import FeatureEngineer
    from algo_trading.models.trainer import ModelTrainer
    from algo_trading.risk.position_sizing import (
        create_position_sizer,
        FixedFractionalSizing,
        KellyCriterion,
        ConfidenceBasedSizing,
    )
    from algo_trading.risk.manager import RiskManager, Position
    from algo_trading.backtesting.engine import BacktestEngine, Order, OrderSide, OrderType
    from algo_trading.strategies.ml_strategy import MLPredictionStrategy
    from algo_trading.utils.config import Config
    from algo_trading.utils.logger import setup_logger
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\nTesting component instantiation...")

try:
    # Test DataLoader
    loader = DataLoader()
    print("✓ DataLoader created")

    # Test FeatureEngineer
    engineer = FeatureEngineer(sma_windows=(20, 60), n_lags=5)
    print("✓ FeatureEngineer created")

    # Test ModelTrainer
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    trainer = ModelTrainer(model, model_name="Test")
    print("✓ ModelTrainer created")

    # Test Position Sizers
    fixed_sizer = create_position_sizer("fixed_fractional", risk_fraction=0.01)
    kelly_sizer = create_position_sizer("kelly", kelly_fraction=0.25)
    conf_sizer = create_position_sizer("confidence_based", min_confidence=0.55)
    print("✓ Position sizers created")

    # Test RiskManager
    risk_mgr = RiskManager(initial_capital=10000)
    print("✓ RiskManager created")

    # Test BacktestEngine
    engine = BacktestEngine(
        initial_capital=10000,
        risk_manager=risk_mgr,
        position_sizer=fixed_sizer,
    )
    print("✓ BacktestEngine created")

    # Test MLPredictionStrategy (will be tested with real features later)
    # For now just test that we can create it
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y_train = pd.Series(np.random.randint(0, 2, 100))
    trainer.train(X_train, y_train)

    # Create strategy without feature columns (will use all numeric columns)
    strategy_dummy = MLPredictionStrategy(
        model=trainer.model,
        confidence_threshold=0.55,
    )
    print("✓ MLPredictionStrategy created")

    # Test Config
    config = Config()
    print("✓ Config created")

    # Test Logger
    logger = setup_logger("test", level="INFO", console=False)
    print("✓ Logger created")

except Exception as e:
    print(f"✗ Component instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting basic workflow...")

try:
    # Create synthetic price data
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    data = pd.DataFrame({"price": prices}, index=dates)

    # Create features
    features, feature_cols = engineer.fit_transform(data, price_col="price")
    print(f"✓ Created {len(feature_cols)} features from synthetic data")

    # Train model on engineered features
    trainer2 = ModelTrainer(
        RandomForestClassifier(n_estimators=10, random_state=42),
        model_name="Synthetic",
    )
    trainer2.train(features[feature_cols], features["direction"])
    print("✓ Model trained on synthetic data")

    # Create strategy with correct feature columns
    strategy = MLPredictionStrategy(
        model=trainer2.model,
        confidence_threshold=0.55,
        feature_columns=feature_cols,
    )

    # Generate signals
    signals, confidences = strategy.generate_signals(features, return_confidence=True)
    print(f"✓ Generated {len(signals)} signals")

    # Create minimal backtest
    backtest_data = data.copy()
    backtest_data.columns = ["Close"]

    # Use subset for quick test
    test_subset = backtest_data.iloc[-50:]
    signals_subset = signals.iloc[-50:]
    confidences_subset = confidences.iloc[-50:]

    # Run backtest
    engine2 = BacktestEngine(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_bps=5,
    )

    results = engine2.run_backtest(
        data=test_subset,
        signals=signals_subset,
        confidences=confidences_subset,
    )

    print(f"✓ Backtest completed successfully")
    print(f"  Initial capital: ${results['initial_capital']:,.2f}")
    print(f"  Final capital:   ${results['final_capital']:,.2f}")
    print(f"  Total return:    {results['total_return']:+.2%}")
    print(f"  Total trades:    {results['total_trades']}")

except Exception as e:
    print(f"✗ Workflow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED ✓")
print("=" * 60)
print("\nThe trading system is ready to use!")
print("\nNext steps:")
print("1. Run 'python main.py' for full backtest with real data")
print("2. Run 'python examples/simple_backtest.py' for a simple example")
print("3. Customize config.yaml for your trading strategy")

sys.exit(0)
