"""Quick test script to verify all modules can be imported and basic functionality works."""

import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")

    try:
        from algo_trading.risk import (
            FixedFractionalSizing,
            KellyCriterion,
            ConfidenceBasedSizing,
            RiskManager,
            create_position_sizer,
        )
        logger.info("✓ Risk module imports successful")
    except Exception as e:
        logger.error(f"✗ Risk module import failed: {e}")
        return False

    try:
        from algo_trading.backtesting import BacktestEngine, Order, OrderType, OrderSide
        logger.info("✓ Backtesting module imports successful")
    except Exception as e:
        logger.error(f"✗ Backtesting module import failed: {e}")
        return False

    try:
        from algo_trading.strategies import MLPredictionStrategy
        logger.info("✓ Strategies module imports successful")
    except Exception as e:
        logger.error(f"✗ Strategies module import failed: {e}")
        return False

    return True

def test_position_sizing():
    """Test position sizing strategies."""
    logger.info("\nTesting position sizing strategies...")

    from algo_trading.risk import (
        FixedFractionalSizing,
        KellyCriterion,
        ConfidenceBasedSizing,
    )

    try:
        # Test FixedFractionalSizing
        sizer = FixedFractionalSizing(risk_fraction=0.01)
        size = sizer.calculate_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=97,
        )
        logger.info(f"✓ FixedFractionalSizing: position size = {size:.2f}")

        # Test KellyCriterion
        kelly = KellyCriterion(kelly_fraction=0.25)
        size = kelly.calculate_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=97,
            win_rate=0.6,
            avg_win=300,
            avg_loss=100,
        )
        logger.info(f"✓ KellyCriterion: position size = {size:.2f}")

        # Test ConfidenceBasedSizing
        conf_sizer = ConfidenceBasedSizing(base_risk_fraction=0.01, max_risk_fraction=0.02)
        size = conf_sizer.calculate_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=97,
            confidence=0.75,
        )
        logger.info(f"✓ ConfidenceBasedSizing: position size = {size:.2f}")

        return True
    except Exception as e:
        logger.error(f"✗ Position sizing test failed: {e}")
        return False

def test_risk_manager():
    """Test RiskManager."""
    logger.info("\nTesting RiskManager...")

    from algo_trading.risk import RiskManager

    try:
        # Create risk manager
        rm = RiskManager(initial_capital=10000)
        logger.info(f"✓ RiskManager created with capital ${rm.initial_capital:,.2f}")

        # Test trade validation
        is_valid, issues = rm.validate_trade(
            symbol="AAPL",
            quantity=10,
            entry_price=100,
            stop_loss=97,
            confidence=0.65,
        )
        logger.info(f"✓ Trade validation: valid={is_valid}, issues={issues}")

        # Open a position
        rm.open_position(
            symbol="AAPL",
            entry_price=100,
            quantity=10,
            entry_date=datetime.now(),
        )
        logger.info(f"✓ Position opened: {len(rm.open_positions)} open positions")

        # Get risk metrics
        metrics = rm.get_risk_metrics()
        logger.info(f"✓ Risk metrics: drawdown={metrics['current_drawdown']:.2%}, heat={metrics['portfolio_heat']:.2%}")

        return True
    except Exception as e:
        logger.error(f"✗ RiskManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_engine():
    """Test BacktestEngine."""
    logger.info("\nTesting BacktestEngine...")

    from algo_trading.backtesting import BacktestEngine

    try:
        # Create backtest engine
        engine = BacktestEngine(initial_capital=10000)
        logger.info(f"✓ BacktestEngine created with capital ${engine.initial_capital:,.2f}")

        logger.info(f"✓ Portfolio value: ${engine.portfolio_value:,.2f}")

        return True
    except Exception as e:
        logger.error(f"✗ BacktestEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running module tests...")
    logger.info("=" * 60)

    results = []

    results.append(("Imports", test_imports()))

    if results[0][1]:  # Only run other tests if imports work
        results.append(("Position Sizing", test_position_sizing()))
        results.append(("Risk Manager", test_risk_manager()))
        results.append(("Backtest Engine", test_backtest_engine()))

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
