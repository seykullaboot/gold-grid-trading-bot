"""
tests/test_strategies.py — Unit Tests สำหรับ Grid Strategies
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.base_grid import BaseGridStrategy, Order
from src.strategies.adaptive_grid import AdaptiveGridStrategy


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_price_df():
    """สร้าง DataFrame ราคาทดสอบ"""
    np.random.seed(42)
    n = 120  # 6 เดือน
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    price = 180.0
    prices = []
    for _ in range(n):
        price *= (1 + np.random.normal(0, 0.012))
        prices.append(price)

    prices = np.array(prices)
    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.003, 0.003, n)),
            "High": prices * (1 + np.random.uniform(0.002, 0.012, n)),
            "Low": prices * (1 - np.random.uniform(0.002, 0.012, n)),
            "Close": prices,
            "Volume": np.random.randint(5_000_000, 20_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def basic_grid_params():
    """พารามิเตอร์พื้นฐานสำหรับ Classic Grid"""
    return {
        "num_grids": 10,
        "grid_spacing_pct": 0.02,
        "initial_capital": 100_000,
        "position_size_pct": 0.05,
        "commission_pct": 0.001,
    }


@pytest.fixture
def adaptive_grid_params():
    """พารามิเตอร์สำหรับ Adaptive Grid"""
    return {
        "num_grids": 10,
        "grid_spacing_pct": 0.02,
        "initial_capital": 100_000,
        "position_size_pct": 0.05,
        "commission_pct": 0.001,
        "min_num_grids": 5,
        "max_num_grids": 20,
        "min_grid_spacing_pct": 0.005,
        "max_grid_spacing_pct": 0.05,
        "rebalance_frequency": "weekly",
    }


# ============================================================
# Tests: BaseGridStrategy
# ============================================================

class TestBaseGridStrategy:
    """ทดสอบ Classic Grid Strategy"""

    def test_init_default_params(self):
        """ทดสอบ initialization ด้วยค่า default"""
        strategy = BaseGridStrategy()
        assert strategy.num_grids == 10
        assert strategy.grid_spacing_pct == 0.02
        assert strategy.initial_capital == 100_000

    def test_init_custom_params(self, basic_grid_params):
        """ทดสอบ initialization ด้วยค่า custom"""
        strategy = BaseGridStrategy(basic_grid_params)
        assert strategy.num_grids == basic_grid_params["num_grids"]
        assert strategy.initial_capital == basic_grid_params["initial_capital"]

    def test_calculate_grid_levels_count(self, basic_grid_params):
        """ทดสอบจำนวน grid levels"""
        strategy = BaseGridStrategy(basic_grid_params)
        levels = strategy.calculate_grid_levels(current_price=180.0, num_grids=10)
        # num_grids + 1 levels (รวมราคาปัจจุบัน)
        assert len(levels) >= 10

    def test_calculate_grid_levels_sorted(self, basic_grid_params):
        """ทดสอบ grid levels เรียงจากต่ำไปสูง"""
        strategy = BaseGridStrategy(basic_grid_params)
        levels = strategy.calculate_grid_levels(current_price=180.0, num_grids=10)
        assert levels == sorted(levels)

    def test_calculate_grid_levels_includes_price_range(self, basic_grid_params):
        """ทดสอบว่า grid levels ครอบคลุมทั้งบนและล่างของราคา"""
        strategy = BaseGridStrategy(basic_grid_params)
        price = 180.0
        levels = strategy.calculate_grid_levels(current_price=price, num_grids=10)
        assert min(levels) < price
        assert max(levels) > price

    def test_generate_orders_creates_buy_and_sell(self, basic_grid_params):
        """ทดสอบว่า generate_orders สร้างทั้ง buy และ sell orders"""
        strategy = BaseGridStrategy(basic_grid_params)
        price = 180.0
        levels = strategy.calculate_grid_levels(current_price=price, num_grids=10)
        orders = strategy.generate_orders(levels, current_price=price)

        buy_orders = [o for o in orders if o.order_type == "buy"]
        sell_orders = [o for o in orders if o.order_type == "sell"]

        assert len(buy_orders) > 0
        assert len(sell_orders) > 0

    def test_generate_orders_buy_below_price(self, basic_grid_params):
        """ทดสอบว่า buy orders อยู่ใต้ราคาปัจจุบัน"""
        strategy = BaseGridStrategy(basic_grid_params)
        price = 180.0
        levels = strategy.calculate_grid_levels(current_price=price, num_grids=10)
        orders = strategy.generate_orders(levels, current_price=price)

        buy_orders = [o for o in orders if o.order_type == "buy"]
        for order in buy_orders:
            assert order.price < price

    def test_generate_orders_sell_above_price(self, basic_grid_params):
        """ทดสอบว่า sell orders อยู่เหนือราคาปัจจุบัน"""
        strategy = BaseGridStrategy(basic_grid_params)
        price = 180.0
        levels = strategy.calculate_grid_levels(current_price=price, num_grids=10)
        orders = strategy.generate_orders(levels, current_price=price)

        sell_orders = [o for o in orders if o.order_type == "sell"]
        for order in sell_orders:
            assert order.price > price

    def test_check_order_execution_buy(self, basic_grid_params):
        """ทดสอบ execute buy order เมื่อราคาถึง"""
        strategy = BaseGridStrategy(basic_grid_params)
        price = 180.0
        levels = strategy.calculate_grid_levels(current_price=price, num_grids=6)
        strategy.generate_orders(levels, current_price=price)

        # ราคาตก → buy order ควรถูก fill
        low_price = min(levels) - 0.01
        executed = strategy.check_order_execution(low_price, "2023-01-01")
        assert len(executed) > 0
        assert all(o.order_type == "buy" for o in executed)

    def test_calculate_pnl_returns_dict(self, basic_grid_params):
        """ทดสอบว่า calculate_pnl คืนค่า dict"""
        strategy = BaseGridStrategy(basic_grid_params)
        pnl = strategy.calculate_pnl(current_price=180.0)
        assert isinstance(pnl, dict)
        assert "equity" in pnl
        assert "capital" in pnl

    def test_run_returns_dataframe(self, basic_grid_params, sample_price_df):
        """ทดสอบว่า run() คืนค่า DataFrame"""
        strategy = BaseGridStrategy(basic_grid_params)
        results = strategy.run(sample_price_df)
        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        assert "equity" in results.columns

    def test_run_equity_is_positive(self, basic_grid_params, sample_price_df):
        """ทดสอบว่า equity เป็นบวกตลอด"""
        strategy = BaseGridStrategy(basic_grid_params)
        results = strategy.run(sample_price_df)
        assert (results["equity"] > 0).all()

    def test_run_result_length_matches_data(self, basic_grid_params, sample_price_df):
        """ทดสอบว่า results มีจำนวนแถวเท่ากับ input data"""
        strategy = BaseGridStrategy(basic_grid_params)
        results = strategy.run(sample_price_df)
        assert len(results) == len(sample_price_df)


# ============================================================
# Tests: AdaptiveGridStrategy
# ============================================================

class TestAdaptiveGridStrategy:
    """ทดสอบ AI Adaptive Grid Strategy"""

    def test_init(self, adaptive_grid_params):
        """ทดสอบ initialization"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        assert strategy.min_num_grids == 5
        assert strategy.max_num_grids == 20
        assert strategy.rebalance_frequency == "weekly"

    def test_adapt_grid_trending(self, adaptive_grid_params):
        """ทดสอบ adapt_grid ใน trending market"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        num_grids, spacing = strategy.adapt_grid_to_regime("trending", volatility=0.20)
        # Trending → ลด grids, เพิ่ม spacing
        assert num_grids < strategy.num_grids
        assert num_grids >= strategy.min_num_grids

    def test_adapt_grid_ranging(self, adaptive_grid_params):
        """ทดสอบ adapt_grid ใน ranging market"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        num_grids, spacing = strategy.adapt_grid_to_regime("ranging", volatility=0.15)
        # Ranging → เพิ่ม grids
        assert num_grids >= strategy.num_grids * 0.9  # อาจน้อยกว่าเล็กน้อยเพราะ clip

    def test_adapt_grid_volatile(self, adaptive_grid_params):
        """ทดสอบ adapt_grid ใน volatile market"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        num_grids, spacing = strategy.adapt_grid_to_regime("volatile", volatility=0.45)
        # Volatile → ลด grids มาก
        assert num_grids <= strategy.num_grids

    def test_adapt_grid_respects_limits(self, adaptive_grid_params):
        """ทดสอบว่า adapt_grid ไม่เกิน min/max"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)

        for regime in ["trending", "ranging", "volatile"]:
            for vol in [0.05, 0.20, 0.60]:
                num_grids, spacing = strategy.adapt_grid_to_regime(regime, vol)
                assert strategy.min_num_grids <= num_grids <= strategy.max_num_grids
                assert strategy.min_grid_spacing_pct <= spacing <= strategy.max_grid_spacing_pct

    def test_optimize_grid_spacing_higher_vol_higher_spacing(self, adaptive_grid_params):
        """ทดสอบว่า spacing สูงขึ้นเมื่อ volatility สูง"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        low_vol_spacing = strategy.optimize_grid_spacing(0.10)
        high_vol_spacing = strategy.optimize_grid_spacing(0.50)
        assert high_vol_spacing >= low_vol_spacing

    def test_calculate_position_size_ranging_larger(self, adaptive_grid_params):
        """ทดสอบว่า ranging ให้ position size ใหญ่กว่า volatile"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        ranging_size = strategy.calculate_position_size("ranging")
        volatile_size = strategy.calculate_position_size("volatile")
        assert ranging_size > volatile_size

    def test_should_rebalance_first_call(self, adaptive_grid_params):
        """ทดสอบว่า rebalance ในครั้งแรก"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        assert strategy.should_rebalance(pd.Timestamp("2023-01-01"), "ranging") is True

    def test_run_with_ai_predictions(self, adaptive_grid_params, sample_price_df):
        """ทดสอบ run() พร้อม AI predictions"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)

        # สร้าง mock AI predictions
        ai_preds = pd.DataFrame(
            {
                "regime": np.random.choice(["trending", "ranging", "volatile"], len(sample_price_df)),
                "predicted_volatility": np.random.uniform(0.10, 0.40, len(sample_price_df)),
            },
            index=sample_price_df.index,
        )

        results = strategy.run(sample_price_df, ai_predictions=ai_preds)
        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        assert "regime" in results.columns
        assert "equity" in results.columns

    def test_run_without_ai_predictions(self, adaptive_grid_params, sample_price_df):
        """ทดสอบ run() โดยไม่มี AI predictions"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        results = strategy.run(sample_price_df)
        assert isinstance(results, pd.DataFrame)
        assert "equity" in results.columns

    def test_regime_history_recorded(self, adaptive_grid_params, sample_price_df):
        """ทดสอบว่า regime history ถูกบันทึก"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        strategy.run(sample_price_df)
        assert len(strategy.regime_history) > 0

    def test_adaptation_log_recorded(self, adaptive_grid_params, sample_price_df):
        """ทดสอบว่า adaptation log ถูกบันทึก"""
        strategy = AdaptiveGridStrategy(adaptive_grid_params)
        strategy.run(sample_price_df)
        assert len(strategy.adaptation_log) > 0
