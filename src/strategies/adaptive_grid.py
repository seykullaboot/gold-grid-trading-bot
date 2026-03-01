"""
src/strategies/adaptive_grid.py — AI Adaptive Grid Trading Strategy

ปรับ Grid Parameters อัตโนมัติตาม:
- Market Regime (Trending / Ranging / Volatile)
- Volatility Prediction จาก LSTM
- Risk Parameters
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_grid import BaseGridStrategy, Order

logger = logging.getLogger(__name__)

# ตาราง multipliers ตาม regime
REGIME_MULTIPLIERS = {
    "trending": {
        "num_grids": 0.7,
        "spacing": 1.5,
        "position_size": 0.8,
    },
    "ranging": {
        "num_grids": 1.2,
        "spacing": 0.8,
        "position_size": 1.2,
    },
    "volatile": {
        "num_grids": 0.5,
        "spacing": 2.0,
        "position_size": 0.5,
    },
}


class AdaptiveGridStrategy(BaseGridStrategy):
    """
    AI Adaptive Grid Trading Strategy

    ต่อยอดจาก BaseGridStrategy โดย:
    1. ใช้ Market Regime เพื่อปรับจำนวน Grid
    2. ใช้ Volatility Prediction เพื่อปรับ Grid Spacing
    3. ปรับ Position Size ตาม Regime และ Risk
    4. Rebalance Grid เป็นระยะตาม regime changes
    """

    def __init__(self, params: dict = None):
        """
        กำหนดค่าเริ่มต้น AdaptiveGridStrategy

        Args:
            params: dict ของ strategy parameters (รวม adaptive params)
        """
        params = params or {}
        super().__init__(params)

        # Adaptive parameters
        self.min_num_grids = params.get("min_num_grids", 5)
        self.max_num_grids = params.get("max_num_grids", 20)
        self.min_grid_spacing_pct = params.get("min_grid_spacing_pct", 0.005)
        self.max_grid_spacing_pct = params.get("max_grid_spacing_pct", 0.05)
        self.rebalance_frequency = params.get("rebalance_frequency", "weekly")
        self.regime_multipliers = params.get("regime_multipliers", REGIME_MULTIPLIERS)

        # State tracking
        self.current_regime = "ranging"
        self.current_volatility = 0.2
        self.last_rebalance_date = None
        self.regime_history = []
        self.adaptation_log = []

    def adapt_grid_to_regime(
        self,
        regime: str,
        volatility: float,
    ) -> Tuple[int, float]:
        """
        ปรับพารามิเตอร์ Grid ตาม Market Regime และ Volatility

        Logic:
        - Trending: ลด grids + เพิ่ม spacing (ราคาวิ่งไกล ต้องห่างขึ้น)
        - Ranging: เพิ่ม grids + ลด spacing (ราคาวนอยู่ใน range)
        - Volatile: ลด grids มาก + เพิ่ม spacing มาก (ป้องกันความเสี่ยง)

        Args:
            regime: สภาวะตลาด ('trending', 'ranging', 'volatile')
            volatility: ค่าความผันผวน (annualized)

        Returns:
            tuple (num_grids, grid_spacing_pct)
        """
        multipliers = self.regime_multipliers.get(regime, REGIME_MULTIPLIERS["ranging"])

        # คำนวณ num_grids
        base_grids = self.num_grids
        adj_grids = int(base_grids * multipliers["num_grids"])
        adj_grids = np.clip(adj_grids, self.min_num_grids, self.max_num_grids)

        # คำนวณ spacing จาก volatility
        # Spacing = volatility รายวัน * factor
        daily_vol = volatility / np.sqrt(252)
        base_spacing = self.grid_spacing_pct
        vol_adj_spacing = daily_vol * 2  # 2x daily volatility

        # ปรับตาม regime multiplier
        adj_spacing = max(base_spacing, vol_adj_spacing) * multipliers["spacing"]
        adj_spacing = np.clip(adj_spacing, self.min_grid_spacing_pct, self.max_grid_spacing_pct)

        logger.debug(
            f"Adapt Grid → Regime: {regime}, Vol: {volatility:.3f}, "
            f"Grids: {self.num_grids}→{adj_grids}, "
            f"Spacing: {self.grid_spacing_pct:.3f}→{adj_spacing:.3f}"
        )

        # บันทึก adaptation
        self.adaptation_log.append({
            "regime": regime,
            "volatility": volatility,
            "num_grids": adj_grids,
            "grid_spacing": adj_spacing,
        })

        self.current_regime = regime
        self.current_volatility = volatility

        return adj_grids, adj_spacing

    def optimize_grid_spacing(
        self,
        volatility: float,
        lookback_vol: float = None,
    ) -> float:
        """
        ปรับ Grid Spacing ตาม Volatility อย่างละเอียด

        ใช้ ATR-based spacing:
        spacing = vol_daily * ATR_multiplier

        Args:
            volatility: ความผันผวน annualized
            lookback_vol: ความผันผวน lookback period (optional)

        Returns:
            optimal grid spacing (%)
        """
        # แปลง annualized volatility → daily
        daily_vol = volatility / np.sqrt(252)

        # ใช้ volatility ratio ถ้ามี lookback
        if lookback_vol is not None and lookback_vol > 0:
            vol_ratio = volatility / lookback_vol
            # ความผันผวนสูงกว่าปกติ → เพิ่ม spacing
            adjustment_factor = np.clip(vol_ratio, 0.5, 2.0)
        else:
            adjustment_factor = 1.0

        # คำนวณ optimal spacing
        optimal_spacing = daily_vol * 2 * adjustment_factor

        # จำกัดในช่วงที่กำหนด
        optimal_spacing = np.clip(
            optimal_spacing,
            self.min_grid_spacing_pct,
            self.max_grid_spacing_pct,
        )

        return float(optimal_spacing)

    def calculate_position_size(
        self,
        regime: str,
        risk_params: dict = None,
    ) -> float:
        """
        คำนวณขนาด Position ต่อ Grid ตาม Regime และ Risk

        Args:
            regime: สภาวะตลาด
            risk_params: dict ของ risk parameters

        Returns:
            position size (% ของ capital)
        """
        risk_params = risk_params or {}
        base_size = self.position_size_pct

        # ปรับตาม regime
        multipliers = self.regime_multipliers.get(regime, REGIME_MULTIPLIERS["ranging"])
        regime_adj_size = base_size * multipliers["position_size"]

        # ปรับตาม current drawdown
        max_dd = risk_params.get("max_drawdown_pct", 0.15)
        current_equity = self.capital + self.position * self.current_volatility
        current_dd = (self.initial_capital - current_equity) / self.initial_capital

        if current_dd > max_dd * 0.7:
            # เมื่อ drawdown ใกล้ limit → ลด position ลง 50%
            regime_adj_size *= 0.5
            logger.warning(f"ลด position size เพราะ drawdown {current_dd:.2%}")

        # จำกัดขนาด
        max_size = risk_params.get("max_position_pct", 0.20)
        final_size = min(regime_adj_size, max_size)

        return float(final_size)

    def should_rebalance(self, current_date, regime: str) -> bool:
        """
        ตรวจสอบว่าควร rebalance grid หรือไม่

        Args:
            current_date: วันที่ปัจจุบัน
            regime: regime ปัจจุบัน

        Returns:
            True ถ้าควร rebalance
        """
        # Rebalance เมื่อ regime เปลี่ยน
        if regime != self.current_regime:
            logger.info(f"Regime เปลี่ยน: {self.current_regime} → {regime}, จะ rebalance")
            return True

        # Rebalance ตาม frequency
        if self.last_rebalance_date is None:
            return True

        if isinstance(current_date, str):
            current_date = pd.Timestamp(current_date)
        if isinstance(self.last_rebalance_date, str):
            self.last_rebalance_date = pd.Timestamp(self.last_rebalance_date)

        days_since_rebalance = (current_date - self.last_rebalance_date).days

        if self.rebalance_frequency == "daily":
            return days_since_rebalance >= 1
        elif self.rebalance_frequency == "weekly":
            return days_since_rebalance >= 7
        elif self.rebalance_frequency == "monthly":
            return days_since_rebalance >= 30

        return False

    def run(
        self,
        df: pd.DataFrame,
        ai_predictions: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        รัน Adaptive Grid Strategy บน historical data

        Args:
            df: DataFrame ที่มีคอลัมน์ OHLCV
            ai_predictions: DataFrame ที่มี 'regime' และ 'predicted_volatility'
                           (ถ้าไม่มี จะใช้ค่า default)

        Returns:
            DataFrame ของผลลัพธ์ (equity curve, trades, metrics)
        """
        results = []
        logger.info(
            f"เริ่มรัน Adaptive Grid Strategy บน {len(df)} วัน"
            + (" พร้อม AI predictions" if ai_predictions is not None else " (ไม่มี AI)")
        )

        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row["Close"]

            # ดึง AI predictions
            if ai_predictions is not None and date in ai_predictions.index:
                regime = ai_predictions.loc[date, "regime"] if "regime" in ai_predictions.columns else "ranging"
                volatility = ai_predictions.loc[date, "predicted_volatility"] if "predicted_volatility" in ai_predictions.columns else 0.2
            else:
                # ใช้ค่า default ถ้าไม่มี AI predictions
                regime = "ranging"
                volatility = row.get("hist_vol_20d", 0.2) if "hist_vol_20d" in row.index else 0.2

            # ปรับ grid parameters ตาม AI
            adj_grids, adj_spacing = self.adapt_grid_to_regime(regime, volatility)

            # Rebalance grid ถ้าจำเป็น
            if self.should_rebalance(date, regime) or i == 0:
                # ยกเลิก active orders เก่า
                self.active_orders.clear()

                # สร้าง grid ใหม่ด้วย adaptive parameters
                grid_levels = self.calculate_grid_levels(current_price, adj_grids, adj_spacing)

                # คำนวณ position size
                position_size = self.capital * self.calculate_position_size(regime)

                self.generate_orders(grid_levels, current_price, position_size)
                self.last_rebalance_date = date
                self.grid_spacing_pct = adj_spacing

            # ตรวจสอบ order execution
            executed = self.check_order_execution(current_price, str(date))

            # อัปเดต grid หลัง execution
            if executed:
                self.update_grid(current_price, executed)

            # คำนวณ PnL
            pnl_data = self.calculate_pnl(current_price)
            pnl_data["date"] = date
            pnl_data["price"] = current_price
            pnl_data["regime"] = regime
            pnl_data["volatility"] = volatility
            pnl_data["num_grids"] = adj_grids
            pnl_data["grid_spacing"] = adj_spacing
            pnl_data["active_orders"] = len(self.active_orders)
            results.append(pnl_data)

            # บันทึก regime history
            self.regime_history.append({"date": date, "regime": regime})

        results_df = pd.DataFrame(results).set_index("date")
        logger.info(
            f"Adaptive Grid เสร็จสิ้น: "
            f"Final equity={results_df['equity'].iloc[-1]:,.2f}, "
            f"Return={results_df['return_pct'].iloc[-1]:.2%}"
        )
        return results_df
