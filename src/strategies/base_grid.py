"""
src/strategies/base_grid.py — Classic Grid Trading Strategy

Grid Trading พื้นฐาน:
- คำนวณ Grid Levels จากราคาปัจจุบัน
- สร้าง Buy/Sell Orders ที่แต่ละ Grid Level
- ติดตามและอัปเดต Grid เมื่อมี execution
- คำนวณ PnL จาก closed trades
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """โครงสร้างข้อมูลสำหรับ Order"""
    order_id: int
    order_type: str          # 'buy' หรือ 'sell'
    price: float             # ราคา limit
    quantity: float          # จำนวนหน่วย
    status: str = "pending"  # 'pending', 'filled', 'cancelled'
    fill_price: float = 0.0
    fill_date: str = ""
    grid_level: int = 0


@dataclass
class Trade:
    """โครงสร้างข้อมูลสำหรับ Trade ที่สำเร็จ"""
    trade_id: int
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    direction: str  # 'long' หรือ 'short'
    pnl: float = 0.0
    pnl_pct: float = 0.0


class BaseGridStrategy:
    """
    Classic Grid Trading Strategy

    กลไก:
    1. สร้าง Grid levels รอบราคาปัจจุบัน
    2. วาง Buy orders ใต้ราคา, Sell orders เหนือราคา
    3. เมื่อ Buy ถูก fill → วาง Sell order เหนือขึ้นไป 1 grid
    4. เมื่อ Sell ถูก fill → วาง Buy order ต่ำลงมา 1 grid
    """

    def __init__(self, params: dict = None):
        """
        กำหนดค่าเริ่มต้น BaseGridStrategy

        Args:
            params: dict ของ strategy parameters
        """
        params = params or {}
        self.num_grids = params.get("num_grids", 10)
        self.grid_spacing_pct = params.get("grid_spacing_pct", 0.02)
        self.initial_capital = params.get("initial_capital", 100_000)
        self.position_size_pct = params.get("position_size_pct", 0.05)
        self.commission_pct = params.get("commission_pct", 0.001)

        # State
        self.capital = self.initial_capital
        self.position = 0.0          # จำนวนหน่วยที่ถืออยู่
        self.grid_levels = []        # ราคา Grid levels
        self.active_orders: Dict[int, Order] = {}
        self.filled_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self._order_counter = 0
        self._trade_counter = 0

    def calculate_grid_levels(
        self,
        current_price: float,
        num_grids: int = None,
        grid_spacing_pct: float = None,
    ) -> List[float]:
        """
        คำนวณ Grid levels รอบราคาปัจจุบัน

        Args:
            current_price: ราคาปัจจุบัน
            num_grids: จำนวน grids (ใช้ค่าจาก __init__ ถ้าไม่ระบุ)
            grid_spacing_pct: ระยะห่าง grid (%)

        Returns:
            รายการของราคา grid levels เรียงจากต่ำไปสูง
        """
        num_grids = num_grids or self.num_grids
        grid_spacing_pct = grid_spacing_pct or self.grid_spacing_pct

        # สร้าง levels รอบราคาปัจจุบัน
        half_grids = num_grids // 2
        levels = []

        for i in range(-half_grids, half_grids + 1):
            # ใช้ geometric spacing (ดีกว่า arithmetic สำหรับ % based)
            level_price = current_price * (1 + grid_spacing_pct) ** i
            levels.append(round(level_price, 4))

        self.grid_levels = sorted(set(levels))
        logger.debug(
            f"Grid levels: {len(self.grid_levels)} levels, "
            f"range [{self.grid_levels[0]:.2f}, {self.grid_levels[-1]:.2f}]"
        )
        return self.grid_levels

    def generate_orders(
        self,
        grid_levels: List[float],
        current_price: float,
        position_size: float = None,
    ) -> List[Order]:
        """
        สร้าง Buy/Sell orders สำหรับแต่ละ Grid level

        Args:
            grid_levels: รายการ grid levels
            current_price: ราคาปัจจุบัน
            position_size: จำนวนเงินต่อ order (USD)

        Returns:
            รายการ Order objects
        """
        if position_size is None:
            position_size = self.capital * self.position_size_pct

        orders = []
        for level in grid_levels:
            if level == current_price:
                continue

            self._order_counter += 1
            order_type = "buy" if level < current_price else "sell"
            quantity = position_size / level  # จำนวนหน่วย

            order = Order(
                order_id=self._order_counter,
                order_type=order_type,
                price=level,
                quantity=quantity,
            )
            orders.append(order)
            self.active_orders[order.order_id] = order

        logger.debug(f"สร้าง {len(orders)} orders ({sum(1 for o in orders if o.order_type=='buy')} buy, {sum(1 for o in orders if o.order_type=='sell')} sell)")
        return orders

    def check_order_execution(
        self, current_price: float, date: str
    ) -> List[Order]:
        """
        ตรวจสอบว่า order ถูก execute ที่ราคาปัจจุบันหรือไม่

        Args:
            current_price: ราคาปัจจุบัน
            date: วันที่ปัจจุบัน

        Returns:
            รายการ orders ที่ถูก execute
        """
        executed = []

        for order_id, order in list(self.active_orders.items()):
            filled = False

            if order.order_type == "buy" and current_price <= order.price:
                filled = True
            elif order.order_type == "sell" and current_price >= order.price:
                filled = True

            if filled:
                # Apply slippage เล็กน้อย
                order.fill_price = order.price
                order.fill_date = date
                order.status = "filled"

                # คำนวณค่าคอมมิสชัน
                commission = order.fill_price * order.quantity * self.commission_pct

                if order.order_type == "buy":
                    self.capital -= order.fill_price * order.quantity + commission
                    self.position += order.quantity
                else:
                    self.capital += order.fill_price * order.quantity - commission
                    self.position -= order.quantity

                self.filled_orders.append(order)
                del self.active_orders[order_id]
                executed.append(order)

        return executed

    def update_grid(
        self,
        current_price: float,
        executed_orders: List[Order],
    ) -> None:
        """
        อัปเดต Grid หลังจาก order execution

        เมื่อ Buy fill → วาง Sell order 1 grid เหนือขึ้นไป
        เมื่อ Sell fill → วาง Buy order 1 grid ต่ำลงมา

        Args:
            current_price: ราคาปัจจุบัน
            executed_orders: orders ที่เพิ่งถูก execute
        """
        position_size = self.capital * self.position_size_pct

        for order in executed_orders:
            if order.order_type == "buy":
                # วาง Sell order 1 grid เหนือขึ้นไป
                sell_price = order.fill_price * (1 + self.grid_spacing_pct)
                self._order_counter += 1
                new_order = Order(
                    order_id=self._order_counter,
                    order_type="sell",
                    price=sell_price,
                    quantity=order.quantity,
                    grid_level=order.grid_level + 1,
                )
                self.active_orders[new_order.order_id] = new_order

            elif order.order_type == "sell":
                # วาง Buy order 1 grid ต่ำลงมา
                buy_price = order.fill_price / (1 + self.grid_spacing_pct)
                self._order_counter += 1
                new_order = Order(
                    order_id=self._order_counter,
                    order_type="buy",
                    price=buy_price,
                    quantity=position_size / buy_price,
                    grid_level=order.grid_level - 1,
                )
                self.active_orders[new_order.order_id] = new_order

    def calculate_pnl(self, current_price: float = None) -> dict:
        """
        คำนวณกำไรขาดทุนสะสม

        Args:
            current_price: ราคาปัจจุบันสำหรับ mark-to-market

        Returns:
            dict ของ PnL metrics
        """
        # Realized PnL จาก filled orders
        realized_pnl = 0.0
        buy_orders = [o for o in self.filled_orders if o.order_type == "buy"]
        sell_orders = [o for o in self.filled_orders if o.order_type == "sell"]

        # จับคู่ buy-sell
        min_pairs = min(len(buy_orders), len(sell_orders))
        for i in range(min_pairs):
            buy = buy_orders[i]
            sell = sell_orders[i]
            trade_qty = min(buy.quantity, sell.quantity)
            realized_pnl += (sell.fill_price - buy.fill_price) * trade_qty

        # Unrealized PnL
        unrealized_pnl = 0.0
        if current_price is not None and self.position > 0:
            # คำนวณ average entry price
            buy_cost = sum(o.fill_price * o.quantity for o in buy_orders)
            buy_qty = sum(o.quantity for o in buy_orders)
            avg_entry = buy_cost / buy_qty if buy_qty > 0 else 0
            unrealized_pnl = (current_price - avg_entry) * self.position

        total_pnl = realized_pnl + unrealized_pnl
        current_equity = self.initial_capital + total_pnl + (self.capital - self.initial_capital)

        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "capital": self.capital,
            "position": self.position,
            "equity": current_equity,
            "return_pct": (current_equity - self.initial_capital) / self.initial_capital,
        }

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        รัน Classic Grid Strategy บน historical data

        Args:
            df: DataFrame ที่มีคอลัมน์ OHLCV

        Returns:
            DataFrame ของผลลัพธ์ (equity curve, trades, etc.)
        """
        results = []
        logger.info(f"เริ่มรัน Classic Grid Strategy บน {len(df)} วัน")

        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row["Close"]

            # Initialize grid ในวันแรก
            if i == 0:
                self.calculate_grid_levels(current_price)
                self.generate_orders(self.grid_levels, current_price)

            # ตรวจสอบ order execution
            executed = self.check_order_execution(current_price, str(date))

            # อัปเดต grid หลัง execution
            if executed:
                self.update_grid(current_price, executed)

            # คำนวณ PnL
            pnl_data = self.calculate_pnl(current_price)
            pnl_data["date"] = date
            pnl_data["price"] = current_price
            pnl_data["active_orders"] = len(self.active_orders)
            results.append(pnl_data)

        results_df = pd.DataFrame(results).set_index("date")
        logger.info(
            f"Classic Grid เสร็จสิ้น: "
            f"Final equity={results_df['equity'].iloc[-1]:,.2f}, "
            f"Return={results_df['return_pct'].iloc[-1]:.2%}"
        )
        return results_df
