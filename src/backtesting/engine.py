"""
src/backtesting/engine.py — Backtesting Engine

รัน simulation ของ strategy บน historical data:
- จำลองการเทรดจริง (commission, slippage)
- ติดตาม portfolio value
- บันทึก trades ทั้งหมด
- คำนวณ equity curve
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration สำหรับ Backtest"""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001    # 0.1%
    slippage_pct: float = 0.0005     # 0.05%
    risk_free_rate: float = 0.05     # 5% annual


class BacktestEngine:
    """
    Backtesting Engine สำหรับ Grid Trading

    จำลองการเทรดย้อนหลังบน historical data
    คำนวณ PnL, equity curve, และ risk metrics
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        risk_manager=None,
        config: BacktestConfig = None,
    ):
        """
        ตั้งค่าเริ่มต้น BacktestEngine

        Args:
            data: DataFrame ข้อมูล OHLCV พร้อม features
            strategy: Strategy object (BaseGrid หรือ AdaptiveGrid)
            risk_manager: Risk manager object (optional)
            config: Backtest configuration
        """
        self.data = data.copy()
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config or BacktestConfig()

        # Portfolio state
        self.capital = self.config.initial_capital
        self.position = 0.0
        self.equity_curve: List[float] = []
        self.trades: List[dict] = []
        self.daily_returns: List[float] = []

        # Results
        self.results: Optional[pd.DataFrame] = None
        self._is_run = False

    def run(
        self,
        start_date: str = None,
        end_date: str = None,
        ai_predictions: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        รัน Backtest

        Args:
            start_date: วันเริ่มต้น (YYYY-MM-DD)
            end_date: วันสิ้นสุด (YYYY-MM-DD)
            ai_predictions: AI predictions DataFrame (optional)

        Returns:
            DataFrame ของผลลัพธ์ทั้งหมด
        """
        # กรองข้อมูลตามช่วงวันที่
        data = self.data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        logger.info(
            f"เริ่ม Backtest: {len(data)} วัน "
            f"({data.index[0].date()} ถึง {data.index[-1].date()})"
        )

        # รัน strategy
        strategy_results = self.strategy.run(data, ai_predictions)

        # คำนวณ benchmark (Buy & Hold)
        benchmark = self._calculate_benchmark(data)

        # รวมผลลัพธ์
        results = strategy_results.copy()
        results["benchmark_equity"] = benchmark
        results["benchmark_return"] = (benchmark / self.config.initial_capital - 1)

        self.results = results
        self._is_run = True

        logger.info(
            f"Backtest เสร็จสิ้น: "
            f"Strategy return={results['return_pct'].iloc[-1]:.2%}, "
            f"Benchmark return={results['benchmark_return'].iloc[-1]:.2%}"
        )
        return results

    def _calculate_benchmark(self, data: pd.DataFrame) -> pd.Series:
        """
        คำนวณ Buy & Hold benchmark

        Args:
            data: DataFrame ข้อมูลราคา

        Returns:
            Series ของ equity curve แบบ buy & hold
        """
        first_price = data["Close"].iloc[0]
        shares = self.config.initial_capital / first_price
        benchmark = shares * data["Close"]
        return benchmark

    def execute_trades(
        self,
        signals: list,
        current_price: float,
        date: str,
    ) -> List[dict]:
        """
        Execute trades ตาม signals

        Args:
            signals: รายการ {'type': 'buy'/'sell', 'quantity': float, 'price': float}
            current_price: ราคาปัจจุบัน
            date: วันที่

        Returns:
            รายการ executed trades
        """
        executed = []

        for signal in signals:
            # Apply slippage
            if signal["type"] == "buy":
                exec_price = signal["price"] * (1 + self.config.slippage_pct)
            else:
                exec_price = signal["price"] * (1 - self.config.slippage_pct)

            # คำนวณ commission
            commission = exec_price * signal["quantity"] * self.config.commission_pct

            # อัปเดต portfolio
            if signal["type"] == "buy":
                cost = exec_price * signal["quantity"] + commission
                if cost <= self.capital:
                    self.capital -= cost
                    self.position += signal["quantity"]
                    trade = {
                        "date": date,
                        "type": "buy",
                        "price": exec_price,
                        "quantity": signal["quantity"],
                        "commission": commission,
                        "capital_after": self.capital,
                    }
                    self.trades.append(trade)
                    executed.append(trade)
            else:
                proceeds = exec_price * signal["quantity"] - commission
                self.capital += proceeds
                self.position = max(0, self.position - signal["quantity"])
                trade = {
                    "date": date,
                    "type": "sell",
                    "price": exec_price,
                    "quantity": signal["quantity"],
                    "commission": commission,
                    "capital_after": self.capital,
                }
                self.trades.append(trade)
                executed.append(trade)

        return executed

    def update_portfolio(
        self,
        current_price: float,
        date: str,
    ) -> float:
        """
        อัปเดตมูลค่า Portfolio

        Args:
            current_price: ราคา close วันนี้
            date: วันที่

        Returns:
            equity ปัจจุบัน
        """
        equity = self.capital + self.position * current_price
        self.equity_curve.append(equity)

        # คำนวณ daily return
        if len(self.equity_curve) > 1:
            daily_ret = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_ret)

        return equity

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        ดึงผลลัพธ์ Backtest

        Returns:
            DataFrame ของผลลัพธ์ หรือ None ถ้ายังไม่ได้รัน
        """
        if not self._is_run:
            logger.warning("ยังไม่ได้รัน backtest — กรุณาเรียก run() ก่อน")
            return None
        return self.results

    def get_trades_df(self) -> pd.DataFrame:
        """
        ดึง trades ทั้งหมดเป็น DataFrame

        Returns:
            DataFrame ของ trades
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
