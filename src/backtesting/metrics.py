"""
src/backtesting/metrics.py — Performance Metrics Calculator

คำนวณ metrics ครบชุดสำหรับประเมิน strategy:
- Total Return, CAGR
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Recovery Factor
- Win Rate, Profit Factor
- Alpha, Beta
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    คลาสคำนวณ Performance Metrics ครบชุด

    ใช้กับ equity curve และ trade list
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        กำหนดค่าเริ่มต้น

        Args:
            risk_free_rate: อัตราดอกเบี้ยปลอดความเสี่ยง (annual)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252  # แปลงเป็น daily

    def calculate_total_return(self, equity_curve: pd.Series) -> float:
        """
        คำนวณ Total Return

        Args:
            equity_curve: Series ของ equity values

        Returns:
            Total return (เช่น 0.35 = 35%)
        """
        if len(equity_curve) < 2:
            return 0.0
        return float((equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0])

    def calculate_cagr(self, equity_curve: pd.Series) -> float:
        """
        คำนวณ Compound Annual Growth Rate (CAGR)

        Args:
            equity_curve: Series ของ equity values

        Returns:
            CAGR (annual)
        """
        if len(equity_curve) < 2:
            return 0.0

        total_return = self.calculate_total_return(equity_curve)
        n_years = len(equity_curve) / 252  # สมมติ 252 trading days/year

        if n_years <= 0:
            return 0.0

        cagr = (1 + total_return) ** (1 / n_years) - 1
        return float(cagr)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None,
        annualize: bool = True,
    ) -> float:
        """
        คำนวณ Sharpe Ratio

        Formula: (Mean Return - Risk-Free Rate) / Std Dev

        Args:
            returns: Series ของ daily returns
            risk_free_rate: อัตราดอกเบี้ยปลอดความเสี่ยง (ถ้าไม่ระบุใช้ค่า init)
            annualize: แปลงเป็น annual หรือไม่

        Returns:
            Sharpe Ratio
        """
        rf = (risk_free_rate or self.risk_free_rate) / 252

        excess_returns = returns - rf
        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std()

        if annualize:
            sharpe *= np.sqrt(252)

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """
        คำนวณ Sortino Ratio (ใช้เฉพาะ downside deviation)

        Args:
            returns: Series ของ daily returns
            annualize: แปลงเป็น annual หรือไม่

        Returns:
            Sortino Ratio
        """
        rf_daily = self.risk_free_rate / 252
        excess_returns = returns - rf_daily

        # Downside deviation — เฉพาะ returns ที่ต่ำกว่า risk-free rate
        downside = excess_returns[excess_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf")

        sortino = excess_returns.mean() / downside.std()

        if annualize:
            sortino *= np.sqrt(252)

        return float(sortino)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> tuple:
        """
        คำนวณ Maximum Drawdown

        Args:
            equity_curve: Series ของ equity values

        Returns:
            tuple (max_drawdown, drawdown_series, drawdown_duration_days)
        """
        # คำนวณ rolling maximum (High Water Mark)
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max

        max_dd = float(drawdown.min())

        # คำนวณ drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            # หาช่วง drawdown ที่ยาวนานที่สุด
            groups = (in_drawdown != in_drawdown.shift()).cumsum()
            dd_durations = in_drawdown.groupby(groups).sum()
            max_duration = int(dd_durations.max())
        else:
            max_duration = 0

        return max_dd, drawdown, max_duration

    def calculate_calmar_ratio(self, equity_curve: pd.Series) -> float:
        """
        คำนวณ Calmar Ratio = CAGR / |Max Drawdown|

        Args:
            equity_curve: Series ของ equity values

        Returns:
            Calmar Ratio
        """
        cagr = self.calculate_cagr(equity_curve)
        max_dd, _, _ = self.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return float("inf")

        return float(cagr / abs(max_dd))

    def calculate_win_rate(self, trades: pd.DataFrame) -> dict:
        """
        คำนวณ Win Rate และ trade statistics

        Args:
            trades: DataFrame ที่มีคอลัมน์ 'pnl'

        Returns:
            dict ของ trade statistics
        """
        if trades.empty or "pnl" not in trades.columns:
            return {"win_rate": 0, "total_trades": 0}

        total_trades = len(trades)
        winning_trades = (trades["pnl"] > 0).sum()
        losing_trades = (trades["pnl"] < 0).sum()
        breakeven_trades = (trades["pnl"] == 0).sum()

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0

        # Profit Factor = Gross Profit / Gross Loss
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        return {
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "breakeven_trades": int(breakeven_trades),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
        }

    def generate_report(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame = None,
        benchmark_curve: pd.Series = None,
        strategy_name: str = "Strategy",
    ) -> dict:
        """
        สร้างรายงาน Performance ครบชุด

        Args:
            equity_curve: Series ของ equity values
            trades: DataFrame ของ trades (optional)
            benchmark_curve: Series ของ benchmark equity (optional)
            strategy_name: ชื่อ strategy

        Returns:
            dict ของ metrics ทั้งหมด
        """
        returns = equity_curve.pct_change().dropna()

        # คำนวณ metrics หลัก
        total_return = self.calculate_total_return(equity_curve)
        cagr = self.calculate_cagr(equity_curve)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, dd_series, dd_duration = self.calculate_max_drawdown(equity_curve)
        calmar = self.calculate_calmar_ratio(equity_curve)

        # Volatility
        annual_vol = returns.std() * np.sqrt(252)

        report = {
            "strategy": strategy_name,
            "period": {
                "start": str(equity_curve.index[0].date()) if hasattr(equity_curve.index[0], 'date') else str(equity_curve.index[0]),
                "end": str(equity_curve.index[-1].date()) if hasattr(equity_curve.index[-1], 'date') else str(equity_curve.index[-1]),
                "trading_days": len(equity_curve),
            },
            "returns": {
                "total_return": total_return,
                "cagr": cagr,
                "annual_volatility": annual_vol,
            },
            "risk_adjusted": {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
            },
            "drawdown": {
                "max_drawdown": max_dd,
                "max_drawdown_duration_days": dd_duration,
            },
        }

        # เพิ่ม trade statistics
        if trades is not None and not trades.empty:
            trade_stats = self.calculate_win_rate(trades)
            report["trades"] = trade_stats

        # เพิ่ม benchmark comparison
        if benchmark_curve is not None:
            bm_return = self.calculate_total_return(benchmark_curve)
            bm_cagr = self.calculate_cagr(benchmark_curve)
            bm_returns = benchmark_curve.pct_change().dropna()
            bm_sharpe = self.calculate_sharpe_ratio(bm_returns)
            bm_max_dd, _, _ = self.calculate_max_drawdown(benchmark_curve)

            report["benchmark"] = {
                "total_return": bm_return,
                "cagr": bm_cagr,
                "sharpe_ratio": bm_sharpe,
                "max_drawdown": bm_max_dd,
            }

            # Alpha (excess return over benchmark)
            report["vs_benchmark"] = {
                "alpha_total": total_return - bm_return,
                "alpha_cagr": cagr - bm_cagr,
                "sharpe_improvement": sharpe - bm_sharpe,
                "drawdown_improvement": max_dd - bm_max_dd,
            }

        # แสดงผล
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 Performance Report: {strategy_name}")
        logger.info(f"{'='*50}")
        logger.info(f"  Total Return:    {total_return:.2%}")
        logger.info(f"  CAGR:            {cagr:.2%}")
        logger.info(f"  Sharpe Ratio:    {sharpe:.3f}")
        logger.info(f"  Sortino Ratio:   {sortino:.3f}")
        logger.info(f"  Max Drawdown:    {max_dd:.2%}")
        logger.info(f"  Calmar Ratio:    {calmar:.3f}")
        logger.info(f"{'='*50}\n")

        return report
