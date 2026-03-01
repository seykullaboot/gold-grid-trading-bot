"""
src/risk/position_limit.py — Position Sizing และ Limit Management

คำนวณขนาด Position ที่เหมาะสม:
- Fixed Fractional
- Kelly Criterion
- Risk-based sizing
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PositionLimit:
    """
    ระบบจัดการขนาด Position และ Limits

    Methods:
    - Fixed fractional: ใช้ % คงที่ของ capital
    - Kelly Criterion: optimize สำหรับ max growth
    - Risk-based: คำนวณจาก stop loss distance
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_total_exposure_pct: float = 0.80,
        kelly_fraction: float = 0.25,
    ):
        """
        กำหนดค่าเริ่มต้น PositionLimit

        Args:
            max_position_pct: ขนาด position สูงสุดต่อ trade (20%)
            max_total_exposure_pct: exposure รวมสูงสุด (80%)
            kelly_fraction: เศษส่วนของ Kelly ที่ใช้ (fractional Kelly)
        """
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        capital: float,
        risk_pct: float,
        stop_loss_pct: float,
        method: str = "risk_based",
    ) -> float:
        """
        คำนวณขนาด Position ที่เหมาะสม

        Args:
            capital: เงินทุนทั้งหมด (USD)
            risk_pct: % ของ capital ที่ยอมเสี่ยงต่อ trade
            stop_loss_pct: ระยะ stop loss (%)
            method: วิธีคำนวณ ('fixed', 'risk_based', 'kelly')

        Returns:
            ขนาด position เป็น USD
        """
        if method == "fixed":
            # Fixed fractional — ใช้ % คงที่
            size = capital * risk_pct

        elif method == "risk_based":
            # Risk-based sizing
            # Position size = (Capital × Risk%) / Stop Loss%
            if stop_loss_pct <= 0:
                logger.warning("stop_loss_pct <= 0 — ใช้ค่า default 0.02")
                stop_loss_pct = 0.02

            risk_amount = capital * risk_pct
            size = risk_amount / stop_loss_pct

        elif method == "kelly":
            size = capital * self.kelly_fraction * risk_pct

        else:
            raise ValueError(f"ไม่รู้จัก method: {method}")

        # จำกัด max position
        max_size = capital * self.max_position_pct
        size = min(size, max_size)

        logger.debug(
            f"Position size: {size:,.2f} USD "
            f"(capital={capital:,.2f}, risk={risk_pct:.2%}, method={method})"
        )
        return float(size)

    def check_position_limit(
        self,
        current_positions: list,
        max_positions: int,
        capital: float,
    ) -> dict:
        """
        ตรวจสอบว่าสามารถเพิ่ม position ใหม่ได้หรือไม่

        Args:
            current_positions: รายการ positions ปัจจุบัน (แต่ละ item เป็น dict มี 'size')
            max_positions: จำนวน positions สูงสุดที่อนุญาต
            capital: เงินทุนทั้งหมด

        Returns:
            dict {'can_add': bool, 'reason': str, 'available_capital': float}
        """
        # ตรวจสอบจำนวน positions
        if len(current_positions) >= max_positions:
            return {
                "can_add": False,
                "reason": f"ถึง limit {max_positions} positions แล้ว",
                "available_capital": 0.0,
            }

        # ตรวจสอบ total exposure
        total_exposure = sum(p.get("size", 0) for p in current_positions)
        exposure_pct = total_exposure / capital if capital > 0 else 0

        if exposure_pct >= self.max_total_exposure_pct:
            return {
                "can_add": False,
                "reason": f"Total exposure {exposure_pct:.2%} ถึง limit {self.max_total_exposure_pct:.2%}",
                "available_capital": 0.0,
            }

        # คำนวณ capital ที่เหลือ
        available = capital * self.max_total_exposure_pct - total_exposure
        available = max(available, 0)

        return {
            "can_add": True,
            "reason": "OK",
            "available_capital": available,
            "current_positions": len(current_positions),
            "total_exposure_pct": exposure_pct,
        }

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        คำนวณ Kelly Criterion

        Formula: f* = (bp - q) / b
        โดยที่:
            f* = Kelly fraction (optimal bet size)
            b  = avg_win / avg_loss (odds)
            p  = win rate
            q  = 1 - p (loss rate)

        Args:
            win_rate: อัตราชนะ (0-1)
            avg_win: กำไรเฉลี่ยต่อ trade ที่ชนะ (USD หรือ %)
            avg_loss: ขาดทุนเฉลี่ยต่อ trade ที่แพ้ (บวก USD หรือ %)

        Returns:
            Kelly fraction (0-1) — ส่วนของ capital ที่ควรลงทุน
        """
        if avg_loss <= 0:
            logger.warning("avg_loss <= 0 — Kelly Criterion คำนวณไม่ได้")
            return 0.0

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  # reward-to-risk ratio

        kelly = (b * p - q) / b

        # Full Kelly มักให้ค่าสูงเกินไป → ใช้ fractional Kelly
        full_kelly = max(kelly, 0.0)
        fractional_kelly = full_kelly * self.kelly_fraction

        # จำกัดไว้ที่ max_position_pct
        result = min(fractional_kelly, self.max_position_pct)

        logger.info(
            f"Kelly Criterion: win_rate={p:.2%}, b={b:.2f}, "
            f"full_kelly={full_kelly:.4f}, "
            f"fractional_kelly ({self.kelly_fraction}x)={result:.4f}"
        )
        return float(result)

    def calculate_units_from_size(
        self,
        position_size_usd: float,
        current_price: float,
    ) -> float:
        """
        แปลงขนาด position จาก USD เป็นจำนวนหน่วย

        Args:
            position_size_usd: ขนาด position (USD)
            current_price: ราคาปัจจุบัน

        Returns:
            จำนวนหน่วย
        """
        if current_price <= 0:
            raise ValueError("current_price ต้องมากกว่า 0")
        return position_size_usd / current_price
