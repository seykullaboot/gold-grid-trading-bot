"""
src/risk/drawdown_protection.py — Drawdown Protection System

ปกป้องพอร์ตจาก Drawdown สูงเกิน:
- ติดตาม equity curve
- คำนวณ current drawdown
- ลด position size เมื่อ drawdown เกิน threshold
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DrawdownProtection:
    """
    ระบบป้องกัน Drawdown

    กลไก:
    - ติดตาม equity สูงสุด (High Water Mark)
    - คำนวณ drawdown จาก HWM
    - ลด position ขั้นบันไดเมื่อ drawdown เพิ่มขึ้น
    - หยุดเทรดเมื่อถึง max drawdown limit
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.15,
        reduce_threshold_pct: float = 0.10,
        position_reduce_factor: float = 0.5,
    ):
        """
        กำหนดค่าเริ่มต้น DrawdownProtection

        Args:
            max_drawdown_pct: Drawdown สูงสุดที่อนุญาต (15%)
            reduce_threshold_pct: เริ่มลด position เมื่อ DD > 10%
            position_reduce_factor: ลด position ลงเหลือ 50%
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.reduce_threshold_pct = reduce_threshold_pct
        self.position_reduce_factor = position_reduce_factor

        self.high_water_mark = 0.0
        self.current_drawdown = 0.0
        self.is_max_dd_breached = False

    def calculate_drawdown(
        self,
        equity_curve: List[float],
    ) -> tuple:
        """
        คำนวณ Drawdown จาก Equity Curve

        Args:
            equity_curve: รายการค่า equity

        Returns:
            tuple (drawdown_series, max_drawdown, current_drawdown)
        """
        equity = np.array(equity_curve)

        # คำนวณ running maximum (High Water Mark)
        running_max = np.maximum.accumulate(equity)

        # Drawdown = (current - peak) / peak
        drawdown = (equity - running_max) / running_max

        max_dd = float(drawdown.min())
        current_dd = float(drawdown[-1])

        # อัปเดต state
        self.high_water_mark = float(running_max[-1])
        self.current_drawdown = current_dd

        return drawdown, max_dd, current_dd

    def check_max_drawdown(
        self,
        current_equity: float,
        high_water_mark: float = None,
    ) -> bool:
        """
        ตรวจสอบว่า drawdown เกิน max limit หรือไม่

        Args:
            current_equity: equity ปัจจุบัน
            high_water_mark: equity สูงสุดตลอดกาล

        Returns:
            True ถ้า drawdown เกิน max limit
        """
        hwm = high_water_mark or self.high_water_mark
        if hwm <= 0:
            return False

        dd = (current_equity - hwm) / hwm
        self.current_drawdown = dd

        if dd <= -self.max_drawdown_pct:
            if not self.is_max_dd_breached:
                logger.warning(
                    f"🚨 MAX DRAWDOWN BREACHED: {dd:.2%} (limit: -{self.max_drawdown_pct:.2%})"
                )
                self.is_max_dd_breached = True
            return True

        self.is_max_dd_breached = False
        return False

    def reduce_position_on_drawdown(
        self,
        drawdown: float,
        current_position_size: float,
    ) -> float:
        """
        ลด position size ตาม drawdown level

        ระดับการลด:
        - DD < threshold: ไม่ลด
        - DD >= threshold: ลด 50%
        - DD >= max: ลด 100% (ปิด positions)

        Args:
            drawdown: Drawdown ปัจจุบัน (ค่าลบ เช่น -0.12)
            current_position_size: ขนาด position ปัจจุบัน

        Returns:
            ขนาด position ที่ปรับแล้ว
        """
        abs_dd = abs(drawdown)

        if abs_dd >= self.max_drawdown_pct:
            # หยุดเทรด — ปิด positions ทั้งหมด
            logger.warning(f"🚨 Drawdown {drawdown:.2%} — ปิด positions ทั้งหมด")
            return 0.0

        elif abs_dd >= self.reduce_threshold_pct:
            # ลด position
            reduction_factor = self.position_reduce_factor

            # ยิ่ง drawdown สูง ยิ่งลดมาก (linear interpolation)
            dd_range = self.max_drawdown_pct - self.reduce_threshold_pct
            dd_excess = abs_dd - self.reduce_threshold_pct
            progressive_factor = 1 - (dd_excess / dd_range) * (1 - reduction_factor)
            progressive_factor = max(progressive_factor, reduction_factor)

            new_size = current_position_size * progressive_factor
            logger.info(
                f"⚠️ ลด position: DD={drawdown:.2%}, "
                f"size: {current_position_size:.4f} → {new_size:.4f}"
            )
            return new_size

        else:
            # Drawdown ยังปกติ
            return current_position_size

    def update_high_water_mark(self, current_equity: float) -> float:
        """
        อัปเดต High Water Mark

        Args:
            current_equity: equity ปัจจุบัน

        Returns:
            High Water Mark ที่อัปเดตแล้ว
        """
        self.high_water_mark = max(self.high_water_mark, current_equity)
        return self.high_water_mark

    def get_recovery_factor(self) -> float:
        """
        คำนวณ Recovery Factor ที่ต้องการเพื่อกลับสู่ HWM

        Returns:
            % ที่ต้องได้กำไรเพื่อกู้คืน drawdown
        """
        if self.current_drawdown >= 0:
            return 0.0
        # ถ้าขาดทุน -15% ต้องกำไร +17.6% เพื่อคืนทุน
        return abs(self.current_drawdown) / (1 + self.current_drawdown)
