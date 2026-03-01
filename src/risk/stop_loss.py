"""
src/risk/stop_loss.py — Stop Loss Management

รองรับ:
- Fixed Stop Loss
- Trailing Stop Loss
- Time-based Stop Loss
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StopLossState:
    """สถานะ Stop Loss ปัจจุบัน"""
    entry_price: float
    stop_price: float
    highest_price: float  # สำหรับ trailing stop
    is_triggered: bool = False
    trigger_price: float = 0.0
    trigger_reason: str = ""


class StopLoss:
    """
    ระบบ Stop Loss สำหรับ Grid Trading

    รองรับ 3 ประเภท:
    1. Fixed Stop Loss — หยุดที่ราคาคงที่
    2. Trailing Stop Loss — ตามราคาขึ้น แต่ไม่ตามลง
    3. Time-based Stop — หยุดหลังถือครบ N วัน
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        trailing_stop_pct: float = 0.05,
        enabled: bool = True,
    ):
        """
        กำหนดค่าเริ่มต้น StopLoss

        Args:
            stop_loss_pct: % ขาดทุนสูงสุดจาก entry price
            trailing_stop_pct: % สำหรับ trailing stop
            enabled: เปิดใช้งาน stop loss หรือไม่
        """
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.enabled = enabled
        self.states: dict = {}  # {position_id: StopLossState}

    def calculate_stop_loss(
        self,
        entry_price: float,
        stop_pct: float = None,
        direction: str = "long",
    ) -> float:
        """
        คำนวณราคา Stop Loss

        Args:
            entry_price: ราคาที่เข้า position
            stop_pct: % stop loss (ใช้ค่า default ถ้าไม่ระบุ)
            direction: 'long' หรือ 'short'

        Returns:
            ราคา stop loss
        """
        stop_pct = stop_pct or self.stop_loss_pct

        if direction == "long":
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)

        logger.debug(
            f"Stop Loss: entry={entry_price:.2f}, stop={stop_price:.2f} ({direction}, {stop_pct:.1%})"
        )
        return stop_price

    def register_position(
        self,
        position_id: str,
        entry_price: float,
        direction: str = "long",
    ) -> StopLossState:
        """
        ลงทะเบียน position ใหม่เพื่อติดตาม stop loss

        Args:
            position_id: รหัสของ position
            entry_price: ราคาที่เข้า
            direction: ทิศทาง ('long' หรือ 'short')

        Returns:
            StopLossState object
        """
        stop_price = self.calculate_stop_loss(entry_price, direction=direction)

        state = StopLossState(
            entry_price=entry_price,
            stop_price=stop_price,
            highest_price=entry_price,
        )
        self.states[position_id] = state

        logger.debug(f"ลงทะเบียน position {position_id}: entry={entry_price:.2f}, stop={stop_price:.2f}")
        return state

    def check_stop_loss(
        self,
        current_price: float,
        stop_price: float,
        direction: str = "long",
    ) -> bool:
        """
        ตรวจสอบว่าราคาปัจจุบันถึง stop loss หรือยัง

        Args:
            current_price: ราคาปัจจุบัน
            stop_price: ราคา stop loss
            direction: 'long' หรือ 'short'

        Returns:
            True ถ้าถึง stop loss
        """
        if not self.enabled:
            return False

        if direction == "long":
            triggered = current_price <= stop_price
        else:
            triggered = current_price >= stop_price

        if triggered:
            logger.warning(
                f"⚠️ Stop Loss triggered: price={current_price:.2f}, stop={stop_price:.2f}"
            )

        return triggered

    def trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        trail_pct: float = None,
        direction: str = "long",
    ) -> tuple:
        """
        คำนวณ Trailing Stop

        สำหรับ Long: Stop ตาม high สูงขึ้น แต่ไม่ลดต่ำลง
        สำหรับ Short: Stop ตาม low ต่ำลง แต่ไม่เพิ่มสูงขึ้น

        Args:
            current_price: ราคาปัจจุบัน
            highest_price: ราคาสูงสุด/ต่ำสุดตั้งแต่เข้า position
            trail_pct: % trailing distance
            direction: 'long' หรือ 'short'

        Returns:
            tuple (new_highest_price, new_stop_price, is_triggered)
        """
        trail_pct = trail_pct or self.trailing_stop_pct

        if direction == "long":
            # อัปเดต highest price
            new_highest = max(highest_price, current_price)
            # คำนวณ trailing stop ใหม่
            new_stop = new_highest * (1 - trail_pct)
            # ตรวจสอบ trigger
            is_triggered = current_price <= new_stop
        else:
            # สำหรับ short position
            new_highest = min(highest_price, current_price)  # lowest price
            new_stop = new_highest * (1 + trail_pct)
            is_triggered = current_price >= new_stop

        if is_triggered:
            logger.warning(
                f"⚠️ Trailing Stop triggered: price={current_price:.2f}, "
                f"peak={new_highest:.2f}, stop={new_stop:.2f}"
            )

        return new_highest, new_stop, is_triggered

    def update_position(
        self,
        position_id: str,
        current_price: float,
        direction: str = "long",
    ) -> bool:
        """
        อัปเดตสถานะ stop loss สำหรับ position ที่ลงทะเบียนแล้ว

        Args:
            position_id: รหัส position
            current_price: ราคาปัจจุบัน
            direction: ทิศทาง

        Returns:
            True ถ้า stop loss ถูก trigger
        """
        if position_id not in self.states:
            logger.warning(f"ไม่พบ position {position_id}")
            return False

        state = self.states[position_id]

        if state.is_triggered:
            return True

        # อัปเดต trailing stop
        new_highest, new_stop, triggered = self.trailing_stop(
            current_price,
            state.highest_price,
            direction=direction,
        )

        state.highest_price = new_highest
        state.stop_price = new_stop

        if triggered:
            state.is_triggered = True
            state.trigger_price = current_price
            state.trigger_reason = "trailing_stop"

        # ตรวจสอบ fixed stop loss
        elif self.check_stop_loss(current_price, self.calculate_stop_loss(state.entry_price), direction):
            state.is_triggered = True
            state.trigger_price = current_price
            state.trigger_reason = "fixed_stop_loss"

        return state.is_triggered

    def remove_position(self, position_id: str) -> None:
        """
        ลบ position ออกจากการติดตาม

        Args:
            position_id: รหัส position
        """
        if position_id in self.states:
            del self.states[position_id]
