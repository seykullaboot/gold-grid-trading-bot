"""
src/risk/circuit_breaker.py — Circuit Breaker System

หยุดการเทรดอัตโนมัติเมื่อ:
- ขาดทุนเกิน Daily Loss Limit
- ขาดทุนติดต่อกันหลาย trades
- ระบบผิดปกติ
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BreakerEvent:
    """บันทึก Circuit Breaker events"""
    timestamp: str
    reason: str
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    reset_time: str = ""


class CircuitBreaker:
    """
    Circuit Breaker — ระบบหยุดเทรดฉุกเฉิน

    เหมือน Circuit Breaker ในตลาดหุ้น:
    เมื่อขาดทุนถึง threshold → หยุดทั้งหมด → รอ cooldown → รีเซ็ต
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_consecutive_losses: int = 5,
        cooldown_hours: int = 24,
    ):
        """
        กำหนดค่าเริ่มต้น CircuitBreaker

        Args:
            max_daily_loss_pct: ขาดทุนสูงสุดต่อวัน (3%)
            max_consecutive_losses: จำนวนครั้งขาดทุนติดต่อกันสูงสุด
            cooldown_hours: ระยะเวลา cooldown (ชั่วโมง)
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_hours = cooldown_hours

        # State
        self.is_triggered = False
        self.trigger_reason = ""
        self.trigger_time: Optional[datetime] = None
        self.reset_time: Optional[datetime] = None
        self.events: List[BreakerEvent] = []

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.last_reset_date = None

        # Consecutive loss tracking
        self.consecutive_losses = 0
        self.recent_trades: List[float] = []

    def check_daily_loss(
        self,
        daily_pnl: float,
        initial_equity: float,
    ) -> bool:
        """
        ตรวจสอบ Daily Loss Limit

        Args:
            daily_pnl: กำไร/ขาดทุนวันนี้ (USD)
            initial_equity: equity ต้นวัน

        Returns:
            True ถ้าเกิน daily loss limit
        """
        if initial_equity <= 0:
            return False

        daily_loss_pct = daily_pnl / initial_equity

        if daily_loss_pct < -self.max_daily_loss_pct:
            reason = (
                f"Daily loss {daily_loss_pct:.2%} เกิน limit "
                f"-{self.max_daily_loss_pct:.2%}"
            )
            self.trigger_breaker(reason, daily_pnl=daily_pnl)
            return True

        return False

    def check_consecutive_losses(
        self,
        trade_pnl: float,
    ) -> bool:
        """
        ตรวจสอบ Consecutive Losses

        Args:
            trade_pnl: กำไร/ขาดทุนของ trade ล่าสุด

        Returns:
            True ถ้าขาดทุนติดต่อกันเกิน limit
        """
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # รีเซ็ตเมื่อมีกำไร

        self.recent_trades.append(trade_pnl)
        # เก็บแค่ 20 trades ล่าสุด
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)

        if self.consecutive_losses >= self.max_consecutive_losses:
            reason = (
                f"ขาดทุนติดต่อกัน {self.consecutive_losses} ครั้ง "
                f"(limit: {self.max_consecutive_losses})"
            )
            self.trigger_breaker(
                reason,
                consecutive_losses=self.consecutive_losses,
            )
            return True

        return False

    def trigger_breaker(
        self,
        reason: str,
        daily_pnl: float = 0.0,
        consecutive_losses: int = 0,
    ) -> None:
        """
        เปิด Circuit Breaker — หยุดการเทรด

        Args:
            reason: สาเหตุที่ trigger
            daily_pnl: PnL วันนี้
            consecutive_losses: จำนวน consecutive losses
        """
        now = datetime.now()
        reset_at = now + timedelta(hours=self.cooldown_hours)

        self.is_triggered = True
        self.trigger_reason = reason
        self.trigger_time = now
        self.reset_time = reset_at

        event = BreakerEvent(
            timestamp=now.isoformat(),
            reason=reason,
            daily_pnl=daily_pnl,
            consecutive_losses=consecutive_losses,
            reset_time=reset_at.isoformat(),
        )
        self.events.append(event)

        logger.warning(
            f"🔴 CIRCUIT BREAKER TRIGGERED!\n"
            f"   Reason: {reason}\n"
            f"   Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"   Auto-reset at: {reset_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def reset_breaker(self, force: bool = False) -> bool:
        """
        รีเซ็ต Circuit Breaker

        Args:
            force: รีเซ็ตทันทีโดยไม่รอ cooldown

        Returns:
            True ถ้ารีเซ็ตสำเร็จ
        """
        if not self.is_triggered:
            return True

        now = datetime.now()

        # ตรวจสอบว่า cooldown ครบแล้วหรือยัง
        if not force and self.reset_time and now < self.reset_time:
            remaining = (self.reset_time - now).total_seconds() / 3600
            logger.info(f"Circuit Breaker ยังไม่รีเซ็ต — เหลือ {remaining:.1f} ชั่วโมง")
            return False

        # รีเซ็ต
        self.is_triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        self.reset_time = None
        self.consecutive_losses = 0
        self.daily_pnl = 0.0

        logger.info("✅ Circuit Breaker รีเซ็ตแล้ว — พร้อมเทรด")
        return True

    def can_trade(self) -> bool:
        """
        ตรวจสอบว่าสามารถเทรดได้หรือไม่

        Returns:
            True ถ้าสามารถเทรดได้
        """
        if self.is_triggered:
            # ลองรีเซ็ตอัตโนมัติ
            self.reset_breaker()

        return not self.is_triggered

    def reset_daily(self, current_equity: float) -> None:
        """
        รีเซ็ตตัวนับ daily สำหรับวันใหม่

        Args:
            current_equity: equity ต้นวัน
        """
        self.daily_pnl = 0.0
        self.daily_start_equity = current_equity
        logger.debug(f"รีเซ็ต daily tracker: equity={current_equity:,.2f}")

    def get_status(self) -> dict:
        """
        ดึงสถานะ Circuit Breaker

        Returns:
            dict ของสถานะทั้งหมด
        """
        return {
            "is_triggered": self.is_triggered,
            "trigger_reason": self.trigger_reason,
            "trigger_time": str(self.trigger_time) if self.trigger_time else None,
            "reset_time": str(self.reset_time) if self.reset_time else None,
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": self.daily_pnl,
            "total_events": len(self.events),
        }
