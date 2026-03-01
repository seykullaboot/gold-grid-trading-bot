# 🛡️ Phase 4: Risk Management (สัปดาห์ 8-9)

**ระยะเวลา:** Day 50-63  
**เป้าหมาย:** ระบบ Risk Management ครบวงจร + Stress Testing

---

## ส่วนประกอบ Risk Management

### 1. Stop Loss System

```python
from src.risk.stop_loss import StopLoss

sl = StopLoss(stop_loss_pct=0.08, trailing_stop_pct=0.05)
stop_price = sl.calculate_stop_loss(entry_price=180.0)
# stop_price = 165.6 (8% ต่ำกว่า entry)

# Trailing Stop
new_high, new_stop, triggered = sl.trailing_stop(
    current_price=175.0,
    highest_price=185.0,
    trail_pct=0.05
)
```

### 2. Drawdown Protection

```python
from src.risk.drawdown_protection import DrawdownProtection

dd_protect = DrawdownProtection(
    max_drawdown_pct=0.15,
    reduce_threshold_pct=0.10,
)

# ลด position เมื่อ drawdown > 10%
new_size = dd_protect.reduce_position_on_drawdown(
    drawdown=-0.12,
    current_position_size=10_000
)
# new_size = 5_000 (ลด 50%)
```

### 3. Circuit Breaker

```python
from src.risk.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(
    max_daily_loss_pct=0.03,
    max_consecutive_losses=5,
    cooldown_hours=24,
)

# ตรวจสอบทุกวัน
if breaker.check_daily_loss(daily_pnl, initial_equity):
    print("🔴 Circuit Breaker Triggered — หยุดเทรด")

if breaker.can_trade():
    # ดำเนินการเทรดต่อ
    pass
```

---

## Stress Testing

### Scenarios ที่ต้องทดสอบ

| Scenario | ช่วงเวลา | การเปลี่ยนแปลง |
|---------|---------|--------------|
| 2008 Financial Crisis | Sep-Dec 2008 | GLD -15% ใน 3 เดือน |
| 2011 Gold Peak & Crash | Sep 2011 - Jun 2013 | GLD -40% |
| COVID-19 (2020) | Feb-Mar 2020 | Volatility สูง 3x |
| 2022 Rate Hike | Jan-Nov 2022 | GLD -20% |

### เกณฑ์ผ่าน Stress Test

- Max Drawdown ≤ -20% ใน worst scenario
- Circuit Breaker ทำงานถูกต้องทุก scenario
- Trailing Stop ลดการขาดทุนได้อย่างน้อย 30%

---

## Monte Carlo Simulation

```python
import numpy as np

def monte_carlo_simulation(returns, n_simulations=1000, n_days=252):
    """จำลอง equity curves ด้วย Monte Carlo"""
    results = []
    for _ in range(n_simulations):
        sampled = np.random.choice(returns, size=n_days, replace=True)
        equity = 100_000 * np.cumprod(1 + sampled)
        results.append(equity[-1])
    return np.array(results)
```

### เป้าหมาย Monte Carlo

- 5th percentile ≥ $85,000 (ไม่ขาดทุนเกิน 15%)
- 95th percentile ≥ $130,000 (กำไรอย่างน้อย 30%)

---

## ✅ Checkpoint Phase 4

- [ ] Stop Loss ทำงานถูกต้องใน unit tests
- [ ] Drawdown Protection ลด position อย่างถูกต้อง
- [ ] Circuit Breaker trigger และ reset ถูกต้อง
- [ ] ผ่าน stress tests ทุก 4 scenarios
- [ ] Monte Carlo: 5th percentile ≥ $85,000
