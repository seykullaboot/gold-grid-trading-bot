# ✅ Checklist Tracker — ทุก Phase

> อัปเดตสถานะเมื่อเสร็จแต่ละ task  
> ✅ = เสร็จแล้ว | 🔄 = กำลังทำ | ⏸️ = รอ | ❌ = ปัญหา

---

## Phase 1: การเตรียมการ (สัปดาห์ 1-2)

### Environment Setup
- [ ] ติดตั้ง Python 3.10+
- [ ] สร้าง virtual environment
- [ ] `pip install -r requirements.txt` สำเร็จ
- [ ] ทดสอบ imports ทั้งหมด

### Data Collection
- [ ] ดาวน์โหลด GLD 2004-2024 (5,000+ แถว)
- [ ] ดาวน์โหลด SPY, TLT, ^VIX
- [ ] Validate ข้อมูล — ไม่มี missing > 1%
- [ ] บันทึกใน `data/raw/`

### EDA
- [ ] วิเคราะห์ distribution ของ returns
- [ ] ตรวจสอบ volatility clustering
- [ ] วิเคราะห์ seasonality
- [ ] Correlation analysis กับ macro

### Architecture
- [ ] วาด System Diagram
- [ ] กำหนด API interfaces

### Classic Grid Prototype
- [ ] `BaseGridStrategy` ทำงานได้
- [ ] คำนวณ grid levels ถูกต้อง
- [ ] Execute orders ถูกต้อง
- [ ] คำนวณ PnL ถูกต้อง

---

## Phase 2: Core Development (สัปดาห์ 3-4)

### Technical Indicators
- [ ] EMA (10, 20, 50, 200)
- [ ] SMA (20, 50, 200)
- [ ] MACD (12, 26, 9)
- [ ] Bollinger Bands (20, 2)
- [ ] ADX (14)
- [ ] RSI (7, 14, 21)
- [ ] Stochastic (14, 3)
- [ ] ATR (7, 14, 21)
- [ ] OBV, MFI, CMF

### Backtesting Framework
- [ ] `BacktestEngine` รันได้
- [ ] Commission + Slippage ถูกต้อง
- [ ] Equity curve ถูกต้อง
- [ ] Benchmark (Buy & Hold) ถูกต้อง

### Performance Metrics
- [ ] Total Return, CAGR
- [ ] Sharpe, Sortino, Calmar
- [ ] Max Drawdown, Duration
- [ ] Win Rate, Profit Factor

### Unit Tests
- [ ] `test_data.py` ผ่านทั้งหมด
- [ ] `test_strategies.py` ผ่านทั้งหมด
- [ ] Coverage ≥ 80%

### Walk-Forward Validation
- [ ] WFV implementation สมบูรณ์
- [ ] Mean Sharpe ≥ 1.0

---

## Phase 3: AI Models (สัปดาห์ 5-7)

### Market Regime Detector
- [ ] สร้าง regime labels (trending/ranging/volatile)
- [ ] Train Random Forest (200 trees)
- [ ] Validation accuracy ≥ 75%
- [ ] บันทึก model ที่ `data/models/regime_detector.pkl`

### Volatility Predictor
- [ ] สร้าง LSTM sequences (30 timesteps)
- [ ] Build LSTM architecture (64→32→16→1)
- [ ] Train ด้วย Early Stopping
- [ ] Test MAPE < 15%
- [ ] บันทึก model ที่ `data/models/volatility_predictor.h5`

### Integration
- [ ] `AdaptiveGridStrategy` รับ AI predictions
- [ ] AI Adaptive Sharpe ≥ 1.8
- [ ] Integration tests ผ่าน

---

## Phase 4: Risk Management (สัปดาห์ 8-9)

### Stop Loss
- [ ] Fixed stop loss คำนวณถูกต้อง
- [ ] Trailing stop อัปเดตถูกต้อง
- [ ] Unit tests ผ่าน

### Drawdown Protection
- [ ] คำนวณ drawdown ถูกต้อง
- [ ] ลด position เมื่อ DD > 10%
- [ ] หยุดเมื่อ DD > 15%

### Circuit Breaker
- [ ] Trigger เมื่อ daily loss > 3%
- [ ] Trigger เมื่อ consecutive losses > 5
- [ ] Auto-reset หลัง 24 ชั่วโมง

### Stress Testing
- [ ] ผ่าน scenario 2008 Crisis
- [ ] ผ่าน scenario 2011 Gold Crash
- [ ] ผ่าน scenario COVID-19 2020
- [ ] ผ่าน scenario 2022 Rate Hike

---

## Phase 5: Optimization (สัปดาห์ 10)

- [ ] Grid Search / Random Search เสร็จ (≥ 200 trials)
- [ ] Optimal params บันทึกแล้ว
- [ ] Walk-Forward: mean Sharpe ≥ 1.5
- [ ] Final Backtest: Sharpe ≥ 1.8, DD ≤ -15%

---

## Phase 6: Production (สัปดาห์ 11-12)

- [ ] Paper Trading API เชื่อมต่อได้
- [ ] Dashboard แสดงผล
- [ ] Docker build สำเร็จ
- [ ] Documentation สมบูรณ์
- [ ] Code Review ผ่าน

---

## 🎯 เป้าหมายสุดท้าย

- [ ] AI Adaptive Return ≥ 20% (annualized)
- [ ] Sharpe Ratio ≥ 1.8
- [ ] Max Drawdown ≤ -15%
- [ ] Win Rate ≥ 65%
- [ ] ดีกว่า Buy & Hold อย่างน้อย 10%
