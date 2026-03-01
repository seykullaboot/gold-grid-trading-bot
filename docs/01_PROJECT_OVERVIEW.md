# 📋 ภาพรวมโปรเจกต์ — Gold Grid Trading Bot

## 1. วัตถุประสงค์ (Objectives)

โปรเจกต์นี้มีวัตถุประสงค์เพื่อ:

1. **พัฒนาระบบเทรดทองคำอัตโนมัติ** โดยใช้ Grid Trading Strategy ที่ปรับตัวได้ด้วย AI
2. **สร้างระบบ AI/ML** เพื่อตรวจจับสภาวะตลาดและพยากรณ์ความผันผวน
3. **ออกแบบ Risk Management** ที่ครบวงจรเพื่อควบคุมความเสี่ยง
4. **ทำ Backtesting** บนข้อมูลย้อนหลัง 20 ปี เพื่อพิสูจน์ประสิทธิภาพ
5. **วางรากฐานสำหรับ Production** ในอนาคต

---

## 2. เทคโนโลยีที่ใช้

### 🐍 Core Language
- **Python 3.10+** — ภาษาหลัก
- **pandas & numpy** — data manipulation
- **yfinance** — ดาวน์โหลดข้อมูลตลาด

### 🤖 Machine Learning
- **scikit-learn** — Random Forest, preprocessing
- **TensorFlow/Keras** — LSTM neural network
- **XGBoost** — gradient boosting (optional)

### 📊 Technical Analysis
- **ta (Technical Analysis library)** — indicators ครบชุด

### 📈 Backtesting & Visualization
- **vectorbt** — vectorized backtesting
- **matplotlib, plotly, seaborn** — charts & visualization

---

## 3. ส่วนประกอบหลัก (Components)

```
┌─────────────────────────────────────────────────────────┐
│                   Gold Grid Trading Bot                  │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Data Layer  │  AI Models   │  Strategies  │    Risk    │
│              │              │              │ Management │
│ • DataLoader │ • Regime     │ • Classic    │ • StopLoss │
│ • Processor  │   Detector   │   Grid       │ • Drawdown │
│ • Feature    │   (RF)       │ • Adaptive   │ • Circuit  │
│   Engineer   │ • Volatility │   Grid (AI)  │   Breaker  │
│              │   Predictor  │              │ • Position │
│              │   (LSTM)     │              │   Limit    │
├──────────────┴──────────────┴──────────────┴────────────┤
│                   Backtesting Engine                     │
│   • Walk-Forward Validation  • Performance Metrics       │
└─────────────────────────────────────────────────────────┘
```

---

## 4. เป้าหมาย KPI

| Metric | Classic Grid | **AI Adaptive** | Buy & Hold |
|--------|-------------|-----------------|------------|
| 📈 Annual Return | 12-18% | **≥ 20-35%** | 8-12% |
| 📊 Sharpe Ratio | 1.0-1.5 | **≥ 1.8-2.5** | 0.6-0.9 |
| 📉 Max Drawdown | -20% | **≤ -15%** | -30% |
| 🎯 Win Rate | 55-65% | **≥ 65-75%** | N/A |
| 💰 Profit Factor | 1.3-1.6 | **≥ 1.8-2.5** | N/A |

---

## 5. ข้อมูลที่ใช้ (Data Sources)

### Primary
- **GLD (SPDR Gold Shares ETF)** — ข้อมูลหลัก
  - ช่วงเวลา: 2004-01-01 ถึง 2024-12-31 (20 ปี)
  - Timeframe: Daily (1D)
  - แหล่งข้อมูล: Yahoo Finance (yfinance)

### Secondary (Macroeconomic)
- **SPY** — S&P 500 ETF (market sentiment)
- **TLT** — Treasury Bond ETF (safe haven indicator)
- **^VIX** — Volatility Index
- **DX-Y.NYB** — US Dollar Index

---

## 6. Success Criteria

โปรเจกต์ถือว่าสำเร็จเมื่อ:

✅ AI Adaptive Grid ให้ผลตอบแทนสูงกว่า Classic Grid อย่างน้อย 5%  
✅ Sharpe Ratio ≥ 1.8 บน test set  
✅ Max Drawdown ≤ -15%  
✅ Win Rate ≥ 65%  
✅ ผ่าน Walk-Forward Validation อย่างน้อย 70% ของ folds มี Sharpe > 1.5  
✅ ระบบ Risk Management ทำงานถูกต้องใน stress tests  
✅ Code coverage ≥ 80%  

---

## 7. ข้อจำกัดและข้อสมมติ

⚠️ **ข้อจำกัดสำคัญ:**
- Backtest ใช้ Daily data — ไม่ครอบคลุม intraday movements
- Commission และ Slippage เป็นค่าประมาณ (0.1% + 0.05%)
- ไม่คำนึงถึง liquidity impact สำหรับ large positions
- ผลการ backtest ไม่รับประกันผลในอนาคต
