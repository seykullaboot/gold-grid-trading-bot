# 🥇 Gold Grid Trading Bot — AI Adaptive Grid Trading System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

---

## 📋 ภาพรวมโปรเจกต์

**Gold Grid Trading Bot** คือระบบเทรดทองคำอัตโนมัติที่ผสมผสาน **Grid Trading Strategy** เข้ากับ **AI/Machine Learning** เพื่อปรับตัวตามสภาวะตลาดแบบ Real-time

ระบบใช้ข้อมูลของ **GLD (SPDR Gold Shares ETF)** ย้อนหลัง 20 ปี เพื่อ Train โมเดล AI ที่สามารถ:
- ตรวจจับ Market Regime (Trending / Ranging / Volatile)
- พยากรณ์ความผันผวน (Volatility Prediction)
- ปรับ Grid Parameters แบบอัตโนมัติ

---

## ✨ คุณสมบัติหลัก

### 🤖 AI & Machine Learning
- **Market Regime Detection** — Random Forest จำแนก 3 สภาวะตลาด
- **Volatility Prediction** — LSTM พยากรณ์ความผันผวนล่วงหน้า
- **Adaptive Grid** — ปรับ Grid Spacing อัตโนมัติตาม Regime + Volatility

### 📊 Grid Trading Engine
- **Classic Grid Strategy** — Grid คงที่สำหรับ Baseline
- **Adaptive Grid Strategy** — Grid ปรับตัวด้วย AI
- คำนวณ Grid Levels, Orders, และ PnL อัตโนมัติ

### 🛡️ Risk Management ครบวงจร
- **Stop Loss** — Fixed + Trailing Stop Loss
- **Drawdown Protection** — ลด Position เมื่อ Drawdown สูง
- **Circuit Breaker** — หยุดเทรดเมื่อขาดทุนเกิน Threshold
- **Position Sizing** — Kelly Criterion + Fixed Fractional

### 📈 Backtesting Framework
- Walk-Forward Validation
- Monte Carlo Simulation
- Comprehensive Performance Metrics
- HTML/PDF Report Generation

---

## 🚀 วิธีการติดตั้ง (Quick Start)

### ขั้นตอนที่ 1: Clone Repository
```bash
git clone https://github.com/yourusername/gold-grid-trading-bot.git
cd gold-grid-trading-bot
```

### ขั้นตอนที่ 2: สร้าง Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# หรือ
venv\Scripts\activate           # Windows
```

### ขั้นตอนที่ 3: ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### ขั้นตอนที่ 4: ดาวน์โหลดข้อมูล
```bash
python scripts/data_collection.py
```

### ขั้นตอนที่ 5: Train AI Models
```bash
python scripts/train_models.py
```

### ขั้นตอนที่ 6: รัน Backtest
```bash
python scripts/backtest.py
```

### ขั้นตอนที่ 7: Optimize Parameters (Optional)
```bash
python scripts/optimize.py
```

---

## 📁 โครงสร้างโปรเจกต์

```
gold-grid-trading-bot/
│
├── 📄 README.md                    # เอกสารหลักของโปรเจกต์
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup
│
├── 📁 config/                      # Configuration files
│   ├── config.yaml                 # ค่าตั้งต้นหลัก
│   └── parameters.yaml             # Parameter ranges สำหรับ optimization
│
├── 📁 data/                        # ข้อมูล (ไม่ commit ลง Git)
│   ├── raw/                        # ข้อมูลดิบจาก yfinance
│   ├── processed/                  # ข้อมูลที่ผ่าน Feature Engineering
│   └── models/                     # Trained AI models (.pkl, .h5)
│
├── 📁 src/                         # Source code หลัก
│   ├── data/
│   │   ├── loader.py               # ดาวน์โหลดและโหลดข้อมูล
│   │   └── processor.py            # ทำความสะอาดและแปลงข้อมูล
│   ├── features/
│   │   └── engineering.py          # Technical indicators & features
│   ├── models/
│   │   ├── regime_detector.py      # Market Regime Detection (RF)
│   │   └── volatility_predictor.py # Volatility Prediction (LSTM)
│   ├── strategies/
│   │   ├── base_grid.py            # Classic Grid Strategy
│   │   └── adaptive_grid.py        # AI Adaptive Grid Strategy
│   ├── risk/
│   │   ├── stop_loss.py            # Stop Loss management
│   │   ├── drawdown_protection.py  # Drawdown protection
│   │   ├── circuit_breaker.py      # Circuit breaker
│   │   └── position_limit.py       # Position sizing
│   ├── backtesting/
│   │   ├── engine.py               # Backtest engine
│   │   └── metrics.py              # Performance metrics
│   └── utils/
│       ├── config.py               # Config loader
│       └── logger.py               # Logging setup
│
├── 📁 scripts/                     # Executable scripts
│   ├── data_collection.py          # ดาวน์โหลดข้อมูล
│   ├── train_models.py             # Train AI models
│   ├── backtest.py                 # รัน backtesting
│   └── optimize.py                 # Parameter optimization
│
├── 📁 tests/                       # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_strategies.py
│
└── 📁 docs/                        # เอกสารโปรเจกต์ (ภาษาไทย)
    ├── 00_TABLE_OF_CONTENTS.md
    ├── 01_PROJECT_OVERVIEW.md
    ├── 02_TIMELINE_GANTT.md
    ├── 03_PHASE1_PREPARATION.md
    ├── 04_PHASE2_DEVELOPMENT.md
    ├── 05_PHASE3_AI_MODELS.md
    ├── 06_PHASE4_RISK_MANAGEMENT.md
    ├── 07_PHASE5_OPTIMIZATION.md
    ├── 08_PHASE6_PRODUCTION.md
    ├── 09_CHECKLIST_TRACKER.md
    ├── 10_KPI_METRICS.md
    ├── 11_RESOURCES_TOOLS.md
    ├── 12_RISK_MITIGATION.md
    └── 13_COMPLETE_PROJECT_PLAN.md
```

---

## 🎯 Performance Targets

| Metric | Classic Grid | **AI Adaptive Grid** | Buy & Hold |
|--------|-------------|---------------------|------------|
| Annual Return | 12–18% | **20–35%** | 8–12% |
| Sharpe Ratio | 1.0–1.5 | **1.8–2.5** | 0.6–0.9 |
| Max Drawdown | -20% | **< -15%** | -30% |
| Win Rate | 55–65% | **65–75%** | N/A |
| Profit Factor | 1.3–1.6 | **1.8–2.5** | N/A |

---

## 🛠️ Technology Stack

| หมวดหมู่ | เทคโนโลยี |
|---------|-----------|
| **Language** | Python 3.10+ |
| **Data** | yfinance, pandas, numpy |
| **ML/AI** | scikit-learn, TensorFlow/Keras, XGBoost |
| **Technical Analysis** | ta (Technical Analysis library) |
| **Backtesting** | vectorbt, custom engine |
| **Visualization** | matplotlib, plotly, seaborn |
| **Configuration** | PyYAML |
| **Testing** | pytest, pytest-cov |
| **Code Quality** | black, flake8 |

---

## 📅 Development Timeline

- **Phase 1 (Week 1-2)**: การเตรียมการและ Data Collection
- **Phase 2 (Week 3-4)**: พัฒนา Core System และ Backtesting Framework
- **Phase 3 (Week 5-7)**: สร้างและ Train AI Models
- **Phase 4 (Week 8-9)**: ระบบ Risk Management ครบวงจร
- **Phase 5 (Week 10)**: Parameter Optimization
- **Phase 6 (Week 11-12)**: Production Deployment

รายละเอียดแต่ละ Phase ดูได้ที่ [docs/](docs/)

---

## 📖 เอกสารประกอบ

| ไฟล์ | รายละเอียด |
|------|-----------|
| [docs/01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md) | ภาพรวมโปรเจกต์ละเอียด |
| [docs/02_TIMELINE_GANTT.md](docs/02_TIMELINE_GANTT.md) | Timeline & Gantt Chart |
| [docs/03_PHASE1_PREPARATION.md](docs/03_PHASE1_PREPARATION.md) | Phase 1: การเตรียมการ |
| [docs/10_KPI_METRICS.md](docs/10_KPI_METRICS.md) | KPI & Performance Metrics |
| [docs/12_RISK_MITIGATION.md](docs/12_RISK_MITIGATION.md) | การบริหารความเสี่ยง |

---

## 🧪 การรัน Tests

```bash
# รัน tests ทั้งหมด
pytest tests/ -v

# รันพร้อม coverage report
pytest tests/ --cov=src --cov-report=html

# รัน test เฉพาะ module
pytest tests/test_strategies.py -v
```

---

## ⚠️ ข้อควรระวัง

> **คำเตือน**: ระบบนี้พัฒนาขึ้นเพื่อการศึกษาและวิจัยเท่านั้น  
> **ผลการ Backtest ในอดีตไม่ได้รับประกันผลในอนาคต**  
> การลงทุนมีความเสี่ยง ผู้ใช้ควรศึกษาและทำความเข้าใจก่อนนำไปใช้งานจริง

---

## 📄 License

MIT License — ดูรายละเอียดที่ [LICENSE](LICENSE)

---

## 👥 Contributing

Pull Requests ยินดีต้อนรับ! กรุณาอ่าน [CONTRIBUTING.md](CONTRIBUTING.md) ก่อนส่ง PR

---

*พัฒนาด้วย ❤️ สำหรับนักเทรดทองคำชาวไทย*
