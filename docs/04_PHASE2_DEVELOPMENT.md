# 🔨 Phase 2: Core Development (สัปดาห์ 3-4)

**ระยะเวลา:** Day 11-28  
**เป้าหมาย:** Technical Indicators, Backtesting Framework, Unit Testing, Walk-Forward Validation

---

## Day 11-13: Technical Indicators

### สร้าง Feature Set ครบชุด

ใช้ `src/features/engineering.py`:

```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

print(f"จำนวน features: {len(engineer.feature_columns)}")
# คาดหวัง: 60-80 features
```

### รายการ Indicators ที่ต้องมี

| หมวดหมู่ | Indicators | จำนวน |
|---------|-----------|-------|
| Trend | EMA(10,20,50,200), SMA(20,50,200), MACD, ADX, PSAR | ~20 |
| Momentum | RSI(7,14,21), Stochastic, Williams %R, CCI, ROC | ~15 |
| Volatility | ATR(7,14,21), BB Width, Hist Vol, Keltner Channel | ~15 |
| Volume | OBV, MFI, CMF, Volume Ratio | ~8 |
| Pattern | Body size, Shadow, Higher High/Lower Low | ~10 |

---

## Day 14-15: Backtesting Framework

### ทดสอบ BacktestEngine

```python
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics
from src.strategies.base_grid import BaseGridStrategy

config = BacktestConfig(
    initial_capital=100_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
)

strategy = BaseGridStrategy(params)
engine = BacktestEngine(df, strategy, config=config)
results = engine.run()

# คำนวณ metrics
metrics = PerformanceMetrics()
report = metrics.generate_report(
    equity_curve=results["equity"],
    strategy_name="Classic Grid"
)
```

### Metrics ที่ต้องคำนวณ

- ✅ Total Return, CAGR
- ✅ Sharpe Ratio, Sortino Ratio, Calmar Ratio
- ✅ Max Drawdown, Duration
- ✅ Win Rate, Profit Factor
- ✅ Benchmark Comparison

---

## Day 16-18: Unit Testing

### รัน Test Suite

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Test Coverage Targets

| Module | Target Coverage |
|--------|----------------|
| `src/data/` | ≥ 90% |
| `src/features/` | ≥ 85% |
| `src/strategies/` | ≥ 85% |
| `src/risk/` | ≥ 90% |
| `src/backtesting/` | ≥ 80% |
| **รวม** | **≥ 80%** |

---

## Day 19-20: Walk-Forward Validation

### Walk-Forward Validation

```
ข้อมูลทั้งหมด (20 ปี):
[Train 18mo][Test 3mo] → Fold 1
      [Train 18mo][Test 3mo] → Fold 2
            [Train 18mo][Test 3mo] → Fold 3
                  ...
```

### ผลลัพธ์ที่คาดหวัง

| Fold | Period | Sharpe |
|------|--------|--------|
| 1 | 2006-2007 | 1.2-2.0 |
| 2 | 2007-2008 | 0.8-1.5 |
| ... | ... | ... |
| Mean | - | **≥ 1.3** |

---

## ✅ Checkpoint Phase 2

- [ ] Features ≥ 60 columns, ไม่มี NaN
- [ ] BacktestEngine รันได้ คำนวณ metrics ถูกต้อง
- [ ] Unit tests ผ่าน ≥ 80% coverage
- [ ] Walk-Forward Validation: mean Sharpe ≥ 1.0
- [ ] Classic Grid Sharpe ≥ 1.0 บน test set
