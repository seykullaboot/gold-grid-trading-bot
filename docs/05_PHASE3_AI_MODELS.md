# 🤖 Phase 3: AI Models (สัปดาห์ 5-7)

**ระยะเวลา:** Day 29-49  
**เป้าหมาย:** Train Market Regime Detector และ Volatility Predictor แล้ว Integrate

---

## Week 5: Market Regime Detection

### 5.1 สร้าง Labels สำหรับ Training

```python
# Rules สำหรับสร้าง labels:
# ADX > 25 + vol ปกติ  → Trending (0)
# ADX < 20             → Ranging (1)
# vol > 90th percentile → Volatile (2)
```

### 5.2 Train Random Forest

```python
from src.models.regime_detector import MarketRegimeDetector

detector = MarketRegimeDetector(n_estimators=200, max_depth=10)
X_train, y_train = detector.prepare_features(train_df)
detector.train(X_train, y_train)

# ประเมิน
val_results = detector.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
```

### 5.3 เป้าหมาย

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 75% |
| F1-Score (weighted) | ≥ 0.72 |
| Regime Balance | ทั้ง 3 classes มี > 15% |

---

## Week 6: Volatility Prediction (LSTM)

### 6.1 LSTM Architecture

```
Input: (30 timesteps × 10 features)
  ↓
LSTM(64 units) → Dropout(0.2)
  ↓
LSTM(32 units) → Dropout(0.2)
  ↓
Dense(16) → ReLU
  ↓
Dense(1) → Linear (predicted volatility)
```

### 6.2 Train LSTM

```python
from src.models.volatility_predictor import VolatilityPredictor

predictor = VolatilityPredictor(
    sequence_length=30,
    lstm_units=[64, 32],
    dropout_rate=0.2,
    learning_rate=0.001,
)

X, y = predictor.prepare_sequences(df, target_col="hist_vol_20d")
history = predictor.train(X_train, y_train, epochs=100)
```

### 6.3 เป้าหมาย

| Metric | Target |
|--------|--------|
| MAPE | < 15% |
| MAE | < 0.02 |
| Direction Accuracy | > 65% |

---

## Week 7: Integration

### 7.1 สร้าง AI Predictions DataFrame

```python
# สร้าง predictions สำหรับทั้ง test period
predictions = pd.DataFrame(index=test_df.index)
predictions["regime"] = detector.predict_regime_series(test_df)
predictions["predicted_volatility"] = vol_series

# รัน Adaptive Grid
strategy = AdaptiveGridStrategy(params)
results = strategy.run(test_df, ai_predictions=predictions)
```

### 7.2 เปรียบเทียบ Classic vs Adaptive

| Metric | Classic Grid | AI Adaptive |
|--------|-------------|-------------|
| Sharpe | 1.2 | **≥ 1.8** |
| Total Return | 15% | **≥ 22%** |
| Max DD | -18% | **≤ -14%** |

---

## ✅ Checkpoint Phase 3

- [ ] Regime Detector: accuracy ≥ 75%
- [ ] Volatility Predictor: MAPE < 15%
- [ ] AI Adaptive Grid: Sharpe ≥ 1.8 (test set)
- [ ] Models บันทึกใน `data/models/`
- [ ] Integration tests ผ่าน
