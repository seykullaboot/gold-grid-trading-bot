# ⚠️ Risk Mitigation Plan

## 1. ความเสี่ยงทางเทคนิค (Technical Risks)

| ความเสี่ยง | ความน่าจะเป็น | ผลกระทบ | แนวทางแก้ไข |
|-----------|-------------|---------|------------|
| AI Model Overfitting | สูง | สูง | Walk-Forward Validation, Regularization |
| Data Quality Issues | ปานกลาง | สูง | Data validation pipeline |
| Backtesting Bias (Lookahead) | ปานกลาง | สูง | ตรวจสอบ `shift()` ทุกที่ |
| Library Version Conflicts | ต่ำ | ปานกลาง | Pin versions ใน requirements.txt |
| Memory Issues (large dataset) | ต่ำ | ต่ำ | ใช้ chunking, data types optimization |

### การแก้ไข Overfitting

```python
# 1. Cross-Validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# 2. Regularization ใน Random Forest
detector = MarketRegimeDetector(
    min_samples_leaf=10,  # ป้องกัน overfitting
    min_samples_split=20,
)

# 3. Dropout ใน LSTM
predictor = VolatilityPredictor(
    dropout_rate=0.2,  # 20% dropout
)
```

---

## 2. ความเสี่ยงของตลาด (Market Risks)

| ความเสี่ยง | ความน่าจะเป็น | ผลกระทบ | แนวทางแก้ไข |
|-----------|-------------|---------|------------|
| Extreme Market Event (Black Swan) | ต่ำ | สูงมาก | Circuit Breaker, Max DD limit |
| Regime Change ที่ไม่คาดคิด | ปานกลาง | สูง | Adaptive parameters, frequent retraining |
| Liquidity Crisis | ต่ำ | สูง | Position limits, stop loss |
| Currency Risk (USD/THB) | ปานกลาง | ปานกลาง | ใช้ GLD (USD-denominated) |
| Correlation Breakdown | ปานกลาง | ปานกลาง | Diversification, regime detection |

### Contingency Plan สำหรับ Black Swan

```
ถ้า GLD ตก > 10% ในวันเดียว:
  1. Circuit Breaker ทำงานทันที
  2. ปิด positions ทั้งหมด
  3. รอ 24-48 ชั่วโมง
  4. ประเมินสถานการณ์ใหม่
  5. Retrain models ถ้าจำเป็น
```

---

## 3. ความเสี่ยงด้านปฏิบัติการ (Operational Risks)

| ความเสี่ยง | ความน่าจะเป็น | ผลกระทบ | แนวทางแก้ไข |
|-----------|-------------|---------|------------|
| Network Outage | ปานกลาง | สูง | Retry logic, fallback data source |
| API Rate Limiting | สูง | ต่ำ | Caching, exponential backoff |
| Broker API Changes | ต่ำ | สูง | Abstract interface, version pinning |
| Hardware Failure | ต่ำ | ปานกลาง | Cloud deployment, backups |
| Human Error | ปานกลาง | สูง | Code review, unit tests |

---

## 4. Monitoring & Alerts

### Alert Rules

```yaml
# alerts.yaml
alerts:
  - name: "High Drawdown"
    condition: "drawdown > 10%"
    action: "reduce_position_50%"
    
  - name: "Circuit Breaker"
    condition: "daily_loss > 3%"
    action: "stop_trading_24h"
    
  - name: "Anomalous Volatility"
    condition: "vol > 3x historical_average"
    action: "switch_to_safe_mode"
    
  - name: "Model Degradation"
    condition: "regime_accuracy < 60%"
    action: "retrain_model + alert_human"
```

---

## 5. Recovery Procedures

### หลัง Circuit Breaker

1. ตรวจสอบสาเหตุ (log analysis)
2. ประเมินสภาวะตลาดปัจจุบัน
3. ตรวจสอบ model predictions ยังสมเหตุสมผลไหม
4. รีเซ็ต Circuit Breaker (manual หรือ auto หลัง 24h)
5. เริ่มต้นด้วย position size เล็กลง 50%
6. กลับสู่ normal position หลัง 3 วัน

### หลัง Drawdown > 15%

1. หยุดเทรดทันที
2. Analyze trades ที่ขาดทุน
3. Retrain AI models ด้วยข้อมูลใหม่
4. Walk-Forward Validation รอบใหม่
5. อนุมัติ manual ก่อน resume
