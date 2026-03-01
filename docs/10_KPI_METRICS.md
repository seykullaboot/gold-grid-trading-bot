# 📊 KPI & Performance Metrics

## 1. ตารางเปรียบเทียบ Strategies

| Metric | Classic Grid | **AI Adaptive Grid** | Buy & Hold | เป้าหมาย AI |
|--------|-------------|---------------------|------------|------------|
| 📈 Annual Return | 12-18% | **20-35%** | 8-12% | **≥ 20%** |
| 📊 Sharpe Ratio | 1.0-1.5 | **1.8-2.5** | 0.6-0.9 | **≥ 1.8** |
| 🔻 Max Drawdown | -15% ถึง -25% | **≤ -15%** | -30% ถึง -45% | **≤ -15%** |
| 🎯 Win Rate | 55-65% | **65-75%** | N/A | **≥ 65%** |
| 💰 Profit Factor | 1.3-1.6 | **1.8-2.5** | N/A | **≥ 1.8** |
| 📉 Sortino Ratio | 1.2-1.8 | **2.0-3.0** | 0.8-1.2 | **≥ 2.0** |
| ⚖️ Calmar Ratio | 0.6-1.0 | **1.2-2.0** | 0.3-0.5 | **≥ 1.2** |
| 🔄 CAGR | 12-18% | **20-30%** | 8-12% | **≥ 20%** |

---

## 2. สูตรคำนวณ (Formulas)

### 2.1 Sharpe Ratio

```
Sharpe = (R_p - R_f) / σ_p × √252

R_p = ผลตอบแทนเฉลี่ยรายวันของ portfolio
R_f = อัตราดอกเบี้ยปลอดความเสี่ยงรายวัน (5%/252)
σ_p = Standard deviation ของ daily returns
√252 = ปัจจัยแปลงเป็น annual
```

### 2.2 Sortino Ratio

```
Sortino = (R_p - R_f) / σ_d × √252

σ_d = Downside deviation (เฉพาะ negative returns)
```

### 2.3 Maximum Drawdown

```
DD_t = (Equity_t - max(Equity_0..t)) / max(Equity_0..t)
Max DD = min(DD_t) สำหรับทุก t
```

### 2.4 Calmar Ratio

```
Calmar = CAGR / |Max Drawdown|
```

### 2.5 Win Rate

```
Win Rate = จำนวน trades ที่กำไร / จำนวน trades ทั้งหมด
```

### 2.6 Profit Factor

```
Profit Factor = Gross Profit / |Gross Loss|
```

### 2.7 Kelly Criterion

```
f* = (b × p - q) / b

b = avg_win / avg_loss  (reward/risk ratio)
p = win rate
q = 1 - p

ใช้ Fractional Kelly = f* × 0.25 (ความปลอดภัย)
```

---

## 3. Benchmark Comparison

### เปรียบเทียบกับ Indices

| Benchmark | Annual Return | Sharpe |
|-----------|--------------|--------|
| GLD Buy & Hold | 9.2% | 0.72 |
| SPY Buy & Hold | 10.8% | 0.78 |
| **AI Adaptive Grid** | **≥ 22%** | **≥ 1.8** |

---

## 4. Risk Metrics

| Risk Metric | ค่าที่ยอมรับได้ | ค่าที่ดีมาก |
|------------|--------------|-----------|
| Value at Risk (95%) | < -2.5%/day | < -1.5%/day |
| Expected Shortfall | < -3.5%/day | < -2.0%/day |
| Beta vs GLD | 0.8-1.2 | 0.5-0.9 |
| Annual Volatility | 15-25% | 12-18% |

---

## 5. Walk-Forward Performance

### เกณฑ์ผ่าน

| Fold Category | เกณฑ์ |
|--------------|-------|
| Mean Sharpe | ≥ 1.5 |
| % Folds Sharpe > 1.0 | ≥ 70% |
| Worst Fold Sharpe | ≥ 0.5 |
| % Profitable Folds | ≥ 75% |
