# ⚙️ Phase 5: Optimization (สัปดาห์ 10)

**ระยะเวลา:** Day 64-70  
**เป้าหมาย:** หา optimal parameters และ validate ด้วย out-of-sample data

---

## Parameter Optimization

### รัน Grid Search

```bash
python scripts/optimize.py --method random_search --n-trials 200
```

### Parameters ที่ Optimize

| Parameter | Min | Max | Step |
|-----------|-----|-----|------|
| num_grids | 5 | 20 | 1 |
| grid_spacing_pct | 0.5% | 6% | 0.5% |
| position_size_pct | 3% | 10% | 1% |
| stop_loss_pct | 3% | 15% | 1% |
| trailing_stop_pct | 2% | 10% | 1% |

### Scoring Function

```python
def scoring_function(params, df):
    """
    ฟังก์ชัน scoring สำหรับ optimization
    Maximize: Sharpe Ratio
    Penalty: ถ้า Max Drawdown > 20%
    """
    sharpe = calculate_sharpe(params, df)
    max_dd = calculate_max_drawdown(params, df)
    
    if max_dd < -0.20:
        sharpe -= (abs(max_dd) - 0.20) * 10
    
    return sharpe
```

---

## Walk-Forward Validation

### ยืนยัน Optimal Parameters

```
Training: 18 เดือน → Test: 3 เดือน (sliding window)

Fold 1: Train 2004-2005, Test Q1 2006
Fold 2: Train 2004-2006, Test Q2 2006
...
Fold N: Train ...-2022, Test 2024
```

### เกณฑ์ผ่าน Walk-Forward

- Mean Sharpe ≥ 1.5
- ≥ 70% ของ folds มี Sharpe > 1.0
- ไม่มี fold ที่ Sharpe < 0.5 เกิน 2 folds ติดต่อกัน

---

## Final Backtest

### รัน Final Validation

```bash
python scripts/backtest.py --config config/config.yaml
```

### Expected Final Results

| Strategy | Sharpe | CAGR | Max DD | Win Rate |
|----------|--------|------|--------|---------|
| Buy & Hold | 0.7 | 9% | -40% | N/A |
| Classic Grid | 1.3 | 16% | -18% | 62% |
| **AI Adaptive** | **≥1.8** | **≥22%** | **≤-15%** | **≥67%** |

---

## ✅ Checkpoint Phase 5

- [ ] Grid Search เสร็จสิ้น ≥ 200 combinations
- [ ] Optimal parameters บันทึกใน `reports/optimal_params.json`
- [ ] Walk-Forward: mean Sharpe ≥ 1.5
- [ ] Final backtest: AI Adaptive Sharpe ≥ 1.8
- [ ] Max DD ≤ -15% ใน final test
