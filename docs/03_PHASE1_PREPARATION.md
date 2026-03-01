# 🚀 Phase 1: การเตรียมการ (สัปดาห์ 1-2)

**ระยะเวลา:** Day 1-14  
**เป้าหมาย:** Setup environment, เก็บข้อมูล, และสร้าง Classic Grid Prototype

---

## Day 1: Setup Environment

### 1.1 สร้าง Virtual Environment

```bash
# Clone repository
git clone https://github.com/yourusername/gold-grid-trading-bot.git
cd gold-grid-trading-bot

# สร้าง virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# หรือ venv\Scripts\activate  # Windows

# ติดตั้ง dependencies
pip install -r requirements.txt
pip install -e .
```

### 1.2 ตรวจสอบการติดตั้ง

```bash
python -c "import pandas; import tensorflow; import yfinance; print('✅ All imports OK')"
```

### 1.3 Setup Git Hooks (Optional)

```bash
# ติดตั้ง pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Day 2-3: Data Collection

### 2.1 ดาวน์โหลดข้อมูล GLD

```bash
python scripts/data_collection.py --start 2004-01-01 --end 2024-12-31
```

ผลลัพธ์ที่คาดหวัง:
- `data/raw/GLD_2004-01-01_2024-12-31.csv` (~5,000 แถว)
- `data/raw/SPY_...csv`
- `data/raw/TLT_...csv`

### 2.2 ตรวจสอบข้อมูล

```python
import pandas as pd
df = pd.read_csv("data/raw/GLD_2004-01-01_2024-12-31.csv", index_col=0, parse_dates=True)
print(f"Shape: {df.shape}")
print(f"Missing: {df.isnull().sum()}")
print(df.describe())
```

---

## Day 4-5: Exploratory Data Analysis (EDA)

### 3.1 วิเคราะห์ข้อมูลเบื้องต้น

**สิ่งที่ต้องตรวจสอบ:**
- Distribution ของ daily returns
- Autocorrelation (ราคามี memory ไหม?)
- Seasonal patterns (มีรูปแบบรายปีไหม?)
- Volatility clustering
- Correlation กับ macro indicators

### 3.2 Key Statistics ที่คาดหวัง

| Metric | ค่าที่คาดหวัง |
|--------|-------------|
| Annual Return (avg) | ~8-12% |
| Annual Volatility | ~15-20% |
| Max Drawdown | ~-40% (2011-2015) |
| Skewness | เล็กน้อย negative |
| Kurtosis | > 3 (fat tails) |

---

## Day 6-7: Architecture Design

### 4.1 System Architecture

```
┌──────────────────────────────────────────────────────┐
│  Input: GLD OHLCV + Macro Indicators                 │
└──────────────────────────┬───────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Features   │
                    │  Engineer   │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌─────────┐   ┌────────────┐  ┌──────────┐
      │ Regime  │   │ Volatility │  │  Signal  │
      │Detector │   │ Predictor  │  │Generator │
      └────┬────┘   └─────┬──────┘  └────┬─────┘
           └───────────────┼─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Adaptive   │
                    │    Grid     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    Risk     │
                    │  Manager   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Output    │
                    │  (Orders)   │
                    └─────────────┘
```

---

## Day 8-10: Classic Grid Prototype

### 5.1 ทดสอบ Classic Grid

```python
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.strategies.base_grid import BaseGridStrategy

# โหลดข้อมูล
loader = DataLoader("data/raw/")
df = loader.load_local_data("data/raw/GLD_2004-01-01_2024-12-31.csv")

# ทำความสะอาด
processor = DataProcessor()
df = processor.clean_data(df)
df = processor.handle_missing_values(df)

# รัน Classic Grid
params = {
    "num_grids": 10,
    "grid_spacing_pct": 0.02,
    "initial_capital": 100_000,
    "position_size_pct": 0.05,
}

strategy = BaseGridStrategy(params)
results = strategy.run(df.iloc[-500:])  # 2 ปีล่าสุด

print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
print(f"Total Return: {results['return_pct'].iloc[-1]:.2%}")
```

---

## ✅ Checkpoint Phase 1

ก่อนเริ่ม Phase 2 ต้องผ่านเกณฑ์เหล่านี้:

- [ ] ติดตั้ง environment สำเร็จ ไม่มี import errors
- [ ] ข้อมูล GLD 20 ปี ดาวน์โหลดและ validate ผ่าน
- [ ] EDA notebook สมบูรณ์ มี insights อย่างน้อย 5 ข้อ
- [ ] Classic Grid รันได้บน test data
- [ ] Unit tests สำหรับ DataLoader ผ่านทั้งหมด
- [ ] README.md อัปเดตแล้ว
