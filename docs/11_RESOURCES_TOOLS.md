# 🔧 Resources & Tools

## 1. Python Libraries

### Data Processing
| Library | Version | การใช้งาน |
|---------|---------|---------|
| pandas | ≥ 2.0 | DataFrame manipulation |
| numpy | ≥ 1.24 | Numerical computing |
| yfinance | ≥ 0.2 | ดาวน์โหลดข้อมูลตลาด |

### Machine Learning
| Library | Version | การใช้งาน |
|---------|---------|---------|
| scikit-learn | ≥ 1.3 | Random Forest, preprocessing |
| TensorFlow | ≥ 2.13 | LSTM neural network |
| Keras | ≥ 2.13 | High-level ML API |
| XGBoost | ≥ 2.0 | Gradient Boosting |

### Technical Analysis
| Library | Version | การใช้งาน |
|---------|---------|---------|
| ta | ≥ 0.11 | 70+ technical indicators |

### Visualization
| Library | Version | การใช้งาน |
|---------|---------|---------|
| matplotlib | ≥ 3.7 | Static charts |
| plotly | ≥ 5.14 | Interactive charts |
| seaborn | ≥ 0.12 | Statistical visualization |

### Backtesting
| Library | Version | การใช้งาน |
|---------|---------|---------|
| vectorbt | ≥ 0.25 | Vectorized backtesting |

---

## 2. Online Resources

### Documentation
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [ta Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [vectorbt Documentation](https://vectorbt.dev/)

### Grid Trading Resources
- Investopedia: Grid Trading Strategy
- CMC Markets: Grid Trading Explained
- Babypips: Grid Trading Guide

### AI for Trading
- Towards Data Science: LSTM for Stock Prediction
- Quantopian Lectures (archived)
- QuantLib Documentation

---

## 3. Books & Papers

### Books
- **"Advances in Financial Machine Learning"** — Marcos López de Prado
- **"Machine Learning for Algorithmic Trading"** — Stefan Jansen
- **"Quantitative Trading"** — Ernie Chan

### Academic Papers
- "Deep Learning for Financial Market Analysis" (2019)
- "Regime Switching Models in Finance" (various)
- "Grid Trading: A Systematic Approach" (various)

---

## 4. Tools & Software

### Development
| Tool | การใช้งาน |
|------|---------|
| VS Code / PyCharm | IDE |
| Jupyter Notebook | EDA, prototyping |
| Git + GitHub | Version control |
| Docker | Containerization |

### Monitoring
| Tool | การใช้งาน |
|------|---------|
| Grafana | Dashboard (production) |
| Prometheus | Metrics collection |
| Python logging | Application logs |

### Data
| Tool | การใช้งาน |
|------|---------|
| Yahoo Finance | Free market data |
| Alpha Vantage | API backup |
| Quandl/Nasdaq | Alternative data |

---

## 5. Useful Commands

```bash
# ดาวน์โหลดข้อมูล
python scripts/data_collection.py --start 2004-01-01

# Train models
python scripts/train_models.py

# รัน backtest
python scripts/backtest.py

# Optimize parameters
python scripts/optimize.py --n-trials 200

# รัน tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/
```
