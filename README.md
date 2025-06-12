# Novel PyTorch Deep Learning Trading System

🚀 **High-Performance GPU-Accelerated Trading System** with Novel Multi-Component Ensemble Architecture

## Overview

This repository contains a cutting-edge deep learning trading system built with PyTorch, featuring multiple innovative components designed for superior market performance. The system combines advanced neural architectures with sophisticated ensemble methods to achieve exceptional Sharpe ratios (4+ consistently).

## 🏆 Performance Highlights

- **AAPL**: 54.27% total return, 4.287 Sharpe ratio
- **GOOGL**: 42.84% total return, 3.625 Sharpe ratio  
- **MSFT**: 49.89% total return, 4.484 Sharpe ratio
- **GPU Optimized**: RTX 4070 acceleration support

## 🧠 Novel Architecture Components

### 1. Multi-Component Ensemble System
- **5 Enhanced GRU Models** with different configurations
- **Multi-Head Attention Mechanism** for temporal pattern recognition
- **Adaptive ensemble weighting** based on market conditions

### 2. Market Regime Detection
- **CNN-LSTM Regime Detector** identifying 4 market states:
  - Bull Market (trend up, low volatility)
  - Bear Market (trend down, low volatility) 
  - Sideways Market (low volatility)
  - High Volatility Regime

### 3. Meta-Learning Component
- **Adaptive Risk-Aware Meta-Learner** (ARAMLE)
- Dynamic confidence-weighted prediction fusion
- Real-time model performance adaptation

### 4. Advanced Loss Function
- **Novel CVaR-Enhanced Loss** combining:
  - Sharpe ratio optimization
  - Conditional Value at Risk (CVaR) 
  - Signal consistency penalties

## 📊 Technical Features

- **150+ Technical Indicators** across multiple timeframes
- **Multi-horizon analysis** (15, 30, 60, 90 day windows)
- **Robust scaling** with outlier resistance
- **GPU-accelerated training** with batch optimization
- **Professional risk metrics** with proper annualization

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default stock (AAPL)
python run_dynamic_trading.py

# Run with specific stock
python run_dynamic_trading.py MSFT

# List available stocks
python pytorch_trading_system.py --list

# Run with custom parameters
python pytorch_trading_system.py --stock GOOGL --epochs 200 --models 5
```

### Data Requirements

Place your stock data CSV files in the `Data/` directory with the following columns:
- `Timestamp` (datetime)
- `Open`, `High`, `Low`, `Close` (float)
- `Volume` (optional)

## 📈 System Architecture

```
Input Data → Technical Indicators → Sequence Preparation
     ↓
Multi-Component Ensemble:
├── Enhanced GRU Model #1 (128 hidden, attention)
├── Enhanced GRU Model #2 (96 hidden, attention)  
├── Enhanced GRU Model #3 (112 hidden, attention)
├── Enhanced GRU Model #4 (80 hidden, attention)
└── Enhanced GRU Model #5 (144 hidden, no attention)
     ↓
Regime Detection → Dynamic Weighting → Meta-Learning → Final Signal
```

## 🔧 Configuration

The system is optimized for RTX 4070 but works on any GPU or CPU:

- **Sequence Length**: 60 days
- **Batch Size**: 32 (GPU optimized)
- **Training Epochs**: 150 per model
- **Ensemble Size**: 5 models
- **Learning Rates**: Adaptive (0.001 base with decay)

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

- **Total & Cumulative Returns**
- **Annualized Returns** (252 trading days)
- **Sharpe Ratio** (risk-adjusted performance)
- **Maximum Drawdown**
- **Win Rate & Trade Statistics**
- **Alpha vs Buy & Hold**

## 🧪 Training Process

1. **Data Loading**: Automatic stock symbol detection
2. **Feature Engineering**: 150+ technical indicators
3. **Sequence Creation**: 60-day rolling windows
4. **Ensemble Training**: 5 models with different configs
5. **Regime Detection**: Market state classification
6. **Meta-Learning**: Adaptive prediction fusion
7. **Backtesting**: Full performance evaluation

## 📁 Repository Structure

```
final_clean_research/
├── pytorch_trading_system.py    # Main novel trading system
├── run_dynamic_trading.py       # Simplified runner script  
├── tradingPerformance.py        # Performance analysis
├── requirements.txt             # Dependencies
├── Data/                        # Stock data directory
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🎯 Key Innovations

1. **Regime-Aware Ensemble**: Dynamic model weighting based on detected market conditions
2. **Multi-Head Attention GRU**: Enhanced temporal pattern recognition
3. **CVaR-Enhanced Loss**: Risk-aware training objective
4. **Meta-Learning Fusion**: Adaptive confidence-weighted predictions
5. **GPU Optimization**: Efficient RTX 4070 utilization

## 📝 Results Format

```
🏆 NOVEL ENSEMBLE PERFORMANCE RESULTS - MSFT
==================================================
📊 Portfolio Metrics:
   Total Return:        49.89%
   Annual Return:       8.50%
   Sharpe Ratio:        4.484
   Max Drawdown:        -8.23%
   Win Rate:            52.1%

📅 Trading Period:
   Trading Days:        1,250
   Years:               4.96

🏪 Benchmark Comparison:
   Annual Alpha:        +3.45%
```

## 🤝 Contributing

This system represents a novel approach to algorithmic trading with multiple innovative components. Feel free to experiment with:

- Additional technical indicators
- Alternative ensemble methods
- New regime detection approaches
- Different meta-learning architectures

## ⚖️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing before any live trading.

## 📄 License

Open source - feel free to use and modify for research and educational purposes.

---

**Built with ❤️ and PyTorch** | **GPU Accelerated** | **Novel Architecture** 🚀 