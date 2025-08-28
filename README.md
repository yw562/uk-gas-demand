
# Energy Demand Forecasting Challenge

This repository demonstrates a complete pipeline for forecasting UK domestic gas demand from weather features.

## Features
- Data preprocessing and feature engineering (calendar + weather aggregation)
- Baseline models (seasonal naive, Ridge regression)
- Gradient Boosting regression (tree-based) as final model
- Out-of-time validation (train on 2009–2016, validate on 2017)
- Synthetic demo dataset for reproducibility

## Data
Original datasets are proprietary and not provided here.  
You can generate a toy dataset with:
```bash
python make_demo_data.py
