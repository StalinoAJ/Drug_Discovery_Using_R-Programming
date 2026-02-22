
Transferred from https://github.com/Stalinosmj/Drug_Discovery_Using_R-Programming.git

# Drug Discovery Using R Programming - Production Pipeline

## 📋 Project Overview

This production-ready pipeline implements large-scale QSAR (Quantitative Structure-Activity Relationship) modeling to accelerate drug discovery for coronavirus proteases. Leveraging a dataset of 10,000–20,000 compounds from ChEMBL, PubChem, and BindingDB, the project develops and evaluates advanced machine learning and ensemble models for fast and accurate bioactivity prediction. The robust, reproducible workflow integrates all stages from data acquisition, preprocessing, descriptor engineering, and model training to performance benchmarking and output analysis.

### Key Highlights
- Massive curated dataset (~20,000 compounds)
- Multiple sources: ChEMBL, PubChem, BindingDB
- Comprehensive feature engineering: ECFP4 (1024), MACCS (166), Lipinski descriptors (9)
- Parallel/GPU-accelerated model training with R (Random Forest, XGBoost, advanced DNN)
- Industry-scale model evaluation: 10+ individual and ensemble methods
- Best model performance: **Linear Stacking** (R² = 0.892, RMSE = 0.367, MAE = 0.285)

---

## 📰 Abstract

This project develops and benchmarks ML models for drug bioactivity prediction using QSAR methods on coronavirus protease targets. Rigorous preprocessing and descriptor extraction yield a curated feature set, with tree-based models (Random Forest, XGBoost), DNNs, and sophisticated ensemble methods trained in parallel across the full dataset. Actual results demonstrate significant improvement from ensemble learning, with linear stacking achieving R² = 0.892, RMSE = 0.367, and MAE = 0.285. The pipeline is a blueprint for high-throughput, reliable compound prioritization in computational drug discovery.

---

## 📁 Directory Structure

```
project_root/
│
├── data_massive/
│   ├── bioactivity_raw_massive.csv
│   ├── bioactivity_clean_massive.csv
│   ├── lipinski_descriptors_massive_fixed.csv
│   ├── full_dataset_massive.csv
│   ├── train_data_massive.csv
│   ├── test_data_massive.csv
│   └── bioactivity_raw_massive.rds
│
├── results_massive/
│   ├── rf_model_massive.rds
│   ├── rf_predictions.csv
│   ├── xgb_model_massive.json
│   ├── xgb_predictions.csv
│   ├── dnn_model_best.pt
│   ├── dnn_advanced_best.pt
│   ├── dnn_predictions.csv
│   ├── ensemble_comparison.csv
│   ├── xgb_rf_ensemble_predictions.csv
│   ├── all_predictions_comparison.csv
│   ├── all_predictions_with_ensemble.csv
│   ├── model_comparison_massive.csv
│   ├── model_comparison_with_ensemble.csv
│   ├── xgb_rf_ensemble_comparison.csv
│   ├── complete_model_comparison.csv
│   ├── final_complete_model_comparison.csv
│   └── residuals_analysis.csv
│
└── drug-discovery-production.qmd
```

---

## 📈 Main Model Results

| Model                     | RMSE  | MAE   | R²    |
|--------------------------|-------|-------|-------|
| Multi: Linear Stacking   | 0.367 | 0.285 | 0.892 |
| XGB+RF: Stacking         | 0.368 | 0.286 | 0.891 |
| XGBoost (CPU)            | 0.369 | 0.285 | 0.891 |
| XGB+RF: BMA              | 0.414 | 0.309 | 0.873 |
| 🏆 XGB+RF: Weighted       | 0.431 | 0.319 | 0.863 |
| XGB+RF: Equal (50-50)    | 0.447 | 0.330 | 0.854 |
| Multi: Median            | 0.542 | 0.392 | 0.789 |
| Random Forest (1000 trees)| 0.624 | 0.450 | 0.700 |
| Multi: Weighted          | 0.992 | 0.809 | 0.591 |
| Multi: Trimmed Mean      | 1.276 | 1.040 | 0.434 |
| DNN Advanced             | 3.537 | 2.864 | 0.052 |

**Linear stacking ensemble provides the highest accuracy and generalization.**

---

## 🔧 Key Technologies

Core R packages: `rcdk`, `ranger`, `xgboost`, `torch`, `luz`, `tidyverse`, `future`, `furrr`, `plotly`

System requirements: R 4.1+, Java, CUDA 11.8+ (optional)

---

## 🗂️ Main Variables
- SMILES/chEMBL identifiers
- 1199 molecular descriptors (ECFP4, MACCS, Lipinski)
- pActivity and bioactivity classes (target values)
- Ensemble predictions (all major model outputs)

---

## 🧪 Methods & Workflow
- Multi-source data acquisition (API queries)
- Data cleaning, standardization, stratified splitting
- Parallel fingerprint and descriptor extraction
- Individual model training (hyperparameter-tuned)
- Ensemble construction (weighted average, stacking, BMA)
- Output saving, reporting, and visualization (Quarto)

---

## 📤 Outputs
- Saved model files (.rds, .json, .pt)
- Prediction CSVs for all models, test set, and ensembles
- Metric tables, residuals, and comprehensive analysis
- Interactive Quarto report generation

---

## 🚩 Citing & Acknowledgement
This project is provided for academic and research use in computational drug discovery. Please cite as:

> Drug Discovery Using R Programming: Production-Scale QSAR Pipeline for Coronavirus Protease Inhibitor Prediction (2025)

---

**Last updated:** November 12, 2025 | **Version:** 1.0 | **Status:** Complete
