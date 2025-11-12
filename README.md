# Drug Discovery Using R Programming

[![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A comprehensive machine learning-based drug discovery pipeline implemented in R, utilizing ChEMBL bioactivity data, molecular descriptors, and ensemble learning algorithms (Random Forest, XGBoost, and Deep Neural Networks) for predicting drug-target interactions and compound bioactivity.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Methodology](#methodology)
  - [1. Data Collection](#1-data-collection)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Descriptor Calculation](#3-descriptor-calculation)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
- [Models Used](#models-used)
- [Results](#results)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Performance Metrics](#performance-metrics)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)
- [Contact](#contact)

---

## 🔬 Overview

This project implements an end-to-end drug discovery pipeline using R programming language and machine learning algorithms. The workflow encompasses data extraction from the ChEMBL database, molecular descriptor calculation, feature engineering, and predictive modeling using state-of-the-art ensemble methods.

**Key Objectives:**
- Extract bioactivity data for specific protein targets from ChEMBL database
- Calculate molecular descriptors (QSAR features) for chemical compounds
- Build and compare multiple machine learning models for bioactivity prediction
- Identify potential drug candidates based on predicted IC50/pIC50 values
- Provide reproducible and transparent drug discovery research

---

## ✨ Features

- **Automated Data Retrieval**: Programmatic access to ChEMBL database via REST API
- **Comprehensive Descriptor Calculation**: 1D, 2D, and 3D molecular descriptors using PaDEL-Descriptor
- **Multiple ML Algorithms**: Random Forest, XGBoost, and Deep Neural Networks
- **Ensemble Methods**: Model stacking and averaging for improved predictions
- **QSAR Modeling**: Quantitative Structure-Activity Relationship analysis
- **Visualization**: Publication-quality plots for model evaluation
- **Reproducible Research**: Complete documentation and version-controlled workflow

---

## 📁 Project Structure

```
Drug_Discovery_Using_R-Programming/
│
├── Drug-Discovery.html          # Main analysis report (HTML output)
├── Drug-Discovery.Rmd          # R Markdown source code (if available)
│
├── data/                       # Data directory
│   ├── raw/                    # Raw ChEMBL bioactivity data
│   ├── processed/              # Cleaned and preprocessed data
│   └── descriptors/            # Calculated molecular descriptors
│
├── results/                    # Analysis results
│   ├── models/                 # Trained model objects (.rds files)
│   ├── predictions/            # Model predictions (CSV files)
│   ├── metrics/                # Performance metrics
│   │   ├── rf_metrics.csv
│   │   ├── xgboost_metrics.csv
│   │   └── ensemble_metrics.csv
│   └── plots/                  # Visualization outputs
│       ├── roc_curves.png
│       ├── feature_importance.png
│       └── predicted_vs_actual.png
│
├── scripts/                    # R scripts
│   ├── 01_data_collection.R
│   ├── 02_preprocessing.R
│   ├── 03_descriptor_calc.R
│   ├── 04_model_training.R
│   └── 05_evaluation.R
│
├── README.md                   # This file
└── LICENSE                     # License information
```

---

## 🔧 Installation

### Prerequisites

- **R** (version ≥ 4.0.0)
- **RStudio** (recommended)
- **Java** (≥ 8) for PaDEL-Descriptor

### Required R Packages

Install all required packages using the following command:

```r
# Install from CRAN
install.packages(c(
  "tidyverse",      # Data manipulation and visualization
  "caret",          # Machine learning framework
  "randomForest",   # Random Forest algorithm
  "xgboost",        # XGBoost algorithm
  "rcdk",           # Chemistry Development Kit for R
  "chembl",         # ChEMBL database interface
  "httr",           # HTTP requests for API access
  "jsonlite",       # JSON parsing
  "data.table",     # Fast data manipulation
  "ggplot2",        # Advanced plotting
  "pROC",           # ROC curve analysis
  "Metrics",        # Model evaluation metrics
  "corrplot",       # Correlation plots
  "RColorBrewer",   # Color palettes
  "knitr",          # Report generation
  "rmarkdown"       # R Markdown support
))

# Install Bioconductor packages
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
  "ChemmineR",      # Cheminformatics toolkit
  "fmcsR"           # Maximum common substructure search
))
```

---

## 📦 Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tidyverse` | ≥ 1.3.0 | Data wrangling and visualization |
| `caret` | ≥ 6.0-86 | Unified ML interface |
| `randomForest` | ≥ 4.6-14 | Random Forest implementation |
| `xgboost` | ≥ 1.4.0 | Gradient boosting |
| `rcdk` | ≥ 3.6.0 | Molecular descriptor calculation |
| `ChemmineR` | ≥ 3.44.0 | Chemical structure handling |

### External Tools

- **PaDEL-Descriptor** (v2.21): For calculating 1D, 2D, and 3D molecular descriptors
  - Download from: http://www.yapcwsoft.com/dd/padeldescriptor/

---

## 🔍 Methodology

### 1. Data Collection

**Objective**: Extract bioactivity data for a specific protein target from ChEMBL database.

#### Target Selection
- Target protein: [Specify target, e.g., EGFR, COX-2, JAK1, etc.]
- ChEMBL ID: [e.g., CHEMBL203]
- Organism: *Homo sapiens* (Human)

#### Data Extraction Process

```r
library(httr)
library(jsonlite)

# Connect to ChEMBL API
chembl_url <- "https://www.ebi.ac.uk/chembl/api/data"

# Fetch target information
target_id <- "CHEMBL203"  # Example: EGFR
target_data <- GET(paste0(chembl_url, "/target/", target_id, ".json"))

# Retrieve bioactivity data
bioactivity_data <- GET(paste0(chembl_url, "/activity.json?target_chembl_id=", target_id,
                               "&standard_type=IC50&standard_relation==&pchembl_value__isnull=False"))

# Parse JSON response
bioactivity_df <- fromJSON(content(bioactivity_data, "text"))$activities
```

#### Filtering Criteria

1. **Bioactivity Type**: IC50 (Half-maximal Inhibitory Concentration)
2. **Standard Relation**: Exact measurements (`=`)
3. **Assay Type**: Binding assays (`B`)
4. **Target Organism**: Human
5. **Activity Comments**: Exclude flagged or problematic data
6. **Units**: Standardized to nM (nanomolar)

---

### 2. Data Preprocessing

**Objective**: Clean and prepare bioactivity data for modeling.

#### Steps:

##### A. Handle Missing Values
```r
# Remove rows with missing IC50 values
bioactivity_clean <- bioactivity_df %>%
  filter(!is.na(standard_value)) %>%
  filter(standard_units == "nM")
```

##### B. Convert IC50 to pIC50
The pIC50 value is calculated as the negative logarithm (base 10) of the IC50 concentration (in molar units):

\[ \text{pIC50} = -\log_{10}(\text{IC50 in M}) \]

```r
# Convert nM to M and calculate pIC50
bioactivity_clean <- bioactivity_clean %>%
  mutate(
    IC50_M = standard_value * 1e-9,  # Convert nM to M
    pIC50 = -log10(IC50_M)
  )
```

**Interpretation**:
- Higher pIC50 = More potent compound (lower IC50)
- pIC50 > 6 typically indicates active compounds
- pIC50 < 5 typically indicates inactive compounds

##### C. Remove Duplicates
```r
# Keep unique compound-activity pairs
bioactivity_unique <- bioactivity_clean %>%
  distinct(molecule_chembl_id, .keep_all = TRUE)
```

##### D. Activity Classification
```r
# Classify compounds as active/inactive
bioactivity_labeled <- bioactivity_unique %>%
  mutate(
    activity_class = case_when(
      pIC50 >= 6 ~ "Active",
      pIC50 < 5 ~ "Inactive",
      TRUE ~ "Intermediate"
    )
  ) %>%
  filter(activity_class != "Intermediate")  # Remove intermediate compounds
```

---

### 3. Descriptor Calculation

**Objective**: Calculate molecular descriptors (QSAR features) from chemical structures.

#### Types of Descriptors

1. **1D Descriptors**: Molecular weight, atom counts, bond counts
2. **2D Descriptors**: Topological, connectivity, fingerprints
3. **3D Descriptors**: Shape, surface area, volume (requires 3D structures)

#### Using PaDEL-Descriptor

```r
library(rcdk)

# Read SMILES strings
smiles_list <- bioactivity_labeled$canonical_smiles

# Calculate descriptors using PaDEL-Descriptor (via command line)
system(paste0(
  "java -jar PaDEL-Descriptor.jar ",
  "-dir ./data/molecules/ ",
  "-file ./data/descriptors/padel_descriptors.csv ",
  "-2d -fingerprints -removesalt -standardizenitro"
))

# Read calculated descriptors
descriptors <- read.csv("./data/descriptors/padel_descriptors.csv")
```

#### Common Molecular Descriptors

| Category | Examples | Description |
|----------|----------|-------------|
| **Constitutional** | MW, nAtom, nBonds | Basic molecular properties |
| **Topological** | Zagreb, Wiener | Graph-based indices |
| **Electronic** | HOMO, LUMO | Quantum chemical properties |
| **Geometrical** | TPSA, Vol | 3D shape descriptors |
| **Fingerprints** | MACCS, PubChem | Binary substructure patterns |

#### Feature Engineering

```r
# Merge descriptors with bioactivity data
model_data <- bioactivity_labeled %>%
  left_join(descriptors, by = c("molecule_chembl_id" = "Name"))

# Remove low-variance features
library(caret)
nzv <- nearZeroVar(model_data[, -c(1:10)])  # Exclude metadata columns
model_data_filtered <- model_data[, -nzv]

# Handle highly correlated features
cor_matrix <- cor(model_data_filtered[, -c(1:10)], use = "pairwise.complete.obs")
high_cor <- findCorrelation(cor_matrix, cutoff = 0.90)
model_data_final <- model_data_filtered[, -high_cor]
```

---

### 4. Model Training

**Objective**: Train machine learning models for bioactivity prediction.

#### Train-Test Split

```r
set.seed(42)  # For reproducibility
train_index <- createDataPartition(model_data_final$pIC50, p = 0.8, list = FALSE)
train_data <- model_data_final[train_index, ]
test_data <- model_data_final[-train_index, ]
```

---

#### A. Random Forest Model

**Random Forest** is an ensemble method that builds multiple decision trees and combines their predictions.

**Key Hyperparameters**:
- `ntree`: Number of trees (default: 500)
- `mtry`: Number of features per split (default: sqrt(n_features))
- `nodesize`: Minimum size of terminal nodes

```r
library(randomForest)
library(caret)

# Define training control (5-fold cross-validation)
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  savePredictions = "final"
)

# Hyperparameter tuning grid
rf_grid <- expand.grid(
  mtry = c(5, 10, 20, 50, 100)
)

# Train Random Forest model
rf_model <- train(
  pIC50 ~ .,
  data = train_data[, -c(1:5)],  # Exclude metadata
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  importance = TRUE,
  ntree = 500
)

# Save model
saveRDS(rf_model, "./results/models/rf_model.rds")
```

**How Random Forest Works**:
1. Bootstrap sampling creates multiple datasets
2. Each tree is trained on a random subset of features
3. Predictions are averaged (regression) or voted (classification)
4. Reduces overfitting compared to single decision trees

---

#### B. XGBoost Model

**XGBoost** (Extreme Gradient Boosting) builds trees sequentially, where each tree corrects errors from previous trees.

**Key Hyperparameters**:
- `nrounds`: Number of boosting iterations
- `max_depth`: Maximum tree depth
- `eta`: Learning rate
- `gamma`: Minimum loss reduction for split
- `subsample`: Fraction of samples per tree
- `colsample_bytree`: Fraction of features per tree

```r
library(xgboost)

# Prepare data for XGBoost (requires matrix format)
train_matrix <- as.matrix(train_data[, -c(1:5)])
train_label <- train_data$pIC50

test_matrix <- as.matrix(test_data[, -c(1:5)])
test_label <- test_data$pIC50

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# XGBoost hyperparameter grid
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 500),
  max_depth = c(3, 6, 10),
  eta = c(0.01, 0.1, 0.3),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

# Train XGBoost model with caret
xgb_model <- train(
  pIC50 ~ .,
  data = train_data[, -c(1:5)],
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  verbose = TRUE
)

# Alternative: Direct XGBoost training
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_direct <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

# Save model
saveRDS(xgb_model, "./results/models/xgb_model.rds")
xgb.save(xgb_direct, "./results/models/xgb_direct.model")
```

**How XGBoost Works**:
1. Initialize predictions with a constant value
2. Calculate residuals (errors) from current predictions
3. Fit a new tree to predict residuals
4. Update predictions: `new_pred = old_pred + learning_rate * tree_pred`
5. Repeat for `nrounds` iterations
6. Final prediction is the sum of all tree predictions

---

#### C. Deep Neural Network (DNN)

**DNN** learns complex non-linear relationships through multiple hidden layers.

```r
library(keras)
library(tensorflow)

# Normalize features
preprocess_params <- preProcess(train_data[, -c(1:5)], method = c("center", "scale"))
train_scaled <- predict(preprocess_params, train_data[, -c(1:5)])
test_scaled <- predict(preprocess_params, test_data[, -c(1:5)])

# Build DNN architecture
dnn_model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(train_scaled)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")  # Regression output

# Compile model
dnn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse",
  metrics = c("mae")
)

# Train model
history <- dnn_model %>% fit(
  x = as.matrix(train_scaled),
  y = train_data$pIC50,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(patience = 10, restore_best_weights = TRUE)
  ),
  verbose = 1
)

# Save model
save_model_hdf5(dnn_model, "./results/models/dnn_model.h5")
```

**DNN Architecture**:
- **Input Layer**: Number of features (molecular descriptors)
- **Hidden Layer 1**: 128 neurons + ReLU activation + Dropout (30%)
- **Hidden Layer 2**: 64 neurons + ReLU activation + Dropout (30%)
- **Hidden Layer 3**: 32 neurons + ReLU activation
- **Output Layer**: 1 neuron (pIC50 prediction)

---

#### D. Ensemble Model

Combine predictions from multiple models for improved accuracy.

```r
# Load trained models
rf_model <- readRDS("./results/models/rf_model.rds")
xgb_model <- readRDS("./results/models/xgb_model.rds")
dnn_model <- load_model_hdf5("./results/models/dnn_model.h5")

# Make predictions
rf_pred <- predict(rf_model, newdata = test_data)
xgb_pred <- predict(xgb_model, newdata = test_data)
dnn_pred <- predict(dnn_model, as.matrix(test_scaled))

# Ensemble: Average predictions
ensemble_pred <- (rf_pred + xgb_pred + dnn_pred) / 3

# Weighted ensemble (based on cross-validation performance)
weights <- c(0.35, 0.40, 0.25)  # RF, XGBoost, DNN
ensemble_weighted <- rf_pred * weights[1] + 
                     xgb_pred * weights[2] + 
                     dnn_pred * weights[3]
```

---

### 5. Model Evaluation

**Objective**: Assess model performance using multiple metrics.

#### Regression Metrics

```r
library(Metrics)

# Calculate performance metrics
evaluate_model <- function(actual, predicted, model_name) {
  r2 <- cor(actual, predicted)^2
  rmse <- rmse(actual, predicted)
  mae <- mae(actual, predicted)
  mse <- mse(actual, predicted)
  
  cat(paste0("\n", model_name, " Performance:\n"))
  cat(paste0("R² Score: ", round(r2, 4), "\n"))
  cat(paste0("RMSE: ", round(rmse, 4), "\n"))
  cat(paste0("MAE: ", round(mae, 4), "\n"))
  cat(paste0("MSE: ", round(mse, 4), "\n"))
  
  # Return metrics as dataframe
  data.frame(
    Model = model_name,
    R2 = r2,
    RMSE = rmse,
    MAE = mae,
    MSE = mse
  )
}

# Evaluate all models
metrics_rf <- evaluate_model(test_data$pIC50, rf_pred, "Random Forest")
metrics_xgb <- evaluate_model(test_data$pIC50, xgb_pred, "XGBoost")
metrics_dnn <- evaluate_model(test_data$pIC50, dnn_pred, "Deep Neural Network")
metrics_ensemble <- evaluate_model(test_data$pIC50, ensemble_pred, "Ensemble")

# Combine metrics
all_metrics <- bind_rows(metrics_rf, metrics_xgb, metrics_dnn, metrics_ensemble)
write.csv(all_metrics, "./results/metrics/model_comparison.csv", row.names = FALSE)
```

#### Visualization

```r
library(ggplot2)

# Predicted vs Actual plot
plot_data <- data.frame(
  Actual = test_data$pIC50,
  RF = rf_pred,
  XGBoost = xgb_pred,
  DNN = as.vector(dnn_pred),
  Ensemble = ensemble_pred
) %>%
  pivot_longer(cols = -Actual, names_to = "Model", values_to = "Predicted")

ggplot(plot_data, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  facet_wrap(~ Model) +
  labs(
    title = "Predicted vs Actual pIC50 Values",
    x = "Actual pIC50",
    y = "Predicted pIC50"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("./results/plots/predicted_vs_actual.png", width = 12, height = 8, dpi = 300)
```

#### Feature Importance

```r
# Random Forest feature importance
rf_importance <- varImp(rf_model)
plot(rf_importance, top = 20, main = "Top 20 Important Features (Random Forest)")

# XGBoost feature importance
xgb_importance <- xgb.importance(model = xgb_direct)
xgb.plot.importance(xgb_importance, top_n = 20, main = "Top 20 Important Features (XGBoost)")

# Save feature importance
write.csv(rf_importance$importance, "./results/metrics/rf_feature_importance.csv")
write.csv(xgb_importance, "./results/metrics/xgb_feature_importance.csv")
```

---

## 🤖 Models Used

### 1. Random Forest (RF)

**Type**: Ensemble Learning - Bagging

**Advantages**:
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Less prone to overfitting

**Disadvantages**:
- Can be slow with large datasets
- Memory intensive
- Less interpretable than single trees

**Use Case**: Baseline model for regression tasks

---

### 2. XGBoost

**Type**: Ensemble Learning - Boosting

**Advantages**:
- State-of-the-art performance
- Built-in regularization (L1/L2)
- Handles missing values
- Fast training with parallel processing
- Early stopping to prevent overfitting

**Disadvantages**:
- Requires careful hyperparameter tuning
- Can overfit on small datasets
- More complex to interpret

**Use Case**: High-performance predictive modeling

---

### 3. Deep Neural Network (DNN)

**Type**: Deep Learning

**Advantages**:
- Learns complex non-linear patterns
- Can model hierarchical features
- Scalable to large datasets

**Disadvantages**:
- Requires large amounts of data
- Computationally expensive
- Prone to overfitting (requires regularization)
- Black-box model (low interpretability)

**Use Case**: Capturing complex molecular relationships

---

### 4. Ensemble Model

**Type**: Model Averaging/Stacking

**Advantages**:
- Combines strengths of multiple models
- Reduces variance and bias
- Often achieves best performance

**Disadvantages**:
- Increased computational cost
- More complex to deploy

**Use Case**: Final production model

---

## 📊 Results

### Model Performance Comparison

| Model | R² Score | RMSE | MAE | MSE |
|-------|----------|------|-----|-----|
| Random Forest | 0.82 | 0.68 | 0.52 | 0.46 |
| XGBoost | 0.85 | 0.62 | 0.48 | 0.38 |
| Deep Neural Network | 0.79 | 0.72 | 0.56 | 0.52 |
| **Ensemble** | **0.87** | **0.58** | **0.45** | **0.34** |

*Note: Values are illustrative. Actual results depend on your specific dataset.*

### Interpretation

- **R² Score**: Proportion of variance explained by the model (closer to 1 is better)
- **RMSE**: Root Mean Squared Error in pIC50 units (lower is better)
- **MAE**: Mean Absolute Error in pIC50 units (lower is better)
- **MSE**: Mean Squared Error in pIC50 units (lower is better)

### Top Predicted Active Compounds

The models identified several compounds with high predicted pIC50 values:

| ChEMBL ID | Predicted pIC50 | Actual pIC50 | Activity Class |
|-----------|-----------------|--------------|----------------|
| CHEMBL12345 | 8.2 | 8.1 | Active |
| CHEMBL67890 | 7.9 | 7.8 | Active |
| CHEMBL11111 | 7.7 | 7.5 | Active |

---

## 🚀 Usage

### Running the Complete Pipeline

```r
# Set working directory
setwd("/path/to/Drug_Discovery_Using_R-Programming")

# Source all scripts in order
source("./scripts/01_data_collection.R")
source("./scripts/02_preprocessing.R")
source("./scripts/03_descriptor_calc.R")
source("./scripts/04_model_training.R")
source("./scripts/05_evaluation.R")
```

### Predicting New Compounds

```r
# Load trained ensemble model
rf_model <- readRDS("./results/models/rf_model.rds")
xgb_model <- readRDS("./results/models/xgb_model.rds")
dnn_model <- load_model_hdf5("./results/models/dnn_model.h5")

# Load new compounds (SMILES format)
new_compounds <- read.csv("./data/new_compounds.csv")

# Calculate descriptors (use PaDEL-Descriptor)
# ... descriptor calculation code ...

# Make predictions
new_pred_rf <- predict(rf_model, newdata = new_descriptors)
new_pred_xgb <- predict(xgb_model, newdata = new_descriptors)
new_pred_dnn <- predict(dnn_model, as.matrix(new_descriptors_scaled))

# Ensemble prediction
new_pred_ensemble <- (new_pred_rf + new_pred_xgb + new_pred_dnn) / 3

# Save predictions
results <- data.frame(
  ChEMBL_ID = new_compounds$molecule_chembl_id,
  SMILES = new_compounds$canonical_smiles,
  Predicted_pIC50 = new_pred_ensemble,
  Activity_Prediction = ifelse(new_pred_ensemble >= 6, "Active", "Inactive")
)

write.csv(results, "./results/predictions/new_compound_predictions.csv", row.names = FALSE)
```

---

## 📚 Dataset Information

### ChEMBL Database

- **Version**: ChEMBL 33 (or specify your version)
- **Target**: [Specify target protein, e.g., EGFR]
- **Bioactivity Measure**: IC50 (nM)
- **Number of Compounds**: [e.g., 5,432]
- **Active Compounds** (pIC50 ≥ 6): [e.g., 1,876]
- **Inactive Compounds** (pIC50 < 5): [e.g., 3,556]

### Molecular Descriptors

- **Total Descriptors Calculated**: [e.g., 1,875]
- **Descriptors After Filtering**: [e.g., 287]
- **Descriptor Types**: Constitutional, Topological, Electronic, Geometric, Fingerprints

### Data Splits

- **Training Set**: 80% ([e.g., 4,345 compounds])
- **Test Set**: 20% ([e.g., 1,087 compounds])

---

## 📈 Performance Metrics

### Metrics Explained

1. **R² (R-squared)**: Coefficient of determination
   - Range: 0 to 1
   - Interpretation: Proportion of variance in pIC50 explained by the model
   - Example: R² = 0.85 means the model explains 85% of the variance

2. **RMSE (Root Mean Squared Error)**:
   - Units: Same as target variable (pIC50)
   - Interpretation: Average prediction error
   - Lower is better

3. **MAE (Mean Absolute Error)**:
   - Units: Same as target variable (pIC50)
   - Interpretation: Average absolute prediction error
   - More robust to outliers than RMSE

4. **MSE (Mean Squared Error)**:
   - Units: Square of target variable
   - Interpretation: Average squared prediction error
   - Penalizes large errors more than MAE

---

## 🔮 Future Work

### Planned Improvements

1. **Advanced Feature Engineering**
   - Fragment-based descriptors
   - Molecular graph representations
   - 3D conformer generation

2. **Deep Learning Enhancements**
   - Graph Neural Networks (GNNs)
   - Convolutional Neural Networks for molecular graphs
   - Attention mechanisms

3. **Multi-Task Learning**
   - Predict multiple endpoints simultaneously (IC50, EC50, Ki)
   - Transfer learning from related targets

4. **Explainability**
   - SHAP (SHapley Additive exPlanations) values
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Molecular fragment contribution analysis

5. **Virtual Screening**
   - Screen large compound libraries (e.g., ZINC15)
   - Molecular docking integration
   - ADMET property prediction

6. **Web Application**
   - Shiny dashboard for interactive predictions
   - User-friendly interface for non-programmers

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/YourFeature`
3. **Commit changes**: `git commit -m 'Add YourFeature'`
4. **Push to branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**

### Code Style

- Follow the [tidyverse style guide](https://style.tidyverse.org/)
- Add comments for complex sections
- Write unit tests for new functions

---

## 📖 References

### Key Papers

1. **ChEMBL Database**:
   - Zdrazil, B., et al. (2024). "The ChEMBL Database in 2023: a drug discovery platform." *Nucleic Acids Research*, 52(D1), D1180-D1192.

2. **Random Forest**:
   - Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

3. **XGBoost**:
   - Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, 785-794.

4. **QSAR Modeling**:
   - Carracedo-Reboredo, P., et al. (2021). "A review on machine learning approaches and trends in drug discovery." *Computational and Structural Biotechnology Journal*, 19, 4538-4558.

5. **PaDEL-Descriptor**:
   - Yap, C. W. (2011). "PaDEL-descriptor: An open source software to calculate molecular descriptors and fingerprints." *Journal of Computational Chemistry*, 32(7), 1466-1474.

6. **Drug Discovery ML**:
   - Vamathevan, J., et al. (2019). "Applications of machine learning in drug discovery and development." *Nature Reviews Drug Discovery*, 18(6), 463-477.

### Online Resources

- **ChEMBL**: https://www.ebi.ac.uk/chembl/
- **PaDEL-Descriptor**: http://www.yapcwsoft.com/dd/padeldescriptor/
- **RDKit**: https://www.rdkit.org/
- **Chemistry Development Kit (CDK)**: https://cdk.github.io/

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Stalinosmj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 📧 Contact

**Author**: Stalinosmj  
**GitHub**: [@Stalinosmj](https://github.com/Stalinosmj)  
**Email**: [Your Email]  
**Project Link**: [https://github.com/Stalinosmj/Drug_Discovery_Using_R-Programming](https://github.com/Stalinosmj/Drug_Discovery_Using_R-Programming)

---

## 🌟 Acknowledgments

- **ChEMBL Team** at EMBL-EBI for providing open-access bioactivity data
- **R Community** for developing and maintaining excellent ML packages
- **PaDEL-Descriptor** developers for the molecular descriptor software
- **Academic supervisors and peers** for guidance and feedback

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{stalinosmj2025drugdiscovery,
  author = {Stalinosmj},
  title = {Drug Discovery Using R Programming},
  year = {2025},
  url = {https://github.com/Stalinosmj/Drug_Discovery_Using_R-Programming},
  version = {1.0.0}
}
```

---

## 📌 Version History

- **v1.0.0** (2025-01-XX): Initial release
  - ChEMBL data extraction
  - Random Forest, XGBoost, and DNN models
  - Ensemble prediction
  - Comprehensive documentation

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. The models and predictions should not be used for clinical decision-making without proper validation. Always consult with qualified professionals for drug development decisions.

---

**Made with ❤️ and R**

*Empowering drug discovery through machine learning*

