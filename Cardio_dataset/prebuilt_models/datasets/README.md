# Prebuilt Models - Datasets

This folder contains all datasets used by prebuilt/library model implementations.

## 📁 Structure

```
datasets/
├── input/                    # Input datasets
│   └── cardio_train_dataset.csv
└── generated/                # Generated synthetic datasets
    ├── ctgan/                # CTGAN outputs
    ├── pategan/              # PATE-GAN outputs
    └── dp_ctgan/             # DP-CTGAN outputs
```

## 📊 Input Datasets

### `input/cardio_train_dataset.csv`
- **Source:** Original cardiovascular dataset
- **Format:** CSV with semicolon separator
- **Columns:** id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio
- **Size:** ~70,000 records
- **Target:** `cardio` (binary classification)

## 🤖 Generated Datasets

### CTGAN Outputs (`generated/ctgan/`)
- **Model:** Conditional Tabular GAN
- **Privacy:** None (standard GAN)
- **Structure:** `YYYYMMDD_HHMMSS/synthetic.csv`
- **Use case:** Baseline comparison

### PATE-GAN Outputs (`generated/pategan/`)
- **Model:** Private Aggregation of Teacher Ensembles GAN
- **Privacy:** PATE mechanism
- **Structure:** `YYYYMMDD_HHMMSS/synthetic.csv`
- **Use case:** Privacy-preserving baseline

### DP-CTGAN Outputs (`generated/dp_ctgan/`)
- **Model:** Differentially Private CTGAN
- **Privacy:** ε-differential privacy
- **Structure:** `YYYYMMDD_HHMMSS/synthetic.csv`
- **Use case:** Privacy-preserving baseline

## 🔄 Dataset Flow

1. **Input:** `input/cardio_train_dataset.csv` → Model training
2. **Processing:** Models generate synthetic data
3. **Output:** `generated/{model_name}/{timestamp}/synthetic.csv`

## 📈 Usage

### Accessing Input Data
```python
import pandas as pd
df = pd.read_csv("prebuilt_models/datasets/input/cardio_train_dataset.csv", sep=";")
```

### Accessing Generated Data
```python
import pandas as pd
# Latest CTGAN output
df_synthetic = pd.read_csv("prebuilt_models/datasets/generated/ctgan/20250909_120000/synthetic.csv")
```

## 🔍 Dataset Characteristics

### Original Dataset
- **Records:** ~70,000
- **Features:** 12 (including target)
- **Target Distribution:** Binary (0/1)
- **Missing Values:** Minimal
- **Data Types:** Mixed (numeric, categorical)

### Generated Datasets
- **Records:** Same as training data (configurable)
- **Features:** Same structure as original
- **Privacy:** Varies by model (none to strong privacy guarantees)
- **Quality:** Evaluated using AUC metrics

## ⚠️ Important Notes

- **Generated datasets** are timestamped for version control
- **Privacy levels** vary by model implementation
- **Evaluation metrics** are saved alongside datasets
- **Original data** is preserved in input folder
- **Generated data** should be used according to privacy requirements
