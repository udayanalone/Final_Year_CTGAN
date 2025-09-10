# Prebuilt GAN Models

This folder contains library-based implementations of various GAN models for synthetic data generation.

## 📁 Structure

```
prebuilt_models/
├── library_implementations/    # Modern library-based implementations
│   └── train_evaluate_gans.py  # Main evaluation script
└── old_implementations/        # Legacy implementations
    ├── ctgan_train.py          # Basic CTGAN
    ├── dp_ctgan_train.py       # DP-CTGAN placeholder
    ├── pate_ctgan_train.py     # PATE-CTGAN placeholder
    └── output/                 # Previous run outputs
```

## 🏗️ Library Implementations

### Main Evaluation Script
- **File:** `library_implementations/train_evaluate_gans.py`
- **Models:** CTGAN, PATE-GAN, DP-CTGAN
- **Libraries:** ydata-synthetic, SDV, ctgan
- **Features:**
  - Multiple CTGAN implementations (fallback support)
  - Comprehensive evaluation with AUC metrics
  - Automatic model selection based on available libraries

**Usage:**
```bash
python library_implementations/train_evaluate_gans.py --ctgan-epochs 50 --pategan-epochs 50 --dpctgan-epochs 50
```

### Available Models

#### 1. **CTGAN (Conditional Tabular GAN)**
- **Libraries:** ydata-synthetic, SDV, ctgan
- **Privacy:** None (standard GAN)
- **Use case:** Baseline comparison
- **Features:** Multiple implementation fallbacks

#### 2. **PATE-GAN (Library)**
- **Library:** ydata-synthetic
- **Privacy:** PATE mechanism
- **Use case:** Privacy-preserving baseline
- **Features:** Teacher-student ensemble

#### 3. **DP-CTGAN (Library)**
- **Library:** ydata-synthetic
- **Privacy:** Differential privacy
- **Use case:** Privacy-preserving baseline
- **Features:** ε-differential privacy guarantees

## 🔧 Old Implementations

### Basic CTGAN
- **File:** `old_implementations/ctgan_train.py`
- **Library:** SDV Single-Table CTGAN
- **Features:** Simple implementation with metadata detection

### DP-CTGAN Placeholder
- **File:** `old_implementations/dp_ctgan_train.py`
- **Note:** Uses CTGAN as placeholder (no actual DP guarantees)
- **Use case:** Legacy compatibility

### PATE-CTGAN Placeholder
- **File:** `old_implementations/pate_ctgan_train.py`
- **Note:** Uses CTGAN as placeholder (no actual PATE mechanism)
- **Use case:** Legacy compatibility

## 🚀 Quick Start

### Run All Library Models
```bash
python library_implementations/train_evaluate_gans.py --ctgan-epochs 50 --pategan-epochs 50 --dpctgan-epochs 50
```

### Run Individual Old Models
```bash
# Basic CTGAN
python old_implementations/ctgan_train.py --data ../../cardio_train_dataset.csv --epochs 50

# DP-CTGAN placeholder
python old_implementations/dp_ctgan_train.py --data ../../cardio_train_dataset.csv --epochs 50

# PATE-CTGAN placeholder
python old_implementations/pate_ctgan_train.py --data ../../cardio_train_dataset.csv --epochs 50
```

## 📊 Output Structure

All models save outputs to:
- `old_implementations/output/` - Model-specific outputs
- `../../Generated_dataset/` - Unified generated datasets

Each run creates:
- `config.json` - Model parameters
- `metrics.json` - Performance metrics
- `result.json` - Run status
- `synthetic.csv` - Generated dataset

## 🔍 Model Selection Logic

The main evaluation script automatically selects the best available implementation:

1. **CTGAN Priority:**
   - ydata-synthetic CTGAN (preferred)
   - ctgan package CTGAN
   - SDV tabular CTGAN
   - SDV single-table CTGAN

2. **Privacy Models:**
   - PATE-GAN: ydata-synthetic PATEGANSynthesizer
   - DP-CTGAN: ydata-synthetic DPCTGANSynthesizer

## ⚠️ Important Notes

- **Old implementations** are placeholders and don't provide actual privacy guarantees
- **Library implementations** provide the most reliable privacy mechanisms
- **Fallback support** ensures compatibility across different environments
- **Evaluation metrics** are consistent across all implementations

## 🔬 Research Usage

For research purposes:
1. **Use library implementations** for reliable baselines
2. **Compare with custom models** for advanced research
3. **Document library versions** for reproducibility
4. **Use consistent evaluation metrics** across all models
