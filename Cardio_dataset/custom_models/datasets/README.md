# Custom Models - Datasets

This folder contains all datasets used by custom model implementations (research-focused).

## 📁 Structure

```
datasets/
├── input/                    # Input datasets
│   └── cardio_train_dataset.csv
└── generated/                # Generated synthetic datasets
    ├── dp_gan/               # Custom DP-GAN outputs
    └── pate_gan/             # Custom PATE-GAN outputs
```

## 📊 Input Datasets

### `input/cardio_train_dataset.csv`
- **Source:** Original cardiovascular dataset
- **Format:** CSV with semicolon separator
- **Columns:** id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio
- **Size:** ~70,000 records
- **Target:** `cardio` (binary classification)

## 🔬 Generated Datasets (Research-focused)

### Custom DP-GAN Outputs (`generated/dp_gan/`)
- **Model:** Custom Differential Privacy GAN
- **Privacy:** ε-differential privacy with gradient clipping
- **Structure:** `YYYYMMDD_HHMMSS/synthetic.csv`
- **Research Value:** ⭐⭐⭐⭐⭐ (Highest)
- **Privacy Parameters:**
  - `epsilon` - Privacy budget
  - `delta` - Failure probability
  - `sigma` - Noise scale
  - `clip_norm` - Gradient clipping

### Custom PATE-GAN Outputs (`generated/pate_gan/`)
- **Model:** Custom Private Aggregation of Teacher Ensembles GAN
- **Privacy:** Teacher-student ensemble with noisy aggregation
- **Structure:** `YYYYMMDD_HHMMSS/synthetic.csv`
- **Research Value:** ⭐⭐⭐⭐
- **Privacy Parameters:**
  - `teachers` - Number of teacher discriminators
  - `teacher_steps` - Training steps for teachers
  - `student_steps` - Training steps for student
  - `sigma_votes` - Noise in vote aggregation

## 🔄 Dataset Flow

1. **Input:** `input/cardio_train_dataset.csv` → Custom model training
2. **Processing:** Custom models with privacy mechanisms generate synthetic data
3. **Output:** `generated/{model_name}/{timestamp}/synthetic.csv`
4. **Evaluation:** AUC metrics saved alongside datasets

## 📈 Usage

### Accessing Input Data
```python
import pandas as pd
df = pd.read_csv("custom_models/datasets/input/cardio_train_dataset.csv", sep=";")
```

### Accessing Generated Data
```python
import pandas as pd
# Latest DP-GAN output
df_dp_gan = pd.read_csv("custom_models/datasets/generated/dp_gan/20250909_120000/synthetic.csv")

# Latest PATE-GAN output
df_pate_gan = pd.read_csv("custom_models/datasets/generated/pate_gan/20250909_120000/synthetic.csv")
```

## 🔬 Research Applications

### Privacy-Utility Trade-offs
- **DP-GAN:** Study impact of ε (privacy budget) on data quality
- **PATE-GAN:** Analyze effect of ensemble size on privacy
- **Comparison:** Compare different privacy mechanisms

### Parameter Sensitivity Analysis
- **Noise levels:** Effect of σ on privacy vs utility
- **Training steps:** Impact of training duration on quality
- **Architecture:** Teacher count vs privacy guarantees

### Dataset-Specific Research
- **Cardiovascular patterns:** How privacy affects medical data patterns
- **Feature preservation:** Which features are most/least preserved
- **Downstream tasks:** Impact on classification performance

## 🔍 Dataset Characteristics

### Original Dataset
- **Records:** ~70,000
- **Features:** 12 (including target)
- **Target Distribution:** Binary (0/1)
- **Medical Domain:** Cardiovascular disease prediction

### Generated Datasets
- **Records:** Same as training data (configurable)
- **Features:** Preprocessed (one-hot encoded, scaled)
- **Privacy:** Strong privacy guarantees (ε-DP or PATE)
- **Quality:** Evaluated using AUC-ROC and AUC-PR

## 🚀 Research Workflow

1. **Baseline:** Run custom models with default parameters
2. **Parameter Sweep:** Test different privacy parameters
3. **Comparison:** Compare with prebuilt model baselines
4. **Analysis:** Study privacy-utility trade-offs
5. **Documentation:** Record all parameter settings and results

## ⚠️ Important Notes

- **Research-focused:** These models are designed for privacy research
- **Parameter tracking:** All privacy parameters are logged
- **Reproducibility:** Timestamped outputs for version control
- **Privacy guarantees:** Mathematical privacy guarantees (DP-GAN)
- **Evaluation:** Comprehensive AUC-based evaluation
- **Documentation:** Detailed parameter and result logging
