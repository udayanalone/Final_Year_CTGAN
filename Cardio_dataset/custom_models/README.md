# Custom GAN Models

This folder contains custom implementations of privacy-preserving GAN models, specifically designed for research on data anonymization.

## 🔬 Research-Focused Models

### 1. **Custom DP-GAN** (Differential Privacy GAN)
- **File:** `dp_pate_train.py` (function: `train_dp_gan`)
- **Privacy Guarantee:** ε-differential privacy
- **Key Features:**
  - Gradient clipping for privacy
  - Gaussian noise injection
  - Conditional generation
  - Configurable privacy parameters

**Research Parameters:**
- `epsilon` - Privacy budget (lower = more private)
- `delta` - Failure probability
- `sigma` - Noise scale
- `clip_norm` - Gradient clipping threshold

### 2. **Custom PATE-GAN** (Private Aggregation of Teacher Ensembles)
- **File:** `dp_pate_train.py` (function: `train_pate_gan`)
- **Privacy Guarantee:** Privacy through ensemble learning
- **Key Features:**
  - Multi-teacher discriminator architecture
  - Noisy aggregation of teacher votes
  - Student generator training
  - Configurable ensemble size

**Research Parameters:**
- `teachers` - Number of teacher discriminators
- `teacher_steps` - Training steps for teachers
- `student_steps` - Training steps for student generator
- `sigma_votes` - Noise in vote aggregation

## 🚀 Usage

### Run Both Custom Models
```bash
python dp_pate_train.py --dpgan-steps 1500 --pategan-teacher-steps 600 --pategan-student-steps 900
```

### Run Only DP-GAN
```bash
python dp_pate_train.py --dpgan-steps 1500 --dp-sigma 0.5 --dp-clip 1.0
```

### Run Only PATE-GAN
```bash
python dp_pate_train.py --pategan-teacher-steps 600 --pategan-student-steps 900 --pate-sigma 2.0
```

## 🔧 Architecture Details

### DP-GAN Architecture
- **Generator:** MLP with conditional input (label concatenation)
- **Discriminator:** MLP with conditional input
- **Privacy Mechanism:** 
  - Gradient clipping (max norm)
  - Gaussian noise addition to gradients
  - Conditional training for better utility

### PATE-GAN Architecture
- **Teachers:** Multiple discriminator networks trained on data shards
- **Student:** Single generator network
- **Privacy Mechanism:**
  - Data sharding across teachers
  - Noisy aggregation of teacher votes
  - Student training on aggregated (noisy) feedback

## 📊 Research Applications

### Privacy-Utility Trade-offs
- Study the impact of privacy parameters on data quality
- Compare different noise levels and their effects
- Analyze the relationship between ensemble size and privacy

### Model Comparison
- Compare DP-GAN vs PATE-GAN vs standard CTGAN
- Evaluate different privacy mechanisms
- Study the impact of conditional generation

### Dataset-Specific Analysis
- How do privacy mechanisms affect cardiovascular data patterns?
- Which features are most/least preserved under privacy constraints?
- Impact of privacy on downstream classification tasks

## 📈 Evaluation Metrics

Models are evaluated using:
- **AUC-ROC** (Area Under ROC Curve)
- **AUC-PR** (Area Under Precision-Recall Curve)
- **Baseline comparison** (real data vs synthetic data)

## 🔬 Research Recommendations

1. **Start with DP-GAN** - Provides mathematical privacy guarantees
2. **Experiment with parameters** - Test different privacy budgets
3. **Compare with baselines** - Use library implementations for comparison
4. **Analyze trade-offs** - Study privacy vs utility relationships
5. **Document findings** - Track all parameter settings and results
