# Anomaly Detection in Industrial Sensor Streams

Unsupervised anomaly detection on the **SWaT (Secure Water Treatment) dataset** using LSTM and TCN Autoencoders. Trained exclusively on normal sensor data; anomalies are flagged at inference time via reconstruction error thresholding.

**Course:** AI-600: Deep Learning, Spring 2026 — LUMS  
**Authors:** Muhammad Umair · Shujat Ali Khan · Muhammad Saad

---

## Results

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Isolation Forest | 0.7426 | 0.9999 | 0.5907 | 0.9354 | 0.8582 |
| One-Class SVM | 0.8248 | 0.9907 | 0.7064 | 0.8725 | 0.8314 |
| TCN-AE (ours) | 0.8758 | 0.9416 | 0.8186 | 0.9718 | 0.9250 |
| **LSTM-AE (ours)** | **0.8950** | **0.9819** | **0.8239** | **0.9787** | **0.9415** |

All thresholds set at 1% FPR on a normal-only validation set — an operationally realistic budget for continuous ICS monitoring.

> **Key insight:** Mean attack reconstruction error (0.181) is **1,437× higher** than normal (0.000126), confirming the LSTM-AE learns a tight normal manifold. The 17.6% missed recalls are structurally embedded within that manifold — a fundamental limit of reconstruction-based scoring, not a threshold artifact (confirmed via t-SNE latent space analysis).

---

## Problem Setup

Industrial control systems rarely have labeled attack data. This makes the detection problem inherently **unsupervised**: the model must learn normal behavior from clean operational data, then flag deviations at deployment.

We use the **SWaT December 2015 dataset** — a standard ICS anomaly detection benchmark from a real water treatment plant with 41 distinct attack scenarios across 51 sensors and actuators at 1 Hz.

- **Training:** 1,109,678 timesteps of normal data (80% split)
- **Validation:** 277,420 timesteps of normal data (for threshold tuning only)
- **Test:** Validation normal + 54,621 attack timesteps (16.5% attack ratio)
- **Features:** 43 sensors (8 constant actuator columns dropped)

---

## Architecture

### LSTM Autoencoder (primary model)
- **Encoder:** 2-layer LSTM (hidden=64, dropout=0.2) → 16-dim bottleneck via linear projection
- **Decoder:** Bottleneck repeated 30× → 2-layer LSTM → reconstruction (B, 30, 43)
- **Parameters:** 132,667
- **Window:** W=30 (matches physical machine cycle duration); stride=5 for training, stride=1 for evaluation

### TCN Autoencoder (comparison)
- 4 dilated causal TCN blocks (dilations: 1, 2, 4, 8; kernel=3; receptive field=90 timesteps)
- Global average pooling for bottleneck extraction
- **Parameters:** 207,419

### Training
- Loss: MSE reconstruction (squares errors to amplify attack deviations)
- Optimizer: Adam (lr=1e-3) with ReduceLROnPlateau (factor=0.5, patience=3)
- Gradient clipping: max norm=1.0
- Early stopping: patience=5

### Threshold Selection
Candidates from two families evaluated on normal-only validation set:
- Percentile-based candidates
- Gaussian candidates: μ + k·σ for k ∈ [1, 6]

Selected threshold = lowest candidate achieving **FPR ≤ 1%**.

---

## Ablation Study

Four axes ablated to justify architectural choices:

| Axis | Finding |
|---|---|
| **Window size** | F1 improves monotonically (W=10→100); W=30 chosen for physical interpretability (machine cycle) |
| **Bottleneck dim** | Stable across 8–64 (ΔF1=0.016); 16-dim sufficient |
| **Training stride** | F1 varies by only 0.003 across all settings; stride=5 is not a bottleneck |
| **Hidden size × depth** | 3-layer LSTMs collapse (vanishing gradients); hidden=128 + 2-layer achieves best F1 (0.8979) but at 3.7× parameter cost for 0.012 F1 gain |

---

## Repository Structure

```
swat-lstm-anomaly-detection/
├── anomaly_detection.ipynb   # Full pipeline: EDA → training → evaluation → ablations
├── DL_Final_Report.pdf       # Course report with full methodology and results
└── README.md
```

---

## Setup & Usage

### Requirements
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### Data
The SWaT dataset requires access request via [iTrust, SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/). Once approved, place files as:
```
swat-dataset/
├── normal.csv
└── attack.csv
```

Update the paths in the notebook:
```python
NORMAL_PATH = '/path/to/swat-dataset/normal.csv'
ATTACK_PATH  = '/path/to/swat-dataset/attack.csv'
```

### Run
Open `anomaly_detection.ipynb` in Jupyter or Google Colab and run all cells sequentially. The notebook covers:
1. EDA and preprocessing
2. Sliding window sequence creation
3. LSTM-AE and TCN-AE training
4. Threshold selection on validation set
5. Test set evaluation and metric reporting
6. Ablation studies (window size, bottleneck, stride, hidden size × depth)
7. Latent space analysis (PCA + t-SNE)

---

## Key Findings

- **LSTM-AE outperforms all baselines** across F1, Recall, ROC-AUC, and PR-AUC
- **Isolation Forest** achieves near-perfect precision (0.9999) but catastrophically low recall (0.5907) — it detects per-sensor anomalies but misses attacks that violate inter-sensor correlations while staying within individual sensor ranges
- **P4/P5 sensors dominate** feature importance (UV401, P501, FIT501–504) due to their highly correlated cluster structure — manipulating one member causes large multivariate reconstruction error
- **Missed attacks (17.6%)** form a structurally hard subset embedded within the normal manifold — not addressable via threshold tuning alone; association-discrepancy or contrastive objectives are promising directions

---

## Citation

If you use this work, please cite the SWaT dataset:

```
Mathur, A. P. and Tippenhauer, N. O. SWaT: A water treatment testbed for research and training on ICS security. 
In Proc. CySWater Workshop, 2016.
```
