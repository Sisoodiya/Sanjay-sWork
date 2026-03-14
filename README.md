# Emotion Recognition with TACO Cross-Attention

Multimodal emotion recognition from EEG and ECG signals using Transformer encoders with Token-Channel Compounded (TACO) Cross-Attention fusion.

**Paper:** "Emotion recognition based on multimodal physiological electrical signals"

## Architecture

```
EEG (128Hz, 14-ch)                          ECG (256Hz, 2-ch)
       │                                           │
  Preprocessing                               Preprocessing
  (Notch → Bandpass → ICA)                    (Notch → Bandpass)
       │                                           │
  1s Segmentation                             1s Segmentation
  (128 samples × 14 ch)                      (256 samples × 2 ch)
       │                                           │
  9×9 Spatial Grid                            GAF + RP + MTF
  (128, 9, 9)                                 (64, 64, 6)
       │                                           │
  Spatial Position Encoding                   Patch Embedding (8×8)
       │                                           │
  Temporal Position Encoding                  Temporal Position Encoding
       │                                           │
  Transformer Encoder ×2                      Transformer Encoder ×2
  (128, 64)                                   (64, 64)
       │                                           │
       └──────────── TACO Fusion ──────────────────┘
                         │
              Token Cross-Attention (TCA)
              Channel Cross-Attention (CCA)
              Bidirectional (EEG↔ECG)
                         │
                  Classification Head
                  Dense(128) → Dense(3, softmax)
                         │
                 Low / Medium / High
```

## Dataset

**DREAMER** — 23 subjects, 18 film-clip trials

| Signal | Channels | Sampling Rate |
|--------|----------|---------------|
| EEG    | 14 (Emotiv EPOC) | 128 Hz |
| ECG    | 2 (lead I, II)   | 256 Hz |

**Labels:** Valence, Arousal, Dominance (1–5 scale → 3 classes: Low/Medium/High)

## Project Structure

```
├── train.py                  # Entry point — LOSOCV training loop
├── requirements.txt          # Python dependencies
├── data/
│   └── DREAMER.mat           # Dataset (tracked via Git LFS)
├── results/                  # Output: model weights & confusion matrices
├── notebooks/
│   └── emotion_recognition_colab.ipynb  # Google Colab notebook
└── src/
    ├── config.py             # Hyperparameters and constants
    ├── data_loader.py        # DREAMER.mat parsing
    ├── preprocessing.py      # Filters, ICA, segmentation
    ├── transforms.py         # GAF/RP/MTF for ECG, 9×9 grid for EEG
    ├── spatial_encoding.py   # 2D sinusoidal positional encoding
    ├── feature_extraction.py # EEG/ECG Transformer encoders
    ├── taco_attention.py     # TACO cross-attention fusion
    ├── model.py              # Full model assembly
    ├── evaluate.py           # Metrics and confusion matrices
    └── utils.py              # Seed, class weights, helpers
```

## Setup

### Requirements

- Python 3.8+
- TensorFlow 2.15+
- scipy, scikit-learn, numpy, matplotlib, seaborn

```bash
pip install -r requirements.txt
```

### Data

`DREAMER.mat` (432 MB) is included in this repo via [Git LFS](https://git-lfs.github.com/). It will be downloaded automatically when you clone:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

If Git LFS is not installed, install it first:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Then initialize
git lfs install
```

## Usage

### Command Line

```bash
# Train on a single target (valence/arousal/dominance)
python train.py --target valence

# Train on all three targets
python train.py --target all

# Quick test with 2 subjects
python train.py --target valence --subjects 2

# Custom data path
python train.py --target valence --data /path/to/DREAMER.mat
```

### Google Colab

Open `notebooks/emotion_recognition_colab.ipynb` in Google Colab:

1. Update `REPO_URL` with your GitHub repo URL
2. Run all cells (GPU runtime recommended)

The dataset is pulled automatically from the repo via Git LFS during the clone step.

## Evaluation

Uses **Leave-One-Subject-Out Cross-Validation (LOSOCV)**: for each of the 23 subjects, train on the other 22 and test on the held-out subject. Reports per-subject and overall accuracy, precision, recall, and F1 score (macro-averaged).
