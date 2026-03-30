# Hidden Markov Models for Integrated Sequence Prediction and Biological State Inference

> **A Simulation-Based Study** | Woxsen University × Pondicherry University

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)](https://numpy.org/)
[![Status: Simulation Study](https://img.shields.io/badge/Status-Simulation%20Study-green)]()
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)]()

---

## 📄 Abstract

Hidden Markov Models (HMMs) are a powerful class of probabilistic graphical models for characterising complex dependencies in sequential data where the underlying system states are unobservable. This repository provides the **complete simulation framework** accompanying the paper:

> **Narendra S., Venkatesu B., Padi T.R.** — *"Hidden Markov Models for Integrated Sequence Prediction and Biological State Inference: A Simulation-Based Study"*

The framework integrates classical parameter estimation (Baum-Welch EM), Bayesian methods (Dirichlet-prior HMM), and MCMC sampling, with dedicated extensions to **DNA sequence analysis** and **ceramics phase-transition detection**.

---

## 📁 Repository Structure

```
hmm-simulation/
│
├── README.md                         ← You are here
├── LICENSE
├── requirements.txt
│
├── data/
│   └── HMM_Simulated_Datasets.xlsx   ← All 6 synthetic datasets (editorial submission)
│
├── src/
│   ├── hmm_core/
│   │   ├── __init__.py
│   │   ├── forward_backward.py       ← Forward-Backward algorithm (log-scaled)
│   │   ├── viterbi.py                ← Viterbi decoding
│   │   ├── baum_welch.py             ← ML-EM parameter estimation
│   │   └── bayesian_hmm.py           ← Bayesian-EM with Dirichlet priors
│   │
│   ├── dna/
│   │   ├── cpg_island.py             ← Task 1: CpG island detection
│   │   ├── gene_structure.py         ← Task 2: Gene structure prediction
│   │   ├── splice_site.py            ← Task 3: Splice-site PWM-HMM
│   │   ├── evolutionary.py           ← Task 4: JC69 / K80 substitution models
│   │   └── pair_hmm.py               ← Task 5: Pair-HMM sequence alignment
│   │
│   └── ceramics/
│       └── phase_transition.py       ← Task 6: Phase-transition detection (DTA/TGA)
│
├── notebooks/
│   ├── 01_core_hmm_demo.ipynb
│   ├── 02_cpg_island_detection.ipynb
│   ├── 03_gene_structure_prediction.ipynb
│   ├── 04_splice_site_identification.ipynb
│   ├── 05_evolutionary_simulation.ipynb
│   ├── 06_pair_hmm_alignment.ipynb
│   └── 07_ceramics_phase_transition.ipynb
│
├── experiments/
│   ├── run_convergence_benchmark.py
│   ├── run_decoding_accuracy.py
│   └── run_parameter_recovery.py
│
└── generate_datasets.py              ← Reproduces HMM_Simulated_Datasets.xlsx
```

---

## 🧬 Simulation Tasks

This repository implements **six biologically and industrially motivated simulation tasks**:

### Task 1 — CpG Island Detection
| Parameter | Values |
|-----------|--------|
| States | `{CpG+, CpG-}` |
| Emission | 16 dinucleotides (`AA`…`TT`) |
| `p` (island → non-island) | `{0.001, 0.005, 0.01}` |
| `q` (non-island → island) | `{0.0001, 0.0005, 0.001}` |
| Island length range | 100 bp – 1 kbp |
| **B-HMM MCC** | **0.831 ± 0.026** |

### Task 2 — Gene Structure Prediction
| Parameter | Values |
|-----------|--------|
| States | `{Intergenic, Exon₀, Exon₁, Exon₂, Intron}` |
| Sequence length | 10⁵ bp |
| Genes per sequence | 3 – 12 |
| **B-HMM Nucleotide Sn** | **82.7%** |
| **B-HMM Nucleotide Sp** | **84.1%** |

### Task 3 — Splice-Site Identification
| Parameter | Values |
|-----------|--------|
| Donor PWM width `W_D` | 9 |
| Acceptor PWM width `W_A` | 23 |
| **Donor AUC** | **0.964 ± 0.008** |
| **Acceptor AUC** | **0.948 ± 0.011** |

### Task 4 — Evolutionary Simulation (JC69 / K80)
| Parameter | Values |
|-----------|--------|
| Models | Jukes-Cantor (JC69), Kimura 2-parameter (K80) |
| κ (ts/tv ratio) | `{1, 2, 4}` |
| Branch lengths `t` | `{0.1, 0.3, 0.5, 1.0}` substitutions/site |
| JC69 bias (t ≤ 0.5) | < 2% |
| K80 corrected bias | < 4% |

### Task 5 — Pair-HMM Sequence Alignment
| Sequence Identity | Pair-HMM AUC | Smith-Waterman AUC | ΔAUC |
|:-----------------:|:------------:|:-----------------:|:----:|
| 95% | 0.999 | 0.998 | +0.001 |
| 70% | 0.981 | 0.974 | +0.007 |
| 50% | 0.952 | 0.934 | +0.018 |
| **40%** | **0.921** | 0.887 | **+0.034** |
| **30%** | **0.864** | 0.811 | **+0.053** |

> The Pair-HMM advantage widens in the **twilight zone** (≤ 50% identity).

### Task 6 — Ceramics Phase-Transition Detection
| Model | Accuracy | Macro F1 | Log-Likelihood | Inference |
|-------|:--------:|:--------:|:--------------:|:---------:|
| ML-HMM | 87.4% | 0.863 | -312.7 | 1.2 ms |
| **B-HMM** | **90.1%** | **0.897** | **-298.4** | **1.4 ms** |
| HSMM | 91.6% | 0.912 | -281.9 | 8.7 ms |
| BiLSTM | 93.2% | 0.928 | — | 24.1 ms |

---

## 📊 Dataset Description

The file `data/HMM_Simulated_Datasets.xlsx` contains **6 sheets** of fully synthetic data:

| Sheet | Rows | Columns | Description |
|-------|-----:|:-------:|-------------|
| `README` | — | 2 | Legend, benchmarks, color key |
| `1_CpG_Island_Detection` | 900 | 9 | Nucleotide sequences with per-position CpG state labels |
| `2_Gene_Structure_Prediction` | 1,200 | 8 | Positions with exon/intron/intergenic annotations and reading frames |
| `3_Splice_Site_ID` | 100 | 8 | Donor/acceptor candidates with PWM scores and IC bits |
| `4_Evolutionary_Simulation` | 3,200 | 11 | Per-site substitution events under JC69 and K80 |
| `5_Pair_HMM_Alignment` | 200 | 14 | Homologous/non-homologous sequence pairs with alignment scores |
| `6_Phase_Transition_Detection` | 600 | 13 | DTA/TGA time series with phase labels and posterior probabilities |

**Reproducing the dataset:**
```bash
python generate_datasets.py
# Output: data/HMM_Simulated_Datasets.xlsx
# Random seed: 42 (NumPy default_rng)
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hmm-simulation.git
cd hmm-simulation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy>=1.24
scipy>=1.10
openpyxl>=3.1
matplotlib>=3.7
pandas>=2.0
scikit-learn>=1.3
jupyter>=1.0
```

---

## 🚀 Quick Start

```python
import numpy as np
from src.hmm_core.baum_welch import BaumWelchHMM
from src.hmm_core.bayesian_hmm import BayesianHMM

# --- Define a 2-state HMM (CpG island example) ---
A = np.array([[0.995, 0.005],
              [0.001, 0.999]])          # transition matrix

B = np.array([[0.0921, 0.0761, 0.0714, 0.0598],   # CpG+ emissions (subset)
              [0.0098, 0.0342, 0.0312, 0.0895]])   # CpG- emissions (subset)

pi = np.array([0.05, 0.95])             # initial state distribution

# --- Simulate a sequence ---
from src.dna.cpg_island import simulate_cpg_sequence
seq, true_states = simulate_cpg_sequence(length=10_000, p=0.005, q=0.0005, seed=42)

# --- Train with ML-EM ---
ml_model = BaumWelchHMM(n_states=2, n_obs=4)
ml_model.fit(seq, max_iter=200, tol=1e-6)

# --- Train with Bayesian-EM ---
b_model = BayesianHMM(n_states=2, n_obs=4, alpha0=1.0)
b_model.fit(seq, max_iter=200)

# --- Decode ---
from src.hmm_core.viterbi import viterbi_decode
predicted_states = viterbi_decode(seq, b_model.A, b_model.B, b_model.pi)

# --- Evaluate ---
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(true_states, predicted_states)
print(f"MCC: {mcc:.3f}")   # Expected ≈ 0.831 (B-HMM)
```

---

## 🔬 Algorithms Implemented

### Core HMM Algorithms
| Algorithm | Complexity | Description |
|-----------|:----------:|-------------|
| Forward (log-scaled) | O(N²T) | Likelihood evaluation; scaling factors prevent underflow |
| Backward | O(N²T) | Posterior state computation |
| Viterbi | O(N²T) | Most probable state-sequence decoding |
| Baum-Welch EM | O(N²T × iter) | Maximum-likelihood parameter estimation |
| Bayesian-EM | O(N²T × iter) | Dirichlet-prior regularised estimation |
| Blocked Gibbs MCMC | O(N²T × samples) | Full posterior sampling |

### DNA-Specific Extensions
- **CpG HMM**: 2-state dinucleotide emission model
- **Gene-finder HMM**: 5-state model with frame-aware exon emissions
- **PWM-HMM**: Position-specific weight matrix splice-site model
- **Pair-HMM**: 2D Forward algorithm over `{M, I_x, I_y}` states
- **Substitution models**: JC69 and K80 with analytical distance correction

---

## 📈 Key Results

### Convergence (N=4, T=500, discrete emissions)
| Method | SNR = 5 dB | SNR = 15 dB | SNR = 25 dB |
|--------|:----------:|:-----------:|:-----------:|
| ML-EM | 84.2 ± 18.4 | 47.6 ± 11.2 | 29.3 ± 6.8 |
| **B-EM** | **61.5 ± 12.1** | **38.9 ± 9.4** | **22.7 ± 5.1** |

> B-EM converges **~27% faster** than ML-EM and achieves **+8.3 pp** decoding accuracy at low SNR / short sequences.

### Parameter Recovery
- Estimation error decreases as **O(T⁻¹/²)** consistent with ML theory
- Optimal prior strength: **α₀ ≈ 1.0** (robust across state-space sizes)

---

## 📐 Mathematical Framework

The model is fully parameterised by the triplet **λ = (A, B, π)**:

$$P(q_t \mid q_{t-1}, \ldots, q_1) = P(q_t \mid q_{t-1}) \quad \forall\, t \geq 2 \quad \text{(Markov property)}$$

$$P(o_t \mid o_{t-1}, \ldots, q_T, \ldots, q_1) = P(o_t \mid q_t) \quad \text{(Output independence)}$$

**Bayesian update** (Dirichlet conjugate):

$$P(\mathbf{a}_i \mid \mathbf{O}, \lambda_{-\mathbf{A}}) = \mathrm{Dir}\!\left(\boldsymbol{\alpha}_i + \sum_t \xi_t(i, \cdot)\right)$$

**Matthews Correlation Coefficient** (primary metric for imbalanced annotation):

$$\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

---

## 🧪 Reproducing Experiments

```bash
# Benchmark convergence across estimation strategies
python experiments/run_convergence_benchmark.py \
    --n_states 2 4 8 \
    --seq_len 100 500 2000 \
    --n_replicates 50 \
    --output results/convergence.csv

# Decoding accuracy grid
python experiments/run_decoding_accuracy.py \
    --snr 5 15 25 \
    --output results/decoding_accuracy.csv

# Parameter recovery (Frobenius norm)
python experiments/run_parameter_recovery.py \
    --n_states 4 \
    --alpha0 0.1 1.0 10.0 \
    --output results/param_recovery.csv
```

---

## 📖 Citation

If you use this code or dataset, please cite:

```bibtex
@article{narendra2025hmm,
  title   = {Hidden {Markov} Models for Integrated Sequence Prediction
             and Biological State Inference: A Simulation-Based Study},
  author  = {Narendra, Surnam and Venkatesu, Boya and Padi, Tirupathi Rao},
  journal = {TBD},
  year    = {2025},
  doi     = {TBD}
}
```

---

## 👥 Authors

| Name | Affiliation | Role |
|------|-------------|------|
| **Surnam Narendra** | School of Sciences, Woxsen University | Conceptualization, Methodology, Writing |
| **Boya Venkatesu** | School of Business, Woxsen University | Validation, Review & Editing |
| **Tirupathi Rao Padi** | Dept. of Statistics, Pondicherry University | Validation, Review |

📧 Corresponding author: [surnamnarendra@gmail.com](mailto:surnamnarendra@gmail.com)

---

## 📚 Key References

- Rabiner, L.R. (1989). A tutorial on hidden Markov models. *Proc. IEEE*, 77(2), 257–286.
- Durbin, R. et al. (1998). *Biological Sequence Analysis*. Cambridge University Press.
- Baum, L.E. et al. (1972). An inequality and associated maximization technique. *Inequalities*, 3, 1–8.
- Jukes, T.H. & Cantor, C.R. (1969). Evolution of protein molecules. *Mammalian Protein Metabolism*.
- Kimura, M. (1980). A simple method for estimating evolutionary rates. *J. Mol. Evol.*, 16, 111–120.
- Eddy, S.R. (1998). Profile hidden Markov models. *Bioinformatics*, 14(9), 755–763.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

> **Data availability**: All simulated datasets are fully synthetic (random seed 42) and contain no patient, proprietary, or third-party data. Available upon reasonable request from the corresponding author.

---

<p align="center">
  <sub>Woxsen University, Sangareddy, Telangana 502345, India &nbsp;|&nbsp; Pondicherry University, Kalapet 605014, India</sub>
</p>
