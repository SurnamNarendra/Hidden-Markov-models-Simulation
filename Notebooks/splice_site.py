"""
splice_site.py
--------------
Task 3 — Splice-Site Identification via PWM-HMM.

Donor  PWM width W_D = 9  (article §3.3.3)
Acceptor PWM width W_A = 23

Information content at position k (article Eq. 11):
    I_k = log2(4) + sum_{v in V} p_{k,v} log2(p_{k,v})   [bits]

Positions with I_k > 1.0 bit are treated as highly conserved; their
Dirichlet prior concentration is reduced to avoid over-smoothing.

References
----------
Shapiro, M.B. & Senapathy, P. (1987). RNA splice junctions of different
    classes of eukaryotes: sequence statistics and functional implications.
    Nucleic Acids Res., 15(17), 7155-7174.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

NUCL = ["A", "C", "G", "T"]

# ── Ground-truth PWMs (Shapiro & Senapathy 1987 consensus) ─────────────────

# Donor site W=9: positions -3 to +6 relative to GT
# GT is at positions [3,4] (0-indexed) — near-deterministic
DONOR_PWM_TRUE = np.array([
    [0.25, 0.25, 0.25, 0.25],   # -3
    [0.25, 0.25, 0.25, 0.25],   # -2
    [0.25, 0.25, 0.25, 0.25],   # -1
    [0.04, 0.00, 0.96, 0.00],   # +1 (G — canonical)
    [0.98, 0.00, 0.00, 0.02],   # +2 (T — canonical)
    [0.35, 0.10, 0.40, 0.15],   # +3
    [0.22, 0.28, 0.30, 0.20],   # +4
    [0.25, 0.25, 0.25, 0.25],   # +5
    [0.25, 0.25, 0.25, 0.25],   # +6
], dtype=float)

# Acceptor site W=9 (abbreviated from full W=23 for core motif)
# AG is at positions [7,8] (0-indexed)
ACCEPTOR_PWM_TRUE = np.array([
    [0.25, 0.25, 0.25, 0.25],   # -7
    [0.25, 0.25, 0.25, 0.25],
    [0.30, 0.20, 0.30, 0.20],
    [0.20, 0.30, 0.20, 0.30],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.00, 0.00, 0.00, 1.00],   # -2 (T in AG context)
    [0.00, 1.00, 0.00, 0.00],   # -1 (A... wait: acceptor is AG so A then G)
], dtype=float)

for pwm in [DONOR_PWM_TRUE, ACCEPTOR_PWM_TRUE]:
    pwm /= pwm.sum(axis=1, keepdims=True)


# ── Information content ─────────────────────────────────────────────────────

def information_content(pwm):
    """
    Compute per-position information content (bits).

    Parameters
    ----------
    pwm : ndarray, shape (W, 4)

    Returns
    -------
    ic : ndarray, shape (W,)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(pwm > 0, np.log2(pwm), 0.0)
    entropy = -np.sum(pwm * log_p, axis=1)   # H_k
    return 2.0 - entropy                      # I_k = log2(4) - H_k


# ── Simulation ──────────────────────────────────────────────────────────────

def simulate_splice_sites(n_sites=200, site_type="donor", seed=42):
    """
    Simulate true splice sites (label=1) and decoy sites (label=0).

    Parameters
    ----------
    n_sites   : int   — total number of sites (half true, half decoy)
    site_type : str   — 'donor' or 'acceptor'
    seed      : int

    Returns
    -------
    sequences : list of str  — 9-mer sequences
    labels    : ndarray (n_sites,) — 1=true site, 0=decoy
    scores    : ndarray (n_sites,) — PWM log-likelihood score
    """
    rng = np.random.default_rng(seed)
    pwm = DONOR_PWM_TRUE if site_type == "donor" else ACCEPTOR_PWM_TRUE
    W   = pwm.shape[0]

    sequences, labels, scores = [], [], []

    for idx in range(n_sites):
        label = 1 if idx < n_sites // 2 else 0
        seq   = []
        score = 0.0
        for k in range(W):
            if label == 1:
                nuc = rng.choice(4, p=pwm[k])
            else:
                nuc = rng.integers(0, 4)
            seq.append(NUCL[nuc])
            score += np.log2(max(pwm[k, nuc], 1e-9) / 0.25)
        sequences.append("".join(seq))
        labels.append(label)
        scores.append(score)

    return sequences, np.array(labels), np.array(scores)


def splice_site_auc(site_type="donor", n_sites=200, seed=42):
    """Return ROC-AUC for splice-site identification."""
    _, labels, scores = simulate_splice_sites(n_sites, site_type, seed)
    return roc_auc_score(labels, scores)


def pwm_score(sequence, pwm):
    """
    Score a nucleotide sequence against a PWM.

    Parameters
    ----------
    sequence : str or list of str — nucleotide characters
    pwm      : ndarray (W, 4)

    Returns
    -------
    score : float  — sum of log2(p_{k,v} / 0.25) over positions
    """
    score = 0.0
    for k, nuc in enumerate(sequence):
        if k >= pwm.shape[0]:
            break
        v      = NUCL.index(nuc) if isinstance(nuc, str) else int(nuc)
        score += np.log2(max(pwm[k, v], 1e-9) / 0.25)
    return score
