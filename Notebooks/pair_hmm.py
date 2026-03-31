"""
pair_hmm.py
-----------
Task 5 — Pair-HMM for pairwise sequence alignment and homology scoring.

Three-state architecture (article §3.3.5):
    M   : Match / mismatch — emits (x_i, y_j) from a 4x4 substitution matrix
    I_x : Insertion in x   — emits x_i under background distribution
    I_y : Insertion in y   — emits y_j under background distribution

The Forward algorithm over the 2D lattice gives the total probability
P(x, y | lambda), marginalising over all alignments.

Log-odds homology score:
    log[ P(x, y | lambda) / (P(x) * P(y)) ]

This probabilistic score outperforms Smith-Waterman bit scores in the
twilight zone (sequence identity < 50%), as demonstrated in Table 5 of
the article.

References
----------
Durbin, R. et al. (1998). Biological Sequence Analysis. Cambridge UP.
    Chapter 4: Pair hidden Markov models.
"""

import numpy as np

NUCL = ["A", "C", "G", "T"]

# Background nucleotide frequencies (uniform)
BG = np.array([0.25, 0.25, 0.25, 0.25])

# Match emission matrix: PAM120-analogue nucleotide substitution
# Diagonal (identity) probability elevated; off-diagonal = substitution
_MATCH_DIAG = 0.70
_MATCH_OFF  = (1.0 - _MATCH_DIAG) / 3.0
MATCH_EMIT = np.full((4, 4), _MATCH_OFF)
np.fill_diagonal(MATCH_EMIT, _MATCH_DIAG)   # shape (4, 4)

# Default Pair-HMM transition probabilities
DEFAULT_TRANSITIONS = {
    "a_mm":  0.90,    # M  -> M
    "a_mix": 0.05,    # M  -> I_x
    "a_miy": 0.05,    # M  -> I_y
    "a_im":  0.70,    # I  -> M
    "a_ii":  0.30,    # I  -> I (gap extension)
}


# ── Core Forward algorithm ───────────────────────────────────────────────────

def pair_hmm_forward(x, y, transitions=None, match_emit=None, log_scale=True):
    """
    Forward algorithm for a three-state Pair-HMM.

    Parameters
    ----------
    x, y        : array-like of int (0-3) — two nucleotide sequences
    transitions : dict of transition probabilities (or None for defaults)
    match_emit  : ndarray (4,4) or None — match emission matrix
    log_scale   : bool — if True work in log domain (recommended)

    Returns
    -------
    log_prob : float — log P(x, y | lambda)
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    lx, ly = len(x), len(y)

    tr  = transitions or DEFAULT_TRANSITIONS
    ME  = match_emit  if match_emit is not None else MATCH_EMIT

    # Log-scale parameters
    lMM  = np.log(tr["a_mm"])
    lMIx = np.log(tr["a_mix"])
    lMIy = np.log(tr["a_miy"])
    lIM  = np.log(tr["a_im"])
    lII  = np.log(tr["a_ii"])
    lME  = np.log(np.maximum(ME, 1e-300))       # (4,4)
    lBG  = np.log(BG)

    NEG_INF = -1e30

    # DP tables: (lx+1, ly+1)
    f_M  = np.full((lx + 1, ly + 1), NEG_INF)
    f_Ix = np.full((lx + 1, ly + 1), NEG_INF)
    f_Iy = np.full((lx + 1, ly + 1), NEG_INF)

    f_M[0, 0] = 0.0   # begin state

    for i in range(1, lx + 1):
        for j in range(1, ly + 1):
            xi, yj = x[i - 1], y[j - 1]

            # Match state
            prev_M  = f_M[i - 1, j - 1]
            prev_Ix = f_Ix[i - 1, j - 1]
            prev_Iy = f_Iy[i - 1, j - 1]
            best    = _log_sum3(prev_M + lMM, prev_Ix + lIM, prev_Iy + lIM)
            f_M[i, j] = best + lME[xi, yj]

            # Insert in x (gap in y)
            if j >= 1:
                bx = _log_sum2(f_M[i - 1, j] + lMIx, f_Ix[i - 1, j] + lII)
                f_Ix[i, j] = bx + lBG[xi]

            # Insert in y (gap in x)
            if i >= 1:
                by = _log_sum2(f_M[i, j - 1] + lMIy, f_Iy[i, j - 1] + lII)
                f_Iy[i, j] = by + lBG[yj]

    log_prob = _log_sum3(f_M[lx, ly], f_Ix[lx, ly], f_Iy[lx, ly])
    return float(log_prob)


def log_odds_score(x, y, transitions=None, match_emit=None):
    """
    Compute Pair-HMM log-odds homology score:
        log[ P(x,y|lambda) / (P(x)*P(y)) ]

    Parameters
    ----------
    x, y : array-like of int

    Returns
    -------
    score : float
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    log_pxy = pair_hmm_forward(x, y, transitions, match_emit)
    log_px  = np.sum(np.log(BG[x]))
    log_py  = np.sum(np.log(BG[y]))
    return log_pxy - log_px - log_py


# ── Simulation of sequence pairs ─────────────────────────────────────────────

def simulate_sequence_pairs(n_pairs=200, seq_len=60,
                             identity_pct=70, seed=42):
    """
    Simulate homologous (label=1) and non-homologous (label=0) sequence pairs.

    Parameters
    ----------
    n_pairs      : int   — total pairs (half homologous, half not)
    seq_len      : int   — reference sequence length
    identity_pct : int   — percent identity for homologous pairs
    seed         : int

    Returns
    -------
    pairs  : list of (x, y) tuples — integer-encoded sequences
    labels : ndarray (n_pairs,)
    """
    rng = np.random.default_rng(seed)
    mut_rate = 1.0 - identity_pct / 100.0
    pairs, labels = [], []

    for idx in range(n_pairs):
        ref = rng.integers(0, 4, seq_len)
        if idx < n_pairs // 2:
            # Homologous: mutate reference
            seq_y = ref.copy()
            for k in range(seq_len):
                if rng.random() < mut_rate:
                    seq_y[k] = rng.integers(0, 4)
            labels.append(1)
        else:
            # Non-homologous: independent random sequence
            seq_y = rng.integers(0, 4, seq_len)
            labels.append(0)
        pairs.append((ref, seq_y))

    return pairs, np.array(labels)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _log_sum2(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -1e30 and b == -1e30:
        return -1e30
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _log_sum3(a, b, c):
    return _log_sum2(_log_sum2(a, b), c)
