"""
cpg_island.py
-------------
Task 1 — CpG Island Simulation and Detection.

Two-state HMM over ordered dinucleotides (16 symbols):
    State 0 : CpG+  (CpG-island region)
    State 1 : CpG-  (non-island region)

Transition matrix parameterised by:
    p  = P(island -> non-island)   controls mean island length  L+ = 1/p
    q  = P(non-island -> island)   controls mean spacing        L- = 1/q

Emission matrices estimated from human chromosome 22 annotation
(Gardiner-Garden & Frommer 1987).

References
----------
Durbin, R. et al. (1998). Biological Sequence Analysis. Cambridge UP.
Gardiner-Garden, M. & Frommer, M. (1987). CpG islands in vertebrate genomes.
    J. Mol. Biol., 196(2), 261-282.
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef

NUCL    = ["A", "C", "G", "T"]
DINUCS  = [a + b for a in NUCL for b in NUCL]          # 16 symbols
DINUC2I = {d: i for i, d in enumerate(DINUCS)}

# Ground-truth emission matrices (per-dinucleotide, 4x4 row=prev, col=next)
# Rows = previous nucleotide {A,C,G,T}, cols = current nucleotide {A,C,G,T}
B_PLUS_DEFAULT = np.array([
    [0.180, 0.274, 0.426, 0.120],   # prev=A
    [0.171, 0.368, 0.274, 0.188],   # prev=C  (elevated CG: C->G)
    [0.161, 0.339, 0.375, 0.125],   # prev=G
    [0.188, 0.291, 0.334, 0.187],   # prev=T
], dtype=float)

B_MINUS_DEFAULT = np.array([
    [0.300, 0.205, 0.285, 0.210],   # prev=A
    [0.322, 0.298, 0.075, 0.305],   # prev=C  (depressed CG)
    [0.248, 0.246, 0.298, 0.208],   # prev=G
    [0.177, 0.239, 0.292, 0.292],   # prev=T
], dtype=float)

# Normalise rows
B_PLUS_DEFAULT  /= B_PLUS_DEFAULT.sum(1, keepdims=True)
B_MINUS_DEFAULT /= B_MINUS_DEFAULT.sum(1, keepdims=True)


def simulate_cpg_sequence(length=10_000, p=0.005, q=0.0005,
                           b_plus=None, b_minus=None, seed=42):
    """
    Simulate a nucleotide sequence from the CpG two-state HMM.

    Parameters
    ----------
    length  : int   — sequence length in nucleotides
    p       : float — island -> non-island transition probability
    q       : float — non-island -> island transition probability
    b_plus  : ndarray (4,4) or None — CpG+ emission matrix (default above)
    b_minus : ndarray (4,4) or None — CpG- emission matrix (default above)
    seed    : int

    Returns
    -------
    sequence    : list of str  — nucleotide characters
    true_states : list of int  — 0=CpG+, 1=CpG-
    """
    rng     = np.random.default_rng(seed)
    b_plus  = (b_plus  if b_plus  is not None else B_PLUS_DEFAULT).copy()
    b_minus = (b_minus if b_minus is not None else B_MINUS_DEFAULT).copy()

    state   = rng.choice([0, 1], p=[0.05, 0.95])   # start mostly non-island
    prev    = rng.integers(0, 4)
    seq, states = [], []

    for _ in range(length):
        B   = b_plus if state == 0 else b_minus
        nxt = rng.choice(4, p=B[prev])
        seq.append(NUCL[nxt])
        states.append(state)
        r = rng.random()
        if state == 0:
            state = 1 if r < p else 0
        else:
            state = 0 if r < q else 1
        prev = nxt

    return seq, states


def cpg_detection_metrics(true_states, predicted_states):
    """
    Compute sensitivity, specificity, and MCC for CpG island detection.

    Parameters
    ----------
    true_states      : array-like of int (0=island, 1=non-island)
    predicted_states : array-like of int

    Returns
    -------
    dict with keys: sensitivity, specificity, mcc
    """
    true = np.asarray(true_states)
    pred = np.asarray(predicted_states)

    TP = np.sum((true == 0) & (pred == 0))
    TN = np.sum((true == 1) & (pred == 1))
    FP = np.sum((true == 1) & (pred == 0))
    FN = np.sum((true == 0) & (pred == 1))

    sn  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    sp  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    mcc = matthews_corrcoef(true == 0, pred == 0)

    return {"sensitivity": sn, "specificity": sp, "mcc": mcc,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def run_cpg_experiment(length=100_000, p=0.005, q=0.0005,
                        alpha0=1.0, seed=42):
    """
    Full end-to-end CpG island experiment:
    simulate -> train B-HMM -> decode -> evaluate.

    Returns
    -------
    metrics : dict
    """
    from ..hmm_core.bayesian_hmm import BayesianHMM
    from ..hmm_core.viterbi import viterbi_decode

    seq, true_states = simulate_cpg_sequence(length, p, q, seed=seed)
    obs = [NUCL.index(n) for n in seq]

    # Build flat 16-symbol emission from 4x4 dinuc matrices
    # (For simplicity we use per-nucleotide unigram emissions here;
    #  the full dinucleotide model requires a second-order HMM wrapper.)
    B_flat = np.vstack([
        B_PLUS_DEFAULT.mean(axis=0),
        B_MINUS_DEFAULT.mean(axis=0),
    ])  # shape (2, 4)

    A  = np.array([[1 - p, p], [q, 1 - q]])
    pi = np.array([0.05, 0.95])

    model = BayesianHMM(n_states=2, n_obs=4, alpha0=alpha0, seed=seed)
    # Initialise with ground-truth A for fair benchmark; train B only
    model.A  = A
    model.B  = B_flat
    model.pi = pi
    model.fit(obs, max_iter=100)

    pred_states, _ = viterbi_decode(obs, model.A, model.B, model.pi)
    metrics = cpg_detection_metrics(true_states, pred_states.tolist())
    return metrics
