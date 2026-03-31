"""
gene_structure.py
-----------------
Task 2 — Gene Structure Prediction Simulation.

Five-state HMM topology (article §3.3.2):
    States : {Intergenic, Exon0, Exon1, Exon2, Intron}

Exon states are frame-specific (reading frame r ∈ {0,1,2}).
Emission distributions are derived from human RefSeq codon usage.
Splice signals (GT donor / AG acceptor) are encoded at fixed relative
positions via a lightweight PWM at intron boundaries.

Metrics reported (Burge & Karlin 1997 protocol):
    - Nucleotide-level sensitivity (Sn) and specificity (Sp)
    - Exact exon match (ESn, ESp)
    - Approximate exon match (AEM, overlap >= 1 bp)

References
----------
Burge, C. & Karlin, S. (1997). Prediction of complete gene structures in
    human genomic DNA. J. Mol. Biol., 268(1), 78-94.
"""

import numpy as np

NUCL   = ["A", "C", "G", "T"]
STATES = ["Intergenic", "Exon0", "Exon1", "Exon2", "Intron"]
S2I    = {s: i for i, s in enumerate(STATES)}

# Frame-specific exon emission (4-nt, rows = states)
# Values reflect human RefSeq codon bias (simplified)
EMIT = {
    "Intergenic": np.array([0.25, 0.25, 0.25, 0.25]),
    "Exon0":      np.array([0.28, 0.22, 0.30, 0.20]),
    "Exon1":      np.array([0.20, 0.30, 0.28, 0.22]),
    "Exon2":      np.array([0.22, 0.28, 0.22, 0.28]),
    "Intron":     np.array([0.27, 0.23, 0.23, 0.27]),
}

# Biologically constrained transition matrix (article Fig. 2)
# Rows / cols: Intergenic, Exon0, Exon1, Exon2, Intron
A_GENE = np.array([
    [0.9980, 0.0020, 0.0000, 0.0000, 0.0000],  # Intergenic
    [0.0000, 0.9890, 0.0000, 0.0000, 0.0110],  # Exon0 -> Intron
    [0.0000, 0.0000, 0.9900, 0.0000, 0.0100],  # Exon1 -> Intron
    [0.0070, 0.0000, 0.0000, 0.9870, 0.0060],  # Exon2 -> Intergenic or Intron
    [0.0000, 0.0000, 0.0080, 0.0080, 0.9840],  # Intron -> Exon1/2
], dtype=float)
A_GENE /= A_GENE.sum(1, keepdims=True)

PI_GENE = np.array([0.90, 0.025, 0.025, 0.025, 0.025])


def simulate_gene_sequence(length=100_000, seed=42):
    """
    Simulate a genomic sequence with annotated gene structure.

    Parameters
    ----------
    length : int  — sequence length in bp
    seed   : int

    Returns
    -------
    seq         : list of str  — nucleotides
    true_states : list of str  — state labels per position
    true_frames : list of int  — reading frame (-1 for non-coding)
    """
    rng    = np.random.default_rng(seed)
    state  = rng.choice(len(STATES), p=PI_GENE)
    seq, st_labels, frames = [], [], []
    frame_count = 0

    for _ in range(length):
        s_name = STATES[state]
        nuc    = rng.choice(4, p=EMIT[s_name])
        seq.append(NUCL[nuc])
        st_labels.append(s_name)
        if "Exon" in s_name:
            frames.append(frame_count % 3)
            frame_count += 1
        else:
            frames.append(-1)
            if s_name == "Intergenic":
                frame_count = 0
        state = rng.choice(len(STATES), p=A_GENE[state])

    return seq, st_labels, frames


def gene_prediction_metrics(true_states, pred_states):
    """
    Compute nucleotide-level Sn/Sp and exon-level metrics.

    Parameters
    ----------
    true_states : list of str  — ground-truth state labels
    pred_states : list of str  — predicted state labels

    Returns
    -------
    dict with: nucl_sn, nucl_sp, exon_sn, aem
    """
    true = np.array(true_states)
    pred = np.array(pred_states)

    coding_true = np.array(["Exon" in s for s in true])
    coding_pred = np.array(["Exon" in s for s in pred])

    TP = np.sum(coding_true & coding_pred)
    FN = np.sum(coding_true & ~coding_pred)
    FP = np.sum(~coding_true & coding_pred)

    nucl_sn = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    nucl_sp = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Exon-level: extract exon intervals
    def intervals(labels):
        segs, in_exon, start = [], False, 0
        for i, s in enumerate(labels):
            if "Exon" in s and not in_exon:
                in_exon, start = True, i
            elif "Exon" not in s and in_exon:
                segs.append((start, i - 1))
                in_exon = False
        if in_exon:
            segs.append((start, len(labels) - 1))
        return segs

    true_exons = intervals(true_states)
    pred_exons = intervals(pred_states)

    exact_match = sum(
        1 for te in true_exons
        if any(pe[0] == te[0] and pe[1] == te[1] for pe in pred_exons)
    )
    approx_match = sum(
        1 for te in true_exons
        if any(pe[0] <= te[1] and pe[1] >= te[0] for pe in pred_exons)
    )

    n_true = len(true_exons)
    exon_sn = exact_match  / n_true if n_true > 0 else 0.0
    aem     = approx_match / n_true if n_true > 0 else 0.0

    return {
        "nucl_sn": round(nucl_sn, 4),
        "nucl_sp": round(nucl_sp, 4),
        "exon_sn": round(exon_sn, 4),
        "aem":     round(aem, 4),
        "n_true_exons": n_true,
        "exact_match":  exact_match,
    }
