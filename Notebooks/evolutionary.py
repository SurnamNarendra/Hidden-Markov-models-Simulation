"""
evolutionary.py
---------------
Task 4 — Evolutionary Sequence Simulation under Substitution Models.

Implements:
    JC69  — Jukes-Cantor (1969): equal rates for all substitutions
    K80   — Kimura two-parameter (1980): distinguishes transitions (alpha)
            from transversions (beta)

Substitution probability formulae (article Eqs. 12-13):

JC69:
    P(j|i,t) = 1/4 + 3/4 * exp(-4*mu*t/3)   if i == j
             = 1/4 - 1/4 * exp(-4*mu*t/3)   if i != j

K80:
    P(transition|t)   = 1/4 + 1/4*exp(-4*beta*t) - 1/2*exp(-2*(alpha+beta)*t)
    P(transversion|t) = 1/4 - 1/4*exp(-4*beta*t)

Distance correction (JC69):
    d_hat = -3/4 * log(1 - 4/3 * p)
    where p = observed proportion of differing sites

References
----------
Jukes, T.H. & Cantor, C.R. (1969). Evolution of protein molecules.
    In Mammalian Protein Metabolism, 21-132. Academic Press.
Kimura, M. (1980). A simple method for estimating evolutionary rates of
    base substitutions through comparative studies of nucleotide sequences.
    J. Mol. Evol., 16(2), 111-120.
Yang, Z. (1994). Maximum likelihood phylogenetic estimation from DNA sequences
    with variable rates over sites. J. Mol. Evol., 39(3), 306-314.
"""

import numpy as np

NUCL     = ["A", "C", "G", "T"]
PURINES  = {0, 2}       # A, G
PYRIMID  = {1, 3}       # C, T


# ── JC69 ─────────────────────────────────────────────────────────────────────

def jc69_prob_matrix(mu, t):
    """
    Full 4x4 substitution probability matrix under JC69.

    P(j|i,t) = diag term  if i==j, else off-diag term.
    """
    diag    = 0.25 + 0.75 * np.exp(-4 * mu * t / 3)
    offdiag = 0.25 - 0.25 * np.exp(-4 * mu * t / 3)
    P = np.full((4, 4), offdiag)
    np.fill_diagonal(P, diag)
    return P


def jc69_corrected_distance(p_diff):
    """
    JC69 distance correction for multiple hits.

    Parameters
    ----------
    p_diff : float — observed proportion of differing sites

    Returns
    -------
    d : float — corrected distance (substitutions per site)
        Returns np.inf if p_diff >= 0.75 (saturation).
    """
    if p_diff >= 0.75:
        return np.inf
    return -0.75 * np.log(1.0 - 4.0 * p_diff / 3.0)


def evolve_jc69(sequence, mu, t, rng=None):
    """
    Evolve a nucleotide sequence along a branch of length t under JC69.

    Parameters
    ----------
    sequence : array-like of int (0-3)
    mu       : float — substitution rate per site per unit time
    t        : float — branch length
    rng      : numpy Generator

    Returns
    -------
    descendant : ndarray of int
    n_subs     : int — number of substitutions
    """
    rng = rng or np.random.default_rng()
    P   = jc69_prob_matrix(mu, t)
    seq = np.asarray(sequence, dtype=int)
    out = np.array([rng.choice(4, p=P[s]) for s in seq])
    return out, int(np.sum(out != seq))


# ── K80 ──────────────────────────────────────────────────────────────────────

def k80_prob_matrix(alpha, beta, t):
    """
    Full 4x4 substitution probability matrix under K80.

    alpha : transition rate  (A<->G, C<->T)
    beta  : transversion rate (A<->C, A<->T, G<->C, G<->T)
    """
    p_same  = 0.25 + 0.25 * np.exp(-4 * beta * t) + 0.50 * np.exp(-2 * (alpha + beta) * t)
    p_trans = 0.25 + 0.25 * np.exp(-4 * beta * t) - 0.50 * np.exp(-2 * (alpha + beta) * t)
    p_transv = 0.25 - 0.25 * np.exp(-4 * beta * t)

    P = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                P[i, j] = p_same
            elif (i in PURINES) == (j in PURINES):
                P[i, j] = p_trans      # same biochemical class -> transition
            else:
                P[i, j] = p_transv     # different class -> transversion
    # Renormalise rows for numerical safety
    P /= P.sum(axis=1, keepdims=True)
    return P


def evolve_k80(sequence, alpha, beta, t, rng=None):
    """
    Evolve a nucleotide sequence under K80.

    Parameters
    ----------
    sequence : array-like of int
    alpha    : float — transition rate
    beta     : float — transversion rate
    t        : float — branch length
    rng      : numpy Generator

    Returns
    -------
    descendant : ndarray of int
    n_ts       : int — number of transitions
    n_tv       : int — number of transversions
    """
    rng = rng or np.random.default_rng()
    P   = k80_prob_matrix(alpha, beta, t)
    seq = np.asarray(sequence, dtype=int)
    out = np.array([rng.choice(4, p=P[s]) for s in seq])

    n_ts = int(sum(
        1 for a, b in zip(seq, out)
        if a != b and (a in PURINES) == (b in PURINES)
    ))
    n_tv = int(sum(
        1 for a, b in zip(seq, out)
        if a != b and (a in PURINES) != (b in PURINES)
    ))
    return out, n_ts, n_tv


# ── Phylogenetic tree simulation ─────────────────────────────────────────────

def simulate_bifurcating_tree(root_seq, branch_lengths, model="JC69",
                               mu=0.1, kappa=2.0, rng=None):
    """
    Simulate sequences at all leaves of a bifurcating tree.

    Parameters
    ----------
    root_seq       : array-like of int  — ancestral sequence
    branch_lengths : list of float — one per leaf (symmetric bifurcation)
    model          : 'JC69' or 'K80'
    mu             : float — substitution rate (JC69)
    kappa          : float — ts/tv ratio = alpha/beta (K80)
    rng            : numpy Generator

    Returns
    -------
    leaf_seqs : list of ndarray — one sequence per leaf
    """
    rng   = rng or np.random.default_rng()
    root  = np.asarray(root_seq, dtype=int)
    alpha = kappa * 0.05
    beta  = 0.05
    leaves = []
    for t in branch_lengths:
        if model == "JC69":
            leaf, _ = evolve_jc69(root, mu, t, rng)
        else:
            leaf, _, _ = evolve_k80(root, alpha, beta, t, rng)
        leaves.append(leaf)
    return leaves
