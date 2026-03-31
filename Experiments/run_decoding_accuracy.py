"""
run_decoding_accuracy.py
------------------------
Experiment: Viterbi decoding accuracy vs SNR, sequence length, N states
across ML-HMM and B-HMM (article §4.2).

Usage
-----
python experiments/run_decoding_accuracy.py \
    --snr 5 15 25 \
    --output results/decoding_accuracy.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hmm_core.baum_welch   import BaumWelchHMM
from src.hmm_core.bayesian_hmm import BayesianHMM
from src.hmm_core.viterbi      import viterbi_decode


N_STATES_LIST = [2, 4, 8]
SEQ_LENS      = [100, 500, 2000]
N_REPLICATES  = 50


def snr_to_conc(snr_db):
    return 0.5 + (snr_db / 25.0) * 4.5


def generate_labeled_sequence(N, T, snr_db, seed):
    rng   = np.random.default_rng(seed)
    conc  = snr_to_conc(snr_db)
    A     = rng.dirichlet(np.ones(N) * 2.0, size=N)
    B     = rng.dirichlet(np.ones(N) * conc, size=N)
    pi    = rng.dirichlet(np.ones(N))

    state = rng.choice(N, p=pi)
    obs, true_states = [], []
    for _ in range(T):
        obs.append(rng.choice(N, p=B[state]))
        true_states.append(state)
        state = rng.choice(N, p=A[state])
    return obs, true_states, A, B, pi


def state_accuracy(true_states, pred_states, N):
    """Best-permutation accuracy (handles label switching)."""
    from itertools import permutations
    true = np.array(true_states)
    pred = np.array(pred_states)
    best = 0.0
    for perm in permutations(range(N)):
        mapped = np.array([perm[p] for p in pred])
        acc    = np.mean(mapped == true)
        if acc > best:
            best = acc
    return best


def main():
    parser = argparse.ArgumentParser(description="Decoding accuracy benchmark")
    parser.add_argument("--snr",    type=int, nargs="+", default=[5, 15, 25])
    parser.add_argument("--output", type=str, default="results/decoding_accuracy.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fieldnames = ["N", "T", "SNR_dB", "method",
                  "mean_accuracy", "std_accuracy",
                  "mean_gap_over_ml"]
    rows = []

    total = len(N_STATES_LIST) * len(SEQ_LENS) * len(args.snr)
    done  = 0

    for N in N_STATES_LIST:
        for T in SEQ_LENS:
            for snr in args.snr:
                ml_accs, b_accs = [], []

                for rep in range(N_REPLICATES):
                    seed = N * 100_000 + T * 1_000 + snr * 10 + rep
                    obs, true_st, _, _, _ = generate_labeled_sequence(N, T, snr, seed)

                    # ML-HMM
                    ml = BaumWelchHMM(n_states=N, n_obs=N, n_restarts=3, seed=seed)
                    ml.fit(obs)
                    ml_pred, _ = viterbi_decode(obs, ml.A, ml.B, ml.pi)
                    ml_accs.append(state_accuracy(true_st, ml_pred, N))

                    # B-HMM
                    bm = BayesianHMM(n_states=N, n_obs=N, alpha0=1.0, seed=seed)
                    bm.fit(obs)
                    b_pred, _ = viterbi_decode(obs, bm.A, bm.B, bm.pi)
                    b_accs.append(state_accuracy(true_st, b_pred, N))

                gap = np.mean(b_accs) - np.mean(ml_accs)

                for method, accs in [("ML-HMM", ml_accs), ("B-HMM", b_accs)]:
                    rows.append({
                        "N":             N,
                        "T":             T,
                        "SNR_dB":        snr,
                        "method":        method,
                        "mean_accuracy": round(np.mean(accs) * 100, 2),
                        "std_accuracy":  round(np.std(accs)  * 100, 2),
                        "mean_gap_over_ml": round(gap * 100, 2) if method == "B-HMM" else 0.0,
                    })

                done += 1
                print(f"[{done}/{total}] N={N}, T={T}, SNR={snr} dB  "
                      f"ML={np.mean(ml_accs)*100:.1f}%  "
                      f"B={np.mean(b_accs)*100:.1f}%")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
