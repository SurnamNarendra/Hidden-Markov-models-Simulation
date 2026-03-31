"""
run_convergence_benchmark.py
----------------------------
Experiment: Compare convergence speed of ML-EM vs Bayesian-EM across
state-space sizes, sequence lengths, and SNR levels (article Table 3).

Usage
-----
python experiments/run_convergence_benchmark.py \
    --n_states 2 4 8 \
    --seq_len 100 500 2000 \
    --n_replicates 50 \
    --output results/convergence.csv
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hmm_core.baum_welch  import BaumWelchHMM
from src.hmm_core.bayesian_hmm import BayesianHMM


SNR_LEVELS = [5, 15, 25]   # dB


def snr_to_emission_sharpness(snr_db):
    """
    Map SNR (dB) to emission peakedness.
    Higher SNR -> more peaked emission -> easier decoding.
    """
    return 0.5 + (snr_db / 25.0) * 4.5   # concentration parameter


def generate_sequence(N, T, snr_db, seed):
    """Generate a random HMM sequence with controllable SNR."""
    rng   = np.random.default_rng(seed)
    conc  = snr_to_emission_sharpness(snr_db)
    A     = rng.dirichlet(np.ones(N) * 2.0, size=N)
    B     = rng.dirichlet(np.ones(N) * conc, size=N)   # N emission symbols
    pi    = rng.dirichlet(np.ones(N))

    state = rng.choice(N, p=pi)
    obs   = []
    for _ in range(T):
        obs.append(rng.choice(N, p=B[state]))
        state = rng.choice(N, p=A[state])
    return obs, A, B, pi


def run_single(N, T, snr_db, seed):
    """Run ML-EM and B-EM on one simulated sequence; return iteration counts."""
    obs, _, _, _ = generate_sequence(N, T, snr_db, seed)

    # ML-EM
    ml = BaumWelchHMM(n_states=N, n_obs=N, n_restarts=1, seed=seed)
    ml.fit(obs, max_iter=500, tol=1e-6)
    ml_iters = len(ml.log_likelihoods_)

    # B-EM
    b = BayesianHMM(n_states=N, n_obs=N, alpha0=1.0, seed=seed)
    b.fit(obs, max_iter=500, tol=1e-6)
    b_iters = len(b.log_likelihoods_)

    return ml_iters, b_iters


def main():
    parser = argparse.ArgumentParser(description="HMM convergence benchmark")
    parser.add_argument("--n_states",    type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seq_len",     type=int, nargs="+", default=[100, 500, 2000])
    parser.add_argument("--n_replicates",type=int, default=50)
    parser.add_argument("--output",      type=str, default="results/convergence.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fieldnames = ["N", "T", "SNR_dB", "method",
                  "mean_iters", "std_iters", "min_iters", "max_iters"]

    rows = []
    total = len(args.n_states) * len(args.seq_len) * len(SNR_LEVELS)
    done  = 0

    for N in args.n_states:
        for T in args.seq_len:
            for snr in SNR_LEVELS:
                ml_iters_list, b_iters_list = [], []
                for rep in range(args.n_replicates):
                    seed = N * 10_000 + T * 100 + snr + rep
                    ml_it, b_it = run_single(N, T, snr, seed)
                    ml_iters_list.append(ml_it)
                    b_iters_list.append(b_it)

                for method, iters in [("ML-EM", ml_iters_list),
                                       ("B-EM",  b_iters_list)]:
                    rows.append({
                        "N":          N,
                        "T":          T,
                        "SNR_dB":     snr,
                        "method":     method,
                        "mean_iters": round(np.mean(iters), 2),
                        "std_iters":  round(np.std(iters),  2),
                        "min_iters":  int(np.min(iters)),
                        "max_iters":  int(np.max(iters)),
                    })

                done += 1
                print(f"[{done}/{total}] N={N}, T={T}, SNR={snr} dB  done")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
