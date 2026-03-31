"""
run_parameter_recovery.py
-------------------------
Experiment: Frobenius-norm transition matrix estimation error
vs sequence length and Dirichlet prior strength (article §4.3).

Verifies O(T^{-1/2}) asymptotic convergence for ML-EM and shows
that B-EM achieves lower error for short sequences.

Usage
-----
python experiments/run_parameter_recovery.py \
    --n_states 4 \
    --alpha0 0.1 1.0 10.0 \
    --output results/param_recovery.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hmm_core.baum_welch   import BaumWelchHMM
from src.hmm_core.bayesian_hmm import BayesianHMM


SEQ_LENS     = [50, 100, 200, 500, 1000, 2000, 5000]
N_REPLICATES = 50


def generate_sequence_with_params(N, T, seed):
    rng  = np.random.default_rng(seed)
    A    = rng.dirichlet(np.ones(N) * 2.0, size=N)
    B    = rng.dirichlet(np.ones(N) * 1.5, size=N)
    pi   = rng.dirichlet(np.ones(N))

    state = rng.choice(N, p=pi)
    obs   = []
    for _ in range(T):
        obs.append(rng.choice(N, p=B[state]))
        state = rng.choice(N, p=A[state])
    return obs, A, B, pi


def frob_error(A_true, A_hat):
    return float(np.linalg.norm(A_true - A_hat, "fro"))


def pi_l2_error(pi_true, pi_hat):
    return float(np.linalg.norm(pi_true - pi_hat))


def main():
    parser = argparse.ArgumentParser(description="Parameter recovery benchmark")
    parser.add_argument("--n_states", type=int,   default=4)
    parser.add_argument("--alpha0",   type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--output",   type=str,   default="results/param_recovery.csv")
    args = parser.parse_args()

    N = args.n_states
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fieldnames = ["N", "T", "method", "alpha0",
                  "mean_frob_A", "std_frob_A",
                  "mean_l2_pi",  "std_l2_pi"]
    rows = []

    total = len(SEQ_LENS) * (1 + len(args.alpha0))
    done  = 0

    for T in SEQ_LENS:
        # --- ML-EM (no prior) ---
        ml_frob, ml_l2 = [], []
        for rep in range(N_REPLICATES):
            seed = T * 10_000 + rep
            obs, A_true, _, pi_true = generate_sequence_with_params(N, T, seed)
            ml = BaumWelchHMM(n_states=N, n_obs=N, n_restarts=3, seed=seed)
            ml.fit(obs)
            ml_frob.append(frob_error(A_true, ml.A))
            ml_l2.append(pi_l2_error(pi_true, ml.pi))

        rows.append({
            "N": N, "T": T, "method": "ML-EM", "alpha0": "N/A",
            "mean_frob_A": round(np.mean(ml_frob), 5),
            "std_frob_A":  round(np.std(ml_frob),  5),
            "mean_l2_pi":  round(np.mean(ml_l2),   5),
            "std_l2_pi":   round(np.std(ml_l2),    5),
        })
        done += 1
        print(f"[{done}/{total}] ML-EM  T={T}  Frob={np.mean(ml_frob):.4f}")

        # --- B-EM across alpha0 values ---
        for a0 in args.alpha0:
            b_frob, b_l2 = [], []
            for rep in range(N_REPLICATES):
                seed = T * 10_000 + rep
                obs, A_true, _, pi_true = generate_sequence_with_params(N, T, seed)
                bm = BayesianHMM(n_states=N, n_obs=N, alpha0=a0, seed=seed)
                bm.fit(obs)
                b_frob.append(frob_error(A_true, bm.A))
                b_l2.append(pi_l2_error(pi_true, bm.pi))

            rows.append({
                "N": N, "T": T, "method": "B-EM", "alpha0": a0,
                "mean_frob_A": round(np.mean(b_frob), 5),
                "std_frob_A":  round(np.std(b_frob),  5),
                "mean_l2_pi":  round(np.mean(b_l2),   5),
                "std_l2_pi":   round(np.std(b_l2),    5),
            })
            done += 1
            print(f"[{done}/{total}] B-EM   T={T}  a0={a0}  Frob={np.mean(b_frob):.4f}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
