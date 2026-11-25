# Numerical checks

These scripts back the numerical statements in §9 of the note. They are floating-point sanity checks, **not part of the proof**. Each is self-contained (Python 3, NumPy), single-threaded, and runs in seconds.

Run them all with `python checks/run_all.py`, or one at a time with `python checks/<script>.py`.

| script | what it checks | where in the note | expected output |
|---|---|---|---|
| `exact_law.py` | exact law of X_n = Syrac(Z/3^n Z) for n ≤ 13 via the recursion of [Tao, Lemma 1.12] | Table 3, Table 4, §9.3 | Φ(2..13) = 0.378, 0.252, 0.177, 0.129, 0.096, 0.076, 0.061, 0.048, 0.038, 0.032, 0.026, 0.022; C_2 = 2.1429, C_12 = 6.8029, C_13 = 7.2725 |
| `phi_and_lemmaB.py` | Φ(n) and its monotonicity (Lemma 3.2); compatibility of the laws mod 3^(n−1) (Result 3.3); 2 is a primitive root mod 3^L; inequality (4) of Lemma B with the weaker constant 0.846/√(n−L), and its unweighted form for 1 ≤ L < n ≤ 10 | §3, Lemma B, §9.2 | Φ column matches Table 3; "compatibility ... ok"; "2 primitive root ok"; worst ratios ≤ 1; "C3 ok" |
| `lemmaB_pairing.py` | the pairing factorisation in the proof of Lemma B, both Cauchy–Schwarz steps, and inequality (4), for ten pairs (n, L) with n ≤ 13; also the mean of Π_ξ over units | Lemma B, §2.4, §9.2 | identity error ≈ 10^(−14); 0 violations in every row; meanQ = 2^(−J) |
| `lemmaC_bound.py` | the bound of Lemma C for L ≤ 12; brute-force check of Steps 1–3 of its proof (decoding, oddness, class-sum and per-slice inequalities) for L ≤ 6; the tail constant 4.449 | Lemma C, Remark 7.1, Table 4, §9.2 | every "ok True"; C_12 = 6.802899; "brute step1-3 failures: 0" |
| `lemmaC_steps.py` | every inequality in the slice + Minkowski proof of Lemma C, for L ≤ 12 | Lemma C | slice ratios ≤ 1; C_L matches `exact_law.py` |
| `lemmaA_numerics.py` | Monte Carlo K·P(S_∞ > K) (consistent with the critical 1/K tail); the moment formula E 2^(−qa) = 1/(2^(1+q) − 1); Σ_k √P(u = k) / n^(1/4) | Lemma A Step 2, Remark 5.1, §9.2 | K·P(S > K) roughly constant (≈ 1.9); the last ratio ≈ 1.78 for n = 200, 800, 3200 |

The Monte Carlo step in `lemmaA_numerics.py` uses a fixed seed, so its output is reproducible.
