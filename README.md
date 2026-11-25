# A shorter route to Tao's almost-bounded-orbits theorem via a polylogarithmic Fourier input

**Author:** Nathan Humphrey

In 2019 Terence Tao proved that almost all Collatz orbits attain almost bounded values ([Tao, *Forum of Mathematics, Pi* 10 (2022) e12](https://arxiv.org/abs/1909.03562)). A key input to his proof is superpolynomial decay of the characteristic function of the Syracuse random variable (his Proposition 1.17).

This note shows that a much weaker input already suffices:

  Φ(n) ≪ n^(−1/4) (log n)^(−3/2−δ),

and proves that weaker bound, Φ(n) ≪ n^(−1/4) (log n)^(−2) (log log n)^3, using only Cases 1–2 of Tao's §7 induction. The route avoids Tao's Corollary 6.3 (the buffer in §6), Case 3 of the proof of Proposition 7.8, and Lemmas 7.9–7.10.

**This is a simplification of Tao's proof, not a new result about the Collatz map.** The ingredients are Tao's own; what changes is that each step is asked to prove less. The rate in Tao's Theorem 3.1 actually gets weaker.

## Status: an invitation to check

This note was produced with substantial use of AI systems (Anthropic's Claude models), working under the author's direction. AI agents developed the arguments, checked them numerically, and reviewed them adversarially against the text of Tao's paper; the text was drafted with heavy assistance by AI. **No independent human expert has checked the arguments yet.** The author is responsible for any errors.

Comments, corrections and counterexamples are very welcome. Please open an issue.

## Contents

| path | what it is |
|---|---|
| `paper/Tao_Shortened_Proof.pdf` | the note |
| `paper/` (LaTeX source) | source of the note |
| `checks/` | the numerical sanity checks of §9.2; see [`checks/README.md`](checks/README.md) |

All section, lemma and equation numbers of the form "Lemma 7.4" or "(7.51)" refer to Tao, arXiv:1909.03562v7.

## Running the checks

Python 3 with NumPy. From the repository root:

```
python checks/run_all.py
```

Each check runs in seconds on one core (the slowest takes under a minute; peak memory about 0.5 GB). These are floating-point sanity checks, not part of the proof.

## How to cite

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be added here once it is minted.

## License

- The paper (`paper/`) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The code (`checks/`) is licensed under the MIT License; see [`LICENSE`](LICENSE).
