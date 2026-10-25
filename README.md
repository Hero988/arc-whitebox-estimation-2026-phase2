# ARC White-Box Estimation Challenge 2026 — Phase 2 estimator (submission #333755)

Cumulant-history moment closure under a metered compute budget. Author: Suliman Tadros (AIcrowd: `suliman_tadros`).

This repository contains the exact source of AIcrowd submission **#333755**, the submission that my Phase 2
algorithmic-contribution write-up corresponds to.

| | |
|---|---|
| Official score (adjusted) | 1.919444765208421e-9 |
| Final-layer MSE (raw) | 1.5706e-8 |
| FLOP fraction of 2^41 | 0.122 |
| Networks graded | 100 / 100 |
| Residual wall time (first run) | median 0.311 s, max 0.335 s |
| `estimator.py` SHA-256 | `b1f91d17c30e70486c014559bdc61595b7ca392426178c1bbff525e819473ceb` |

`estimator.py` and `manifest.json` are byte-for-byte the files of the uploaded package (`whestbench` 0.16.1,
`flopscope` 0.12.1 at packaging time). All fitted constants are embedded in `estimator.py`; there are no data files.

## What it does

Sampling-free propagation of a moment state through all 16 layers of a width-1024 ReLU MLP: the mean, a dense covariance
on near-kink units, a compressed history of third cumulants in canonical-polyadic form (rank-one star factors born at
near-kink units, transported through the weights, weighted-SVD compression of aged blocks), and an isotropic fourth-order
term. The ReLU step uses Gaussian-plus-Edgeworth closures; small learned heads correct what the closure misses. Large
products use a metered Strassen-Winograd recursion, and the code is written to keep client-server round trips low.
See the write-up for the method, measurements and negative results:
[`arc-whitebox-phase2-writeup-333755-Tadros.pdf`](arc-whitebox-phase2-writeup-333755-Tadros.pdf).

## Running it

Install the official starter-kit tooling (see https://github.com/AIcrowd/whest-starterkit), then:

```bash
whest validate --estimator estimator.py
whest run --estimator estimator.py --runner server --n-mlps 10
```

Note: the organisers later moved the evaluator to flopscope 0.12.3, which changed how some operations are billed.

## Credits and licence

Released under the MIT licence (see `LICENSE`). Development began from the AIcrowd starter kit and ARC's public work on
cumulant propagation (https://github.com/alignment-research-center/mlp_cumulant_propagation, MIT); fitted constants were
derived from the public ARC WhestBench dataset (CC BY 4.0). Notices are in `THIRD_PARTY_NOTICES.md`.

Most of the implementation and experiments were carried out by AI coding agents (Anthropic's Claude and OpenAI's Codex)
working under my direction.
