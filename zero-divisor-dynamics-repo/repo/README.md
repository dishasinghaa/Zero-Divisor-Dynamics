# Zero-Divisor Dynamics: Learning Geometry of Algebraic Singularities

**Core question:** if an algebraic construction forces a genuinely singular
point to exist inside a dynamical system, can a statistical procedure — with
no access to its coordinates, no labels, and no knowledge of the underlying
algebra — independently detect that same point from its observable
consequences alone?

**One-line result:** yes — but not with the tool we first reached for.
A direct **state-space loss-zeta exponent** (computed from local samples of
the vector field itself, no neural network involved) cleanly and robustly
detects the engineered singularity, survives realistic observation noise up
to ~5%, generalizes to a structurally different second system, and — most
importantly — **blindly rediscovers x\* from an unlabeled equilibrium search**,
with no hint given to the algorithm at any step. A neural-network Local
Learning Coefficient (LLC) approach, tried first because it's the more
fashionable SLT tool, spent an entire session fighting its own calibration
and never produced a result we could fully trust — see `exploratory/` for
why, and what we learned from it.

## Quick start

Everything in `final/` runs with Python + NumPy + SciPy only (no torch
needed for the pipeline that actually works). Run in order:

```bash
cd final
python 01_baseline_equilibria.py          # finds the 7 ordinary equilibria at Omega=0.620
python 02_find_saddle_node.py             # solves the augmented system for the saddle-node
python 03_construct_xstar.py              # refines x*, confirms the 7->6 collision
python 00_plot_bifurcation.py             # regenerates saddle_node_bifurcation.png
python 04_blind_equilibria_discovery.py   # BLIND test: finds x* with zero hints
python 05_state_zeta_noise_robustness.py  # confirms the signal survives real noise
python 06_generality_check_system2.py     # confirms it's not an artifact of one equation
```

## What's actually established

| Claim | Status | Evidence |
|---|---|---|
| A genuine non-hyperbolic point exists | **Supported** | det J ≈ 10⁻¹⁴, simple zero eigenvalue, verified by hand |
| It's a saddle-node collision | **Supported** | 7→6 equilibria, √(Ω*−Ω) scaling law |
| Direct Jacobian rank-loss detects it | **Supported** | σ_min ≈ 0.03 at x*, 1.1–4.0 at controls |
| Convergence is sub-exponential at x* | **Supported** | T½ ∝ r⁻¹·⁰⁰⁸ (theory: −1) |
| State-space zeta exponent detects it | **Supported** | λ≈0.75 at x*, λ≈1.0 at controls (theory exact) |
| Survives realistic noise | **Supported to ~5%** | clean separation 0–5% noise, breaks ~10% |
| Blindly rediscoverable, no hints | **Supported** | unsupervised outlier rule flags x* correctly (fixed after an initial z-score bug at n=2) |
| Generalizes beyond one system | **Supported, n=2 systems** | second, structurally different system shows same pattern |
| Neural-network LLC detects it | **Not established** | see `exploratory/` — architecture/leash calibration dominated the signal |
| Phenomenon derives from real octonion algebra | **Not claimed** | coefficients were engineered for phase-portrait richness; algebra is origin inspiration only |

## Repo structure

- **`final/`** — the pipeline that works, in run order. Pure NumPy/SciPy.
- **`exploratory/`** — the neural-network SGLD/LLC attempts, kept deliberately
  rather than deleted. Files are numbered by attempt; each one's docstring
  explains what setting was tried and why it didn't give a trustworthy
  result (R-hat failures, a bias artifact at high leash strength, a
  replication check that correctly caught a false-positive-looking signal).
  This is the "wrong instrument, here's why" record, not a loose end.

## Honest open questions

- Exact noise breaking point is bracketed (5%–10%) but not pinned down.
- Generality tested on 2 systems; a third, independently-styled system would
  strengthen the claim further.
- Whether a properly-calibrated neural LLC *would* eventually detect the
  same signal is still genuinely unknown — we stopped chasing it once the
  state-space approach gave a cleaner, faster, more robust answer, not
  because it was proven incapable.
