# Engineering Standards — Blaze / Keystone Engine

Standards for research code in this repository. The goal is **reproducible,
scalable, defensible** code: every published number regenerable from a committed
config, every borrowed method cited, every non-obvious formula linked to its
derivation. Research velocity and code quality are not in tension here — the
discipline below is what has repeatedly caught subtle, expensive bugs early.

> Guiding principle: **reproducibility over elegance, correctness over speed,
> enforcement over discipline.** Style and types are checked mechanically so they
> are never a review conversation.

---

## 1. Repository structure

The repo separates three concerns that must not bleed into each other: the
**library** (reusable, typed, tested), **scripts** (entry points, allowed to be
thin), and **artifacts** (data, weights, outputs — never committed).

```
blaze/
├── pyproject.toml            # single source of build + tool config (ruff, mypy, pytest)
├── README.md
├── STANDARDS.md              # this file
├── CITATIONS.md              # every borrowed method + license, human-readable
├── references.bib            # BibTeX for the paper
├── .pre-commit-config.yaml   # ruff + mypy run before every commit
├── .gitignore                # excludes artifacts (see §1.3)
│
├── engine/
│   ├── src/                  # THE LIBRARY. Clean, typed, tested. Importable package.
│   │   ├── __init__.py
│   │   ├── protocols.py      # core interfaces: ForwardOperator, DiffusionPrior,
│   │   │                     #   VelocityPrior, Sampler  (§4)
│   │   ├── operators/        # measurement operators A, adjoints, pseudoinverse
│   │   ├── decomposition/    # range-null (Paradigm A)
│   │   ├── priors/           # velocity/score priors, schedules, normalization
│   │   ├── samplers/         # null-space flow, DDNM
│   │   ├── pipeline/         # composition layer (reconstructors) — was reconstruct.py
│   │   ├── training/         # Trainer, checkpointing, losses  (Phase 3)
│   │   ├── eval/             # metrics, run logging, synthetic test images
│   │   └── utils/            # seeding, io, invariants — cross-cutting helpers
│   │
│   ├── tests/                # pytest. Unit (fast) + integration (needs weights).
│   │   ├── conftest.py       # shared fixtures (operators, schedules, tmp dirs)
│   │   ├── unit/
│   │   └── integration/
│   │
│   └── scripts/              # thin CLI entry points. Import from src, hold no logic.
│
├── configs/                  # Hydra/YAML run configs — the reproducibility root (§2)
│   ├── operator/
│   ├── prior/
│   ├── sampler/
│   └── experiment/
│
├── external/                 # third-party clones (guided-diffusion). GITIGNORED. (§1.3)
│
└── artifacts/                # ALL non-code outputs. GITIGNORED. (§1.3)
    ├── models/               # checkpoints (borrowed + your own trained)
    ├── samples/              # test images, degraded inputs
    ├── runs/                 # JSONL logs, experiment outputs, reconstructions
    └── cache/                # anything regenerable
```

### 1.1 Library vs scripts — the hard rule

`engine/src/` is the library: fully typed, tested, no `sys.path` hacks, no CLI
parsing, no hardcoded paths. `engine/scripts/` holds entry points that parse args
and call the library. **Scripts import from `src`; `src` never imports from
`scripts`.** If a script grows logic worth testing, that logic moves into `src`.

### 1.2 Where images, samples, and weights go

Everything that is **not source code** lives under `artifacts/` and is gitignored:

- **Checkpoints** (`artifacts/models/`) — borrowed (`256x256_diffusion_uncond.pt`)
  and your own trained checkpoints. Never committed (size + license). Recreation
  is documented in `README.md`; versioning via DVC when training begins.
- **Test/sample images** (`artifacts/samples/`) — degraded inputs, natural photos,
  synthetic images. *Generated* synthetic images are reproducible from
  `engine/src/eval/` code, so they need not be committed; downloaded benchmark
  sets (Set5, DIV2K) are referenced by a download script, not committed.
- **Run outputs** (`artifacts/runs/`) — JSONL logs, reconstructions, metrics.

Rationale: the repo stays small and cloneable; nothing that is large,
regenerable, or license-encumbered enters version history. A fresh checkout +
`pip install -e .` + the documented download steps reproduces everything.

### 1.3 `.gitignore` essentials

```gitignore
# artifacts — never commit data, weights, or outputs
artifacts/
external/
*.pt
*.ckpt
*.png
*.jpg
# python
__pycache__/
*.egg-info/
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

External third-party clones are gitignored but their recreation is recorded in
`README.md` (e.g. the editable `guided-diffusion` install), so the dependency is
reproducible without vendoring someone else's source into your history.

---

## 2. Reproducibility (the top priority)

Every result must be regenerable from committed inputs. Non-negotiable:

- **Configs, not literals.** Hyperparameters live in `configs/*.yaml` (Hydra),
  not hardcoded in code. A run is defined by its config file.
- **Seeded.** Every entry point calls `utils.set_seed(seed)` (seeds `torch`,
  `numpy`, python `random`; sets `cudnn.deterministic` where required). The seed
  is part of the config and is logged.
- **Logged provenance.** Every run records: full config, git commit hash, seed,
  and metrics (via the JSONL `RunLogger`, graduating to MLflow once there are many
  runs to compare). A number with no recoverable config is treated as broken.
- **Pinned environment.** Exact dependency versions in `pyproject.toml`; the
  `guided-diffusion` editable install and checkpoint download documented in
  `README.md`.

---

## 3. Formatting & tooling (enforced, not discussed)

Style is mechanical. Config lives in `pyproject.toml`; hooks run on commit.

```toml
[tool.ruff]
indent-width = 2
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM"]  # errors, pyflakes, isort, naming, etc.

[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["engine/tests"]
markers = ["integration: requires model weights", "slow: long-running"]
```

`.pre-commit-config.yaml` runs `ruff format`, `ruff check`, and `mypy` before every
commit. Nothing lands unformatted, unlinted, or untyped. **Indentation is 2 spaces,
enforced by `ruff`** — no file drifts to 4.

---

## 4. Typed interfaces & contracts

Core abstractions are explicit Protocols/ABCs in `engine/src/protocols.py`. This is
not ceremony: two of this project's most expensive bugs were interface mismatches
(an integer diffusion step passed where flow-time in [0,1] was expected; a stale
noise term reused after a projection). Typed contracts turn those into
construction-time errors under `mypy` rather than runtime collapses.

Minimum contracts:

```python
class ForwardOperator(Protocol):
  def A(self, x: Tensor) -> Tensor: ...          # measurement
  def AT(self, y: Tensor) -> Tensor: ...         # exact adjoint: <Ax,y> = <x,ATy>

class DiffusionPrior(Protocol):
  def eps_at_step(self, x: Tensor, step: int) -> Tensor: ...   # INTEGER step 0..T-1

class VelocityPrior(Protocol):
  def velocity(self, x: Tensor, t: float) -> Tensor: ...       # FLOW time t in [0,1]

class Sampler(Protocol):
  def sample(self, *, generator: torch.Generator | None = None) -> Tensor: ...
```

The `int` step vs `float` flow-time distinction is encoded in the types precisely
because conflating them cost a −23 dB debugging session. A sampler that consumes a
`DiffusionPrior` cannot be wired to a `VelocityPrior` without a type error.

---

## 5. Numerical code carries its derivation (math comments)

Every non-obvious numerical line gets a comment stating **(a) the symbolic
equation, (b) the source, (c) any non-obvious convention**, and — where one exists
— a pointer to the derivations document section. A reviewer must never have to
reverse-engineer a formula.

**Standard pattern:**

```python
# DDIM x0-prediction: invert  x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps  for x0.
# Exact given the true eps. (Ho et al. 2020 Eq.15; Song et al. 2021 DDIM Eq.12.)
# Derivations §8. abar_t = alpha_bar at integer step t.
x0 = (x_t - sqrt_1m_abar_t * eps) / max(sqrt_abar_t, 1e-8)
```

**Where this is mandatory in this codebase:**

| File | Formula needing a math comment | Reference |
|---|---|---|
| `samplers/ddnm.py` | x0-inversion, range-null projection, **ε-consistency renoise** | DDNM (Wang 2023); Derivations §8–9 |
| `priors/score_to_velocity.py` | score = −ε/√(1−ᾱ); flow-time flip τ=1−t | Song 2021 SDE; Derivations §6 |
| `priors/schedules.py` | ᾱ = ∏(1−β) discrete product; linear β | Ho 2020 DDPM; Derivations §3 |
| `operators/pseudoinverse.py` | Tikhonov filter σ/(σ²+ε); ε=1e-6 rationale | Derivations §3 |
| `decomposition/range_null.py` | A·P_N = 0 consistency identity; P_R, P_N | Derivations §4 |
| `priors/normalized.py` | velocity rescaling v_pipeline = v_model/scale | Derivations §10 |

The **ε-consistency renoise** in `ddnm.py` gets the fullest comment — it must
explain *why* the stale ε fails, so no one "simplifies" it back to `noise = eps`
and silently reintroduces the −14 dB collapse.

---

## 6. Docstrings

NumPy-style, uniform across the library. Every public function documents purpose,
args, returns, and — for tensor functions — **shapes**. Math functions state the
equation they implement. Module headers cite the method(s) they implement (§7).

```python
def eps_at_step(self, x: Tensor, step: int) -> Tensor:
  """Predicted noise at an integer diffusion step.

  Parameters
  ----------
  x : Tensor, shape (N, C, H, W)
      Noised image in the model's native space ([-1, 1]).
  step : int
      Integer diffusion step in [0, T-1]. NOT flow-time — see `eps` for that.

  Returns
  -------
  Tensor, shape (N, C, H, W)
      Predicted noise (learn_sigma variance channels already sliced off).
  """
```

---

## 7. Citations & licensing (required)

This is an IP-driven lab intending to publish; attribution is both academic
honesty and legal hygiene. Two obligations:

**7.1 In code.** Each module implementing a published method cites it in the module
docstring. See `CITATIONS.md` / `references.bib`. Methods currently in use:

- **DDNM** — Wang, Yu, Zhang, *Zero-Shot Image Restoration Using Denoising
  Diffusion Null-Space Model*, ICLR 2023. (core sampler)
- **Guided Diffusion + prior weights** — Dhariwal, Nichol, *Diffusion Models Beat
  GANs on Image Synthesis*, NeurIPS 2021. (borrowed prior)
- **DDPM** — Ho, Jain, Abbeel, NeurIPS 2020. (schedule, forward process)
- **DDIM** — Song, Meng, Ermon, ICLR 2021. (deterministic sampling)
- **Score SDE / PF-ODE** — Song et al., ICLR 2021. (probability-flow ODE)
- **Flow Matching** — Lipman et al., ICLR 2023; Rectified Flow, Liu et al. 2023.
  (velocity path)

**7.2 Licensing — action required before commercialization.** The
`guided-diffusion` code and its checkpoint carry a license that governs commercial
and on-prem use, including fine-tuning from those weights. Verify the license terms
and record them in `CITATIONS.md` **before** shipping or training from the borrowed
weights in a commercial product. This is a real obligation for an on-prem
deliverable, not academic courtesy.

**7.3 In the paper.** State precisely what is borrowed vs. novel. The contribution
is the composition — certified operator framework, physics-consistency layer,
domain application — over DDNM's range-null sampling with a borrowed prior. Being
scrupulous here strengthens the IP position; a reviewer will ask "what is novel
beyond DDNM?" and the paper must answer crisply.

---

## 8. Testing

`pytest`, split into fast unit tests (run in CI on every commit) and integration
tests (need weights; skipped when absent). Two principles specific to this work:

- **Test against ground truth, not "it runs."** Assert values against hand-computed
  or closed-form references (the metric certification, the operator adjoint test,
  the oracle reconstruction). A test that only checks for absence of exceptions is
  insufficient.
- **A test input that satisfies the constraint exactly cannot test the
  constraint-enforcement path.** The pure oracle (which satisfies measurement
  consistency exactly) could not catch the ε-consistency bug because the projection
  was a no-op on it. The **semi-oracle** (a deliberately-shifted truth, so the
  projection genuinely acts) is required to exercise enforcement. When testing a
  correction/projection step, always include an input that *violates* the target so
  the correction has to do work.

---

## 9. Commit hygiene

- Small, focused commits with descriptive messages.
- Green pre-commit (format + lint + types) before every commit.
- Certification tests pass before merging to `dev`.
- Structural refactors (interfaces, package moves) land as their own commits,
  separate from feature work.
