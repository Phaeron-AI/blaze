# Citations & Licensing — Blaze / Keystone Engine

Every borrowed method used in this codebase, where it is implemented, and its
licensing status. This file is both academic attribution and legal record. See
`references.bib` for BibTeX.

> **Contribution boundary.** Keystone's novelty is the *composition*: a certified
> measurement-operator framework, the physics/consistency layer, and the target
> scientific-imaging application — layered on DDNM's range-null sampling with a
> borrowed pretrained prior. The methods below are borrowed and must be cited; the
> integration and the physics-consistency framework are the contribution.

---

## Methods implemented

| Method | Where used | Citation |
|---|---|---|
| **DDNM** (range-null diffusion sampling) | `samplers/ddnm.py` — the core reconstruction loop | Wang, Yu, Zhang. *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model.* ICLR 2023. |
| **Guided Diffusion** (architecture + prior weights) | `priors/pretrained_score.py`; `external/guided-diffusion` | Dhariwal, Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021. |
| **DDPM** (forward process, β/ᾱ schedule) | `priors/schedules.py` | Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. |
| **DDIM** (deterministic sampling, x0-inversion) | `samplers/ddnm.py` | Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. |
| **Score SDE / probability-flow ODE** | `priors/score_to_velocity.py`; Derivations §5 | Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. *Score-Based Generative Modeling through SDEs.* ICLR 2021. |
| **Flow Matching / Rectified Flow** | `priors/score_to_velocity.py`, velocity path | Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. Liu, Gong, Liu. *Rectified Flow.* 2023. |
| **Tikhonov regularization** (damped pseudoinverse) | `operators/pseudoinverse.py` | Standard; Tikhonov & Arsenin, *Solutions of Ill-Posed Problems*, 1977. |

---

## Licensing — ACTION REQUIRED before commercialization

The borrowed prior is the item with commercial-use consequences for an on-prem,
IP-driven product. **Verify and record the following before Phase-2
commercialization or before training/fine-tuning from the borrowed weights in a
commercial deliverable:**

1. **`guided-diffusion` code license** — check the `openai/guided-diffusion`
   repository LICENSE (believed MIT; confirm and paste the exact terms here).
   MIT permits commercial use with attribution, but confirm.

2. **Checkpoint (`256x256_diffusion_uncond.pt`) usage terms** — a model's *code*
   license and its *weights* license can differ. Confirm whether the pretrained
   weights permit commercial use, redistribution, and fine-tuning-and-shipping.
   This governs whether the borrowed prior can be part of a shipped on-prem
   product or must be replaced by a from-scratch / permissively-licensed prior.

3. **Downstream implication** — if the checkpoint's terms restrict commercial use,
   the borrowed prior is fine for *research validation* (the current phase) but the
   commercial product needs an owned or permissively-licensed prior. This is a
   reason the Phase-3 domain fine-tune (your own checkpoints) matters beyond
   quality: it is also the path to a prior you fully own.

> Status: **UNVERIFIED — confirm before shipping.** Record findings inline above.

---

## In the paper

- Cite all methods above.
- State the borrowed/novel boundary explicitly (see contribution note at top).
- Anticipate the reviewer question "what is novel beyond DDNM?" — the answer is the
  certified operator + physics-consistency framework and the scientific-imaging
  application, not the range-null sampling itself.