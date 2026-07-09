"""DDNM sampler: diffusion-native reconstruction on the range-null subspace.

Why this exists (and why NullSpaceFlowSampler alone is insufficient for a
pretrained diffusion prior): a diffusion model's noise prediction eps(x_t, t) is
only meaningful when x_t actually lies on the diffusion trajectory at noise level
t, i.e. x_t is a genuinely noised image. A naive null-space ODE feeds the model
states of the form A_dag y + P_N z, which are NOT noised samples at the claimed
level, so the model's prediction is garbage and the flow diverges (verified
empirically: a 23 dB collapse, identical at 1 and 20 steps, off-manifold from the
first evaluation).

DDNM (Wang, Yu, Zhang, "Zero-Shot Image Restoration Using Denoising Diffusion
Null-Space Model", ICLR 2023) resolves this by keeping every intermediate state
on the diffusion manifold. Each step: denoise to a clean estimate, project it
through the range-null decomposition to pin the measured component, then re-noise
to the next level. The clean-image inversion and deterministic step follow DDIM
(Song, Meng, Ermon, ICLR 2021); the schedule follows DDPM (Ho, Jain, Abbeel,
NeurIPS 2020). See Derivations sections 8 and 9.

Guidance schedule (this engine's addition beyond vanilla DDNM): at high noise the
denoised estimate x0 is unreliable, and its null-space content is the term that
resonates at aggressive super-resolution (the prior amplifies its own high-
frequency output, which survives projection because it lives in the null space,
and compounds into a grid). Weighting the null-space contribution by w(t) < 1 at
high noise reduces the feedback gain below one so it cannot compound, while w(t)
ramps to 1 as noise falls so the final reconstruction keeps the full generative
contribution. w(t) == 1 for all t recovers vanilla DDNM exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..decomposition.range_null import RangeNullDecomposition
from ..priors.schedules import DiscreteLinearSchedule


class GuidanceSchedule:
  """Per-step weight on the null-space (generative) contribution.

  The weight ramps from guidance_min at the noisiest step to 1.0 at the cleanest,
  as a function of the signal level sqrt(alpha_bar_t). At guidance_min == 1.0 the
  weight is 1.0 everywhere and the sampler is exactly vanilla DDNM.
  """

  def __init__(self, guidance_min: float = 1.0, power: float = 1.0):
    if not 0.0 <= guidance_min <= 1.0:
      raise ValueError("guidance_min must be in [0, 1]")
    self.guidance_min = guidance_min
    self.power = power

  def weight(self, abar_t: float) -> float:
    signal_level = math.sqrt(max(abar_t, 0.0))
    return float(self.guidance_min + (1.0 - self.guidance_min) * (signal_level**self.power))


class DDNMSampler:
  """Range-null DDNM sampler for a pretrained eps-prediction diffusion prior.

  eps_fn(x_t, step) -> predicted noise, where step is an integer diffusion step in
  [0, T-1] and x_t is in the model's native space. The sampler handles the
  pipeline/model normalization internally (default [0,1] <-> [-1,1]).
  """

  def __init__(
    self,
    eps_fn: Callable[[torch.Tensor, int], torch.Tensor],
    decomposition: RangeNullDecomposition,
    schedule: DiscreteLinearSchedule,
    num_steps: int = 100,
    num_train_steps: int = 1000,
    prior_scale: float = 2.0,
    prior_shift: float = -1.0,
    stochastic: bool = False,
    check_consistency: bool = False,
    consistency_rtol: float = 1e-3,
    guidance: GuidanceSchedule | None = None,
  ):
    self.eps_fn = eps_fn
    self.dec = decomposition
    self.schedule = schedule
    self.num_steps = num_steps
    self.T = num_train_steps
    self.scale = prior_scale
    self.shift = prior_shift
    self.stochastic = stochastic
    self.check_consistency = check_consistency
    self.consistency_rtol = consistency_rtol
    # Default guidance is a no-op (weight 1 everywhere) so the sampler is
    # vanilla DDNM unless a damping schedule is explicitly supplied.
    self.guidance = guidance or GuidanceSchedule(guidance_min=1.0)

  def _to_model(self, x: torch.Tensor) -> torch.Tensor:
    return self.scale * x + self.shift

  def _to_pipeline(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.shift) / self.scale

  def _timesteps(self) -> list[int]:
    # Evenly spaced integer diffusion steps, high noise to low, inclusive of 0.
    step = self.T // self.num_steps
    ts = list(range(self.T - 1, -1, -step))
    if ts[-1] != 0:
      ts.append(0)
    return ts

  def _project(self, x0_pipeline: torch.Tensor, weight: float) -> torch.Tensor:
    # A_dag y + w * P_N x0. At w == 1 this equals dec.reconstruct(x0) exactly
    # (A_dag y + P_N x0); at w < 1 the generative null-space contribution is
    # damped, which is what suppresses high-noise resonance.
    if weight == 1.0:
      return self.dec.reconstruct(x0_pipeline)
    return self.dec.pinv_reconstruction + weight * self.dec.project_null(x0_pipeline)

  @torch.no_grad()
  def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
    # Returns the reconstruction in pipeline space, measurement-consistent.
    dec = self.dec
    shape = dec.image_shape
    device = dec.y.device

    x_t = torch.randn(shape, device=device, generator=generator)

    ts = self._timesteps()
    for i, t in enumerate(ts):
      abar_t = float(self.schedule.alpha_bar_at_step(t))
      sqrt_abar_t = math.sqrt(abar_t)
      sqrt_1m_abar_t = math.sqrt(max(1.0 - abar_t, 0.0))

      # 1. DENOISE: predict the clean image x0 (model space) from x_t.
      eps = self.eps_fn(x_t, t)
      x0_model = (x_t - sqrt_1m_abar_t * eps) / max(sqrt_abar_t, 1e-8)

      # 2. PROJECT: range-null correction in pipeline space, with the guidance
      # weight damping the generative contribution at high noise. The final step
      # uses weight 1 (signal_level -> 1), so the reconstruction keeps the full
      # prior contribution and consistency is exact regardless of the schedule.
      weight = self.guidance.weight(abar_t)
      x0_pipeline = self._to_pipeline(x0_model)
      x0_pipeline = self._project(x0_pipeline, weight)
      if self.check_consistency:
        dec.assert_consistent(x0_pipeline, rtol=self.consistency_rtol)
      x0_model = self._to_model(x0_pipeline)

      if i == len(ts) - 1:
        return x0_pipeline

      # 3. RENOISE to the next (lower) level.
      t_next = ts[i + 1]
      abar_next = float(self.schedule.alpha_bar_at_step(t_next))
      sqrt_abar_next = math.sqrt(abar_next)
      sqrt_1m_abar_next = math.sqrt(max(1.0 - abar_next, 0.0))

      if self.stochastic:
        noise = torch.randn(shape, device=device, generator=generator)
      # DDIM-deterministic. After the range-null projection the predicted eps no
      # longer matches the corrected x0. Re-noising with the stale eps makes the
      # trajectory internally inconsistent (a collapse the oracle cannot expose,
      # since projection is a no-op on the true image). Recompute the eps
      # consistent with the corrected x0:
      #   eps_consistent = (x_t - sqrt(abar_t) x0) / sqrt(1 - abar_t)
      elif sqrt_1m_abar_t > 1e-8:  # noqa: SIM108
        noise = (x_t - sqrt_abar_t * x0_model) / sqrt_1m_abar_t
      else:
        noise = eps
      x_t = sqrt_abar_next * x0_model + sqrt_1m_abar_next * noise

    return self._to_pipeline(x_t)