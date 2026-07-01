"""DDNM sampler — diffusion-native reconstruction on the range-null subspace.

Why this exists (and why NullSpaceFlowSampler alone is insufficient for a
pretrained diffusion prior): a diffusion model's noise prediction eps(x_t, t) is
only meaningful when x_t actually lies on the diffusion trajectory at noise level
t. A naive null-space ODE feeds the model states A_dag y + P_N z, which are NOT
noised samples at the claimed level, so the prediction is garbage and the flow
diverges. DDNM (Wang et al. 2023) keeps every intermediate state ON the manifold:

  1. DENOISE:  x0_hat = (x_t - sqrt(1 - abar_t) * eps(x_t, t)) / sqrt(abar_t)
  2. PROJECT:  x0_hat <- A_dag y + P_N x0_hat        (the certified core IP)
  3. RENOISE:  x_{t-1} = sqrt(abar_{t-1}) x0_hat + sqrt(1 - abar_{t-1}) eps_consistent

Determinism: DDIM-style. CRITICAL — after the projection changes x0, the renoise
must use the eps CONSISTENT with the corrected x0, not the stale predicted eps;
otherwise the trajectory becomes internally inconsistent and collapses. This bug
is invisible to a pure oracle (projection is a no-op on the true image); the
semi-oracle gate in the test exercises it.

Reuses RangeNullDecomposition and the prior's eps UNCHANGED.
"""

from __future__ import annotations
import math
from typing import Callable, Optional
import torch
from torch import Tensor

from ..decomposition.range_null import RangeNullDecomposition
from ..priors.schedules import DiscreteLinearSchedule


class DDNMSampler:
  def __init__(
    self,
    eps_fn: Callable[[Tensor, int], Tensor],
    decomposition: RangeNullDecomposition,
    schedule: DiscreteLinearSchedule,
    num_steps: int = 100,
    num_train_steps: int = 1000,
    prior_scale: float = 2.0,
    prior_shift: float = -1.0,
    stochastic: bool = False,
    check_consistency: bool = False,
    consistency_rtol: float = 1e-3,
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

  # --- pipeline <-> model space (measured y and A_dag y live in pipeline space) ---
  def _to_model(self, x: Tensor) -> Tensor:
    return self.scale * x + self.shift

  def _to_pipeline(self, x: Tensor) -> Tensor:
    return (x - self.shift) / self.scale

  def _timesteps(self) -> list[int]:
    """Evenly spaced integer diffusion steps, high noise -> low, inclusive of 0."""
    step = self.T // self.num_steps
    ts = list(range(self.T - 1, -1, -step))
    if ts[-1] != 0:
      ts.append(0)
    return ts

  @torch.no_grad()
  def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
    """Run DDNM sampling. Returns the reconstruction in PIPELINE space,
    measurement-consistent (A x_hat = y)."""
    dec = self.dec
    shape = dec.image_shape
    device = dec.y.device

    # start from pure noise at the highest level, in model space.
    x_t = torch.randn(shape, device=device, generator=generator)

    ts = self._timesteps()
    for i, t in enumerate(ts):
      abar_t = float(self.schedule.alpha_bar_at_step(t))
      sqrt_abar_t = math.sqrt(abar_t)
      sqrt_1m_abar_t = math.sqrt(max(1.0 - abar_t, 0.0))

      # 1. DENOISE: predict clean image x0 (model space) from x_t.
      eps = self.eps_fn(x_t, t)
      x0_model = (x_t - sqrt_1m_abar_t * eps) / max(sqrt_abar_t, 1e-8)

      # 2. PROJECT: range-null correction in PIPELINE space (where A, A_dag live).
      x0_pipeline = self._to_pipeline(x0_model)
      x0_pipeline = dec.reconstruct(x0_pipeline)        # A_dag y + P_N x0
      if self.check_consistency:
        dec.assert_consistent(x0_pipeline, rtol=self.consistency_rtol)
      x0_model = self._to_model(x0_pipeline)

      # last step: return the corrected clean estimate directly.
      if i == len(ts) - 1:
        return x0_pipeline

      # 3. RENOISE to the next (lower) level.
      t_next = ts[i + 1]
      abar_next = float(self.schedule.alpha_bar_at_step(t_next))
      sqrt_abar_next = math.sqrt(abar_next)
      sqrt_1m_abar_next = math.sqrt(max(1.0 - abar_next, 0.0))

      if self.stochastic:
        noise = torch.randn(shape, device=device, generator=generator)
      else:
        # DDIM-deterministic. CRITICAL: after the range-null projection the
        # predicted eps no longer matches the CORRECTED x0. Re-noising with the
        # stale eps makes the trajectory internally inconsistent (a collapse the
        # pure oracle cannot expose, since projection is a no-op on the true
        # image). Recompute the eps consistent with the corrected x0:
        #   eps_consistent = (x_t - sqrt(abar_t) x0) / sqrt(1 - abar_t)
        if sqrt_1m_abar_t > 1e-8:
          noise = (x_t - sqrt_abar_t * x0_model) / sqrt_1m_abar_t
        else:
          noise = eps
      x_t = sqrt_abar_next * x0_model + sqrt_1m_abar_next * noise

    return self._to_pipeline(x_t)  # unreachable (loop returns at last step)