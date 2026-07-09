from __future__ import annotations

import torch
from torch import Tensor

from typing import Any

from ..decomposition.range_null import RangeNullDecomposition
from ..priors import VelocityPrior


class NullSpaceFlowSampler:
  def __init__(
    self,
    prior: VelocityPrior,
    decomposition: RangeNullDecomposition,
    num_steps: int = 50,
    method: str = "heun",  # 'euler' (1st order) or 'heun' (2nd order)
    check_consistency: bool = False,
    consistency_rtol: float = 1e-3,
  ):
    if method not in ("euler", "heun"):
      raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")
    self.prior = prior
    self.dec = decomposition
    self.num_steps = num_steps
    self.method = method
    self.check_consistency = check_consistency
    self.consistency_rtol = consistency_rtol

  def _null_v(self, x_hat: Tensor, t: float, **cond: Any) -> Tensor:
    """P_N v_phi(x_hat, t): the learned velocity projected into the null space."""
    v = self.prior.velocity(x_hat, t, **cond)
    return self.dec.null_velocity(v)

  @torch.no_grad()
  def sample(self, z0: Tensor, **cond: Any) -> Tensor:
    """Integrate from a base draw z0 to the t=1 reconstruction.

    Returns x_hat at t=1, guaranteed measurement-consistent: A x_hat = y.
    z0: (N,C,H,W) base noise; only its null component matters (the range
        component is supplied by A^dagger y).
    """
    # Start ON the consistent subspace: x_hat_0 = A^dagger y + P_N z0.
    x_hat = self.dec.reconstruct(z0)
    dt = 1.0 / self.num_steps

    for i in range(self.num_steps):
      t = i * dt
      if self.method == "euler":
        v = self._null_v(x_hat, t, **cond)
        x_hat = x_hat + dt * v
      else:  # Heun: predictor-corrector, 2nd order
        v1 = self._null_v(x_hat, t, **cond)
        x_pred = x_hat + dt * v1
        v2 = self._null_v(x_pred, min(t + dt, 1.0), **cond)
        x_hat = x_hat + 0.5 * dt * (v1 + v2)

      if self.check_consistency:
        self.dec.assert_consistent(x_hat, rtol=self.consistency_rtol)

    return x_hat
