"""Adapter from a score/eps-parameterized prior to a velocity field.

Converts the noise prediction of a diffusion model into the velocity of
the probability-flow ODE, so a pretrained score model can drive a flow
sampler. The conversion follows the score-SDE probability-flow ODE (Song,
Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, "Score-Based Generative
Modeling through SDEs", ICLR 2021) under the VP/DDPM schedule. See
Derivations sections 5 and 6.

There are two independent, sign-critical conventions here, both verified
against an analytic Gaussian where the velocity is known in closed form:
the score sign (score points toward clean data, eps toward noise), and
the flow-time orientation (the sampler integrates noise-to-clean while the
diffusion process runs clean-to-noise). Getting either wrong yields
plausible-looking but incorrect output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from typing import Any

from .base import VelocityPrior


@dataclass
class NoiseSchedule:
  """Continuous VP schedule providing beta and alpha_bar over [0, 1].

  Defaults to the standard linear VP schedule. For the discrete DDPM
  schedule the borrowed prior was trained on, use DiscreteLinearSchedule
  instead; both satisfy this interface.
  """

  beta_min: float = 0.1
  beta_max: float = 20.0

  def beta(self, tau: Tensor | float) -> Tensor:
    tau = torch.as_tensor(tau, dtype=torch.float32)
    return self.beta_min + tau * (self.beta_max - self.beta_min)

  def alpha_bar(self, tau: Tensor | float) -> Tensor:
    # Continuous VP form: alpha_bar(tau) = exp(-integral_0^tau beta(s)
    # ds), closed form for a linear beta.
    tau = torch.as_tensor(tau, dtype=torch.float32)
    integral = self.beta_min * tau + 0.5 * (self.beta_max - self.beta_min) * tau**2
    return torch.exp(-integral)


class ScoreToVelocity(VelocityPrior):
  """Wrap an eps- or score-predicting model as a flow VelocityPrior.

  Parameters
  ----------
  model_fn : Callable[[Tensor, Tensor], Tensor]
      Returns a tensor shaped like its input, interpreted as predicted
      noise (parameterization "eps") or the score (parameterization
      "score").
  schedule : NoiseSchedule or None
      The VP schedule supplying beta and alpha_bar.
  parameterization : str
      Either "eps" or "score".
  flow_time_is_clean_at_one : bool
      If True, flow time t = 1 is the clean end and t = 0 is pure noise,
      so diffusion time is tau = 1 - t and the returned velocity is
      negated to point toward clean.
  """

  def __init__(
    self,
    model_fn: Callable[[Tensor, Tensor], Tensor],
    schedule: NoiseSchedule | None = None,
    parameterization: str = "eps",
    flow_time_is_clean_at_one: bool = True,
  ) -> None:
    if parameterization not in ("eps", "score"):
      raise ValueError("parameterization must be 'eps' or 'score'")
    self.model_fn = model_fn
    self.schedule = schedule or NoiseSchedule()
    self.parameterization = parameterization
    self.flow_clean_at_one = flow_time_is_clean_at_one

  def _score(self, x_tau: Tensor, tau: float) -> Tensor:
    out = self.model_fn(x_tau, torch.as_tensor(tau, device=x_tau.device))
    if self.parameterization == "score":
      return out
    # eps -> score:  s = -eps_hat / sqrt(1 - alpha_bar)
    ab = self.schedule.alpha_bar(tau).to(x_tau.device)
    return -out / torch.sqrt(1.0 - ab).clamp(min=1e-8)

  def velocity(self, x_t: Tensor, t: Any, **cond: Any) -> Tensor:
    """Return the flow velocity at flow time t.

    Parameters
    ----------
    x_t : Tensor, shape (N, C, H, W)
        The current state, in the prior's native space.
    t : float
        Flow time in [0, 1].

    Returns
    -------
    Tensor, shape (N, C, H, W)
        The probability-flow velocity, oriented for the sampler.
    """
    t_val = float(t)

    # Map flow time to diffusion time. When the clean end is at t = 1,
    # the diffusion process (clean at tau = 0) is the reverse
    # orientation.
    tau = (1.0 - t_val) if self.flow_clean_at_one else t_val
    beta = self.schedule.beta(tau).to(x_t.device)
    score = self._score(x_t, tau)

    # Probability-flow drift for the VP-SDE (Derivations section 5): v =
    # -0.5 beta (x + score), in diffusion time.
    v_diffusion = -0.5 * beta * (x_t + score)

    # The sampler integrates the reverse orientation, so negate the
    # drift to point toward the clean end.
    return -v_diffusion if self.flow_clean_at_one else v_diffusion
