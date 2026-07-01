from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .operators import ForwardOperator
from .decomposition.range_null import RangeNullDecomposition
from .priors.base import VelocityPrior
from .priors.normalized import NormalizedVelocityPrior
from .samplers.null_space_flow import NullSpaceFlowSampler

@dataclass
class ReconstructionConfig:
  num_steps: int = 50
  method: str = "heun"
  cg_iters: int = 800
  prior_scale: float = 2.0
  prior_shift: float = -1.0
  check_consistency: bool = False

class SuperResReconstructor:
  def __init__(self, operator: ForwardOperator, prior: VelocityPrior, config: Optional[ReconstructionConfig] = None)-> None:
    self.operator = operator
    self.config = config or ReconstructionConfig()
    self.prior = NormalizedVelocityPrior(
      prior, scale=self.config.prior_scale, shift=self.config.prior_shift,
    )
  
  @torch.no_grad()
  def reconstruct(self, y: Tensor, image_shape: tuple, z0: Optional[Tensor] = None) -> dict:
    dec = RangeNullDecomposition(
      self.operator, y, image_shape=image_shape, cg_iters=self.config.cg_iters,
    )
    sampler = NullSpaceFlowSampler(
      self.prior, dec,
      num_steps=self.config.num_steps, method=self.config.method,
      check_consistency=self.config.check_consistency,
    )
    if z0 is None:
      z0 = torch.randn(image_shape, device=y.device, dtype=y.dtype)

    x_hat = sampler.sample(z0)
    return {
      "x_hat": x_hat,
      "consistency_residual": dec.consistency_residual(x_hat),
      "pinv_floor": dec.pinv_reconstruction, 
    }