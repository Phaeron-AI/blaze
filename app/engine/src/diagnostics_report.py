from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .operators.blur_downsample import BlurDownsampleOperator, gaussian_psf


def matched_psf_sigma(scale: int) -> float:
  # A downsampling operator must low-pass below the new Nyquist frequency or
  # aliased high frequencies fold back as a checkerboard grid that the prior
  # then amplifies. A Gaussian anti-alias filter with sigma proportional to
  # the scale factor suppresses that grid; sigma ~= scale is the practical
  # rule that also keeps the sampling phase centered.
  return float(scale)


def build_sr_operator(scale: int, device: str = "cpu") -> BlurDownsampleOperator:
  # Build a super-resolution operator whose blur is matched to the scale so it
  # acts as a proper anti-alias filter. The kernel spans ~3 sigma each side.
  sigma = matched_psf_sigma(scale)
  size = int(2 * round(3 * sigma) + 1)
  return BlurDownsampleOperator(gaussian_psf(size, sigma).to(device), scale=scale) 


def checkerboard_energy(image: Tensor) -> float:
  # Quantify grid/checkerboard artifacts: the mean absolute deviation of the
  # image from its local 3x3 mean. A smooth reconstruction scores near zero; a
  # gridded one scores high because the checkerboard is high-frequency detail
  # that a local average cannot follow.
  gray = image.mean(1, keepdim=True)
  kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9.0
  smoothed = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kernel)
  return float((gray - smoothed).abs().mean())


@dataclass
class DiagnosticMaps:
  difference: Tensor
  error: Tensor | None
  recon_checkerboard: float
  floor_checkerboard: float

  @property
  def prior_added_artifacts(self) -> bool:
    # True when the reconstruction is markedly more gridded than the floor,
    # which flags the prior amplifying high-frequency artifacts rather than
    # adding faithful detail.
    return self.recon_checkerboard > self.floor_checkerboard * 1.5


def compute_diagnostics(
  reconstruction: Tensor,
  floor: Tensor,
  ground_truth: Tensor | None = None,
) -> DiagnosticMaps:
  # The difference map isolates exactly what the prior contributed on top of
  # the linear-algebra floor. In a healthy reconstruction it is concentrated
  # in genuine texture; a uniform or gridded difference map signals the prior
  # is injecting artifacts rather than recovering structure.
  difference = (reconstruction - floor).abs()
  error = None
  if ground_truth is not None:
    error = (reconstruction - ground_truth).abs()
  return DiagnosticMaps(
    difference=difference,
    error=error,
    recon_checkerboard=checkerboard_energy(reconstruction),
    floor_checkerboard=checkerboard_energy(floor),
  )