"""Forward operator A x = S (k * x): PSF blur then phase-centered subsample.

The electron-microscopy stand-in (Derivations section 2). Two properties are
essential and both are certified: the adjoint matches A exactly, and the
sampling is anti-aliased and phase-centered.

Anti-aliasing: a downsampling operator must low-pass below the new Nyquist
frequency, or aliased high frequencies fold back as a checkerboard grid that a
generative prior then amplifies. The blur is the anti-alias filter, so its width
must scale with the downsampling factor; a fixed narrow PSF under-filters at
large scales and produces the grid. Use matched_psf for a correctly sized PSF.

Phase-centering: naive strided slicing samples from pixel 0, half a block
off-center, which the alignment certification flags as sub-pixel drift. Sampling
from offset scale // 2 puts the sample at each block's center, so an impulse
round-trips without spatial shift.

The adjoint needs the original spatial size, since A discards it; pass it to AT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseOperator


class BlurDownsampleOperator(BaseOperator):
  def __init__(self, kernel: torch.Tensor, scale: int):
    if kernel.dim() != 2:
      raise ValueError(f"kernel must be 2D (kH,kW), got {tuple(kernel.shape)}")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
      raise ValueError("kernel dims must be odd so the PSF has a defined center")
    self.scale = int(scale)
    k = kernel / kernel.sum()
    self.kH, self.kW = k.shape
    self.weight = k.view(1, 1, self.kH, self.kW)
    self.pad = (self.kW // 2, self.kH // 2)

    # Sample from the center of each scale x scale block, not pixel 0, so the
    # downsampling introduces no sub-pixel shift (alignment certification).
    self.offset = self.scale // 2

  def to(self, device_or_dtype: object) -> BlurDownsampleOperator:
    self.weight = self.weight.to(device_or_dtype) # type: ignore
    return self

  def A(self, x: torch.Tensor) -> torch.Tensor:
    self._in_hw = (x.shape[-2], x.shape[-1])
    c = x.shape[1]
    w = self.weight.to(device=x.device, dtype=x.dtype).expand(c, 1, self.kH, self.kW)
    xp = F.pad(x, (self.pad[0], self.pad[0], self.pad[1], self.pad[1]))
    blurred = F.conv2d(xp, w, groups=c)
    return blurred[:, :, self.offset :: self.scale, self.offset :: self.scale]

  def A_T(self, y: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    # A discards the exact input size, so the adjoint needs it back. Prefer the
    # size cached by the most recent A call; fall back to y * scale.
    n, c, h, wdt = y.shape
    if out_hw is None:
      out_hw = getattr(self, "_in_hw", (h * self.scale, wdt * self.scale))
    hh, ww = out_hw 

    w = self.weight.to(device=y.device, dtype=y.dtype).expand(c, 1, self.kH, self.kW)
    up = y.new_zeros(n, c, hh, ww)
    up[:, :, self.offset :: self.scale, self.offset :: self.scale] = y
    out = F.conv_transpose2d(up, w, groups=c)
    return out[:, :, self.pad[1] : self.pad[1] + hh, self.pad[0] : self.pad[0] + ww]


def gaussian_psf(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
  ax = torch.arange(size) - size // 2
  g = torch.exp(-(ax**2) / (2 * sigma**2))
  return g[:, None] * g[None, :]


def matched_psf(scale: int) -> torch.Tensor:
  # Anti-alias PSF matched to the downsampling factor: sigma grows with scale so
  # the filter always cuts below the new Nyquist frequency. The kernel spans
  # about three sigma each side and is forced odd for a defined center.
  sigma = 0.5 * scale
  size = int(2 * round(3 * sigma) + 1)
  return gaussian_psf(size, sigma)