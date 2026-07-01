from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F

from .base import ForwardOperator


class BlurDownsampleOperator(ForwardOperator):
  def __init__(self, kernel: Tensor, scale: int):
    if kernel.dim() != 2:
      raise ValueError(f"kernel must be 2D (kH,kW), got shape {tuple(kernel.shape)}")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
      raise ValueError("kernel dims must be odd so the PSF has a well-defined center")
    self.scale = int(scale)
    k = kernel / kernel.sum()
    self.kH, self.kW = k.shape
    self.weight = k.view(1, 1, self.kH, self.kW)
    self.pad = (self.kW // 2, self.kH // 2)

  def to(self, device_or_dtype) -> "BlurDownsampleOperator":
    self.weight = self.weight.to(device_or_dtype)
    return self

  def A(self, x: Tensor) -> Tensor:
    c = x.shape[1]
    w = self.weight.to(device=x.device, dtype=x.dtype).expand(c, 1, self.kH, self.kW)
    xp = F.pad(x, (self.pad[0], self.pad[0], self.pad[1], self.pad[1]))
    blurred = F.conv2d(xp, w, groups=c)
    return blurred[:, :, ::self.scale, ::self.scale]

  def A_T(self, y: Tensor) -> Tensor:
    n, c, h, wdt = y.shape
    w = self.weight.to(device=y.device, dtype=y.dtype).expand(c, 1, self.kH, self.kW)
    # adjoint of strided slice = scatter into a zero grid
    up = y.new_zeros(n, c, h * self.scale, wdt * self.scale)
    up[:, :, ::self.scale, ::self.scale] = y
    # adjoint of grouped conv2d(pad(.)) = grouped conv_transpose2d, then un-pad
    out = F.conv_transpose2d(up, w, groups=c)
    return out[:, :, self.pad[1]:self.pad[1] + h * self.scale, self.pad[0]:self.pad[0] + wdt * self.scale]


def gaussian_psf(size: int = 5, sigma: float = 1.0) -> Tensor:
  ax = torch.arange(size) - size // 2
  g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
  return g[:, None] * g[None, :]