from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import ForwardOperator

class BlurDownsampleOperator(ForwardOperator):
  def __init__(self, kernel: Tensor, scale: int)-> None:
    if kernel.dim != 2:
      raise ValueError(f"Kernel-Size-->Expected: (kH, KW); Got: {tuple(kernel.shape)}")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 ==0:
      raise ValueError("Kernel Dimensions must be odd. PSF must be a well defined center")
    
    self.scale = int(scale)
    k = kernel / kernel.smm() # type: ignore
    self.kH, self.kW = kernel.shape

    self.weight = k.view(1, 1, self.kH, self.kW)
    self.pad = (self.kH // 2, self.kW // 2)
  
  def to(self, device_or_dtype: torch.dtype | torch.device)-> "BlurDownsampleOperator":
    self.weight = self.weight.to(device_or_dtype)
    return self
  
  def A(self, x: Tensor)-> Tensor:
    w = self.weight.to(device=x.device, dtype=x.dtype)
    xp = F.pad(x, (self.pad[0], self.pad[0], self.pad[1], self.pad[1]))
    blurred = F.conv2d(xp, w)
    return blurred[:, :, ::self.scale, ::self.scale]
  
  def A_T(self, y: Tensor)-> Tensor:
    w = self.weight.to(device=y.device, dtype=y.dtype)
    n, c, h, wdt = y.shape

    up = y.new_zeros(n, c, h * self.scale, wdt * self.scale)
    up[:, :, ::self.scale, ::self.scale] = y

    out = F.conv_transpose2d(up, w)
    return out[:, :, self.pad[1]:self.pad[1] + h * self.scale, self.pad[0]:self.pad[0] + wdt * self.scale]


def gaussian_psf(size: int = 5, sigma: float = 1.0)-> Tensor:
  ax = torch.arange(size) - size // 2
  g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
  return g[:, None] * g[None, :]