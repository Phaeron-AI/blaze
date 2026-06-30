from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _check_inputs(a: Tensor, b: Tensor, data_range: float)-> None:
  if a.shape != b.shape:
    raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
  if data_range < 0:
    raise ValueError(f"Data Range must be positive. Got: {data_range}")
  
  for name, t in (("pred", a), ("target", b)):
    lo, hi = t.min().item(), t.max().item()

    if hi-lo > data_range * 1.5 + 1e-6:
      raise ValueError (
        f"{name} spans {hi - lo:.3f} but data_range={data_range}. "
        "Likely a normalization mismatch (e.g. [-1,1] data scored as [0,1]). "
        "Pass the correct data_range or renormalize before scoring."
      )

def psnr(pred: Tensor, target: Tensor, data_range: float = 1.0)-> float:
  _check_inputs(pred, target, data_range)
  mse = torch.mean((pred.float() - target.float()) ** 2).item()
  if mse == 0:
    return float("inf")
  return 10.0 * math.log10(data_range ** 2 / mse)

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype)-> Tensor:
  coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
  g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
  g = (g / g.sum()).unsqueeze(0)
  window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
  return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: Tensor, target: Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5)-> float:
  _check_inputs(pred, target, data_range)
  a = pred.float()
  b = target.float()
  # normalize shape to (N,C,H,W)
  while a.dim() < 4:
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
  n, c, h, w = a.shape
  if min(h, w) < window_size:
      raise ValueError(f"image {h}x{w} smaller than window {window_size}")

  win = _gaussian_window(window_size, sigma, c, a.device, a.dtype)
  pad = window_size // 2

  mu_a = F.conv2d(a, win, padding=pad, groups=c)
  mu_b = F.conv2d(b, win, padding=pad, groups=c)
  mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

  sigma_a2 = F.conv2d(a * a, win, padding=pad, groups=c) - mu_a2
  sigma_b2 = F.conv2d(b * b, win, padding=pad, groups=c) - mu_b2
  sigma_ab = F.conv2d(a * b, win, padding=pad, groups=c) - mu_ab

  c1 = (0.01 * data_range) ** 2
  c2 = (0.03 * data_range) ** 2
  ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
  return ssim_map.mean().item()

def evaluate(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> dict:
  return {
    "psnr_db": psnr(pred, target, data_range),
    "ssim": ssim(pred, target, data_range),
  }