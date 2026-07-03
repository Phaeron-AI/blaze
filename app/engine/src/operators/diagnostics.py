from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional

from .base import ForwardOperator
from .pseudoinverse import ConjugateGradientInverse


def _psnr(a: Tensor, b: Tensor, data_range: float = 1.0) -> float:
  mse = torch.mean((a - b) ** 2).item()
  if mse == 0:
    return float("inf")
  return 10.0 * torch.log10(Tensor(data_range**2 / mse)).item()


def adjoint_test(op: ForwardOperator, image_shape: tuple[int, ...], device, seed: int = 0)-> tuple[Optional[float], Optional[float], float]:
  g = torch.Generator(device=device).manual_seed(seed)
  x = torch.randn(*image_shape, generator=g, device=device)
  y = op.A(x)
  yr = torch.randn(*y.shape, generator=g, device=device)
  ATyr = op.A_T(yr)
  if not isinstance(ATyr, Tensor) or ATyr.shape != x.shape:
    return None, None, float("inf") 
  lhs = (op.A(x) * yr).sum().item()
  rhs = (x * ATyr).sum().item()
  rel = abs(lhs - rhs) / (abs(lhs) + 1e-12)
  return lhs, rhs, rel


def consistency_test(op: ForwardOperator, x: Tensor, max_iters: int = 200):
  inv = ConjugateGradientInverse(op, max_iters=max_iters)
  y = op.A(x)
  x_dag = inv.solve(y, x.shape)
  y_round = op.A(x_dag)
  return _psnr(y_round, y), (y_round - y).abs().max().item()


def reconstruction_floor(op: ForwardOperator, x: Tensor, max_iters: int = 200) -> float:
  inv = ConjugateGradientInverse(op, max_iters=max_iters)
  y = op.A(x)
  return _psnr(inv.solve(y, x.shape), x)


def impulse_alignment_test(op: ForwardOperator, H: int, W: int, device, max_iters: int = 200):
  inv = ConjugateGradientInverse(op, max_iters=max_iters)
  cy, cx = H // 2, W // 2
  delta = torch.zeros(1, 1, H, W, device=device)
  delta[0, 0, cy, cx] = 1.0
  resp = inv.solve(op.A(delta), delta.shape)[0, 0].abs()
  total = resp.sum().clamp(min=1e-12)
  ys = torch.arange(H, device=device).float()
  xs = torch.arange(W, device=device).float()
  cy_hat = (resp.sum(1) * ys).sum() / total
  cx_hat = (resp.sum(0) * xs).sum() / total
  dy = (cy_hat - cy).item()
  dx = (cx_hat - cx).item()
  return dy, dx, (dy**2 + dx**2) ** 0.5


def assert_adjoint(op: ForwardOperator, image_shape, device, rtol: float = 1e-4) -> None:
  lhs, _, rel = adjoint_test(op, image_shape, device)
  if lhs is None:
    raise ValueError(
      "Operator failed adjoint certification: A^T output shape != image shape "
      "(A and A^T are not a transpose pair). Refusing to build decomposition."
    )
  if rel >= rtol:
    raise ValueError(
      f"Operator failed adjoint test: <Ax,y> vs <x,A^T y> relative error "
      f"{rel:.2e} >= {rtol:.0e}. A^T is not the true adjoint of A. "
      "Range-null projectors would not be orthogonal. Refusing to build."
    )


def assert_alignment(op: ForwardOperator, H: int, W: int, device, max_drift: float = 0.05) -> None:
  _, _, drift = impulse_alignment_test(op, H, W, device)
  if drift > max_drift:
    raise ValueError(
      f"Operator failed alignment test: impulse centroid drift {drift:.3f}px "
      f"> {max_drift}px. A sub-pixel spatial shift is present (pixel-shuffle / "
      "resample-phase bug). Refusing to build decomposition."
    )
