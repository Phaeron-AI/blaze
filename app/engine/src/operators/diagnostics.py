"""Operator certification: the four invariants from Phase 0 as reusable checks.

These are the gate that keeps a misadjoint or misaligned operator out of every
pipeline. RangeNullDecomposition calls the assert_* wrappers in its constructor,
so a bad operator physically cannot instantiate the decomposition.

  1. adjoint_test        <Ax,y> == <x,A_T y>        -> A_T really is the transpose
  2. consistency_test    A A_dagger y == y          -> A_dagger inverts A on range(A)
  3. reconstruction      PSNR(A_dagger y, x)        -> the honest reconstruction floor
  4. impulse_alignment   centroid(A_dagger A delta) -> no sub-pixel spatial drift

See Derivations section 2 for the why behind each.
"""

from __future__ import annotations

import torch

from ..protocols import ForwardOperator
from .pseudoinverse import ConjugateGradientInverse


def _psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> float:
  mse = torch.mean((a - b) ** 2).item()
  if mse == 0:
    return float("inf")
  return 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()


def adjoint_test(
  op: ForwardOperator,
  image_shape: tuple[int, ...],
  device: str | torch.device,
  seed: int = 0,
) -> tuple[float | None, float | None, float]:
  # Returns (lhs, rhs, rel_error); rel_error < 1e-4 means pass. lhs is None if
  # A_T is malformed (returns None, a non-tensor, or a shape that disagrees with
  # the image shape). The gate must never crash on a broken operator; it must
  # resolve to a refusal.
  g = torch.Generator(device=device).manual_seed(seed)
  x = torch.randn(*image_shape, generator=g, device=device)
  y = op.A(x)
  yr = torch.randn(*y.shape, generator=g, device=device)
  a_t_yr = op.A_T(yr)
  if not isinstance(a_t_yr, torch.Tensor) or a_t_yr.shape != x.shape:
    return None, None, float("inf")
  lhs = (op.A(x) * yr).sum().item()
  rhs = (x * a_t_yr).sum().item()
  rel = abs(lhs - rhs) / (abs(lhs) + 1e-12)
  return lhs, rhs, rel


def consistency_test(
  op: ForwardOperator, x: torch.Tensor, max_iters: int = 200
) -> tuple[float, float]:
  # A A_dagger y == y for y = A x (so y lies in range(A)). Returns
  # (psnr_dB, max_residual).
  inv = ConjugateGradientInverse(op, max_iters=max_iters)
  y = op.A(x)
  x_dag = inv.solve(y, x.shape)
  y_round = op.A(x_dag)
  return _psnr(y_round, y), (y_round - y).abs().max().item()


def reconstruction_floor(
  op: ForwardOperator, x: torch.Tensor, max_iters: int = 200
) -> float:
  # PSNR(A_dagger y, x): the floor the generative null-space component climbs from.
  inv = ConjugateGradientInverse(op, max_iters=max_iters)
  y = op.A(x)
  return _psnr(inv.solve(y, x.shape), x)


def impulse_alignment_test(
  op: ForwardOperator, H: int, W: int, device: str | torch.device, max_iters: int = 200
) -> tuple[float, float, float]:
  # Centroid drift of A_dagger A applied to a single bright pixel. Returns
  # (dy, dx, drift) in pixels; drift ~ 0 for an aligned operator. Measured
  # through A_dagger, so treat as a binary gate, not a calibrated ruler (the
  # inner CG loop amplifies magnitude). For true sub-pixel drift, probe A_T A.
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


# Assert wrappers, called by RangeNullDecomposition's constructor.


def assert_adjoint(
  op: ForwardOperator, image_shape: tuple[int, ...], device: str | torch.device, rtol: float = 1e-4
) -> None:
  lhs, _, rel = adjoint_test(op, image_shape, device)
  if lhs is None:
    raise ValueError(
      "Operator failed adjoint certification: A_T output shape != image shape "
      "(A and A_T are not a transpose pair). Refusing to build decomposition."
    )
  if rel >= rtol:
    raise ValueError(
      f"Operator failed adjoint test: <Ax,y> vs <x,A_T y> relative error "
      f"{rel:.2e} >= {rtol:.0e}. A_T is not the true adjoint of A. "
      "Range-null projectors would not be orthogonal. Refusing to build."
    )


def assert_alignment(
  op: ForwardOperator, H: int, W: int, device: str | torch.device, max_drift: float = 0.05
) -> None:
  _, _, drift = impulse_alignment_test(op, H, W, device)
  if drift > max_drift:
    raise ValueError(
      f"Operator failed alignment test: impulse centroid drift {drift:.3f}px "
      f"> {max_drift}px. A sub-pixel spatial shift is present (pixel-shuffle / "
      "resample-phase bug). Refusing to build decomposition."
    )