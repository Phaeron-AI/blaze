from __future__ import annotations

from torch import Tensor

from ..operators.base import ForwardOperator
from ..operators.pseudoinverse import ConjugateGradientInverse
from ..operators.diagnostics import assert_adjoint, assert_alignment


class RangeNullDecomposition:
  def __init__(
    self,
    operator: ForwardOperator,
    y: Tensor,
    image_shape: tuple,
    cg_iters: int = 200,
    certify: bool = True,
    certify_align: bool = True,
  ):
    self.operator = operator
    self.image_shape = tuple(image_shape)
    self.inv = ConjugateGradientInverse(operator, max_iters=cg_iters)

    device = y.device
    if certify:
      assert_adjoint(operator, self.image_shape, device)
    if certify_align:
      H, W = self.image_shape[-2], self.image_shape[-1]
      assert_alignment(operator, H, W, device)

    self.y = y
    self._A_dag_y = self.inv.solve(y, self.image_shape)

  def project_range(self, x: Tensor) -> Tensor:
    return self.inv.solve(self.operator.A(x), self.image_shape)

  def project_null(self, x: Tensor) -> Tensor:
    return x - self.project_range(x)


  def reconstruct(self, z: Tensor) -> Tensor:
    return self._A_dag_y + self.project_null(z)

  def null_velocity(self, v: Tensor) -> Tensor:
    return self.project_null(v)

  @property
  def pinv_reconstruction(self) -> Tensor:
    return self._A_dag_y

  def consistency_residual(self, x_hat: Tensor) -> float:
    num = (self.operator.A(x_hat) - self.y).norm()
    return (num / (self.y.norm() + 1e-12)).item()

  def assert_consistent(self, x_hat: Tensor, rtol: float = 1e-4) -> None:
    res = self.consistency_residual(x_hat)
    if res > rtol:
      raise AssertionError(
        f"Range-space leak: ||A x_hat - y||/||y|| = {res:.2e} > {rtol:.0e}. "
        "The reconstruction has drifted off the measurement-consistent set."
      )