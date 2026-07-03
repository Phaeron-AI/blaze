from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class ForwardOperator(ABC):
  @abstractmethod
  def A(self, x: Tensor) -> Tensor: ...
  @abstractmethod
  def A_T(self, y: Tensor) -> Tensor: ...

  def __call__(self, x: Tensor) -> Tensor:
    return self.A(x)
