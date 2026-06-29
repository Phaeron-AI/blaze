from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import Tensor

class ForwardOperator(ABC):
  
  @abstractmethod
  def A(self, x: Tensor)-> Tensor:
    pass

  def A_T(self, y: Tensor)-> Tensor:
    pass

  def __call__(self, x: Tensor)-> Tensor:
    return self.A(x)