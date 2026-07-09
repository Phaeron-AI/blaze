"""Optional base class for operators, providing shared behaviour.

The operator CONTRACT lives in engine.src.protocols.ForwardOperator (the single
source of truth for the method names and signatures). This module does not
re-declare that contract; it provides a convenience base an operator MAY inherit
to get the __call__ shortcut and to have the type checker verify, via the
declared ForwardOperator base, that it implements A and A_T correctly.

Inheriting BaseOperator is optional: any class with A and A_T structurally
satisfies the Protocol. Inherit it only for the shared __call__ helper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from ..protocols import ForwardOperator


class BaseOperator(ForwardOperator, ABC):
  """Abstract operator base with a call shortcut.

  Subclasses implement A and A_T (the contract from ForwardOperator). Declaring
  ForwardOperator as a base makes mypy check that the signatures match the
  single source of truth, so a name or signature drift is a type error here
  rather than a runtime surprise at a call site.
  """

  @abstractmethod
  def A(self, x: Tensor) -> Tensor: ...

  @abstractmethod
  def A_T(self, y: Tensor) -> Tensor: ...

  def __call__(self, x: Tensor) -> Tensor:
    return self.A(x)