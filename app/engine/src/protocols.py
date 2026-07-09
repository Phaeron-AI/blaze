"""Core interfaces for the engine's substitutable components.

These Protocols are the single source of truth for the contracts that
operators, priors, and samplers must satisfy. Type signatures throughout the
engine reference these Protocols, not concrete classes, so implementations are
swappable and mismatches are caught by the type checker rather than at runtime.

Structural typing means any class with the right methods conforms without
inheriting anything, which keeps test stubs and third-party implementations
first-class. Where shared implementation is useful (for example an operator
base with a __call__ helper), a concrete base class declares that it implements
the relevant Protocol; the method names and signatures still live only here.

The distinction between DiffusionPrior.eps_at_step (integer diffusion step) and
VelocityPrior.velocity (float flow time in [0, 1]) is encoded deliberately.
Conflating the two conventions once produced a silent reconstruction collapse;
with the contracts typed, wiring an integer-step sampler to a flow-time prior is
a type error at construction. See STANDARDS.md section 4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class ForwardOperator(Protocol):
  """A linear measurement operator and its exact adjoint.

  Implementations must satisfy the adjoint identity <A x, y> = <x, A_T y> to
  machine precision, because the orthogonality of the range-null projector
  A_dag A depends on A_T being the true adjoint. This is verified by the
  operator diagnostics, not assumed.
  """

  def A(self, x: Tensor) -> Tensor:
    """Apply the forward operator. Maps image space to measurement space."""
    ...

  def A_T(self, y: Tensor) -> Tensor:
    """Apply the exact adjoint. Maps measurement space to image space."""
    ...


@runtime_checkable
class DiffusionPrior(Protocol):
  """A prior parameterized by noise prediction at integer diffusion steps.

  This is the native interface for diffusion-time samplers such as DDNM. The
  step is an integer in [0, T-1]; do not pass flow time here.
  """

  def eps_at_step(self, x: Tensor, step: int) -> Tensor:
    """Predicted noise at an integer diffusion step.

    Parameters
    ----------
    x : Tensor, shape (N, C, H, W)
        Noised image in the prior's native space.
    step : int
        Diffusion step in [0, T-1].

    Returns
    -------
    Tensor, shape (N, C, H, W)
        Predicted noise.
    """
    ...


@runtime_checkable
class VelocityPrior(Protocol):
  """A prior expressed as a probability-flow velocity field.

  This is the native interface for flow samplers. The time argument is a float
  flow time in [0, 1]; do not pass an integer diffusion step here.
  """

  def velocity(self, x_t: Tensor, t: float) -> Tensor:
    """Flow velocity at flow time t.

    Parameters
    ----------
    x_t : Tensor, shape (N, C, H, W)
        Current state.
    t : float
        Flow time in [0, 1].

    Returns
    -------
    Tensor, shape (N, C, H, W)
        The probability-flow velocity.
    """
    ...


@runtime_checkable
class Sampler(Protocol):
  """A reconstruction sampler that returns a measurement-consistent image.

  Both the flow and DDNM samplers satisfy this, so the composition layer
  depends on a single type. The returned tensor is in pipeline space and
  satisfies the measurement to solver tolerance.
  """

  def sample(self, generator: torch.Generator | None = None) -> Tensor:
    """Run sampling and return a pipeline-space reconstruction."""
    ...