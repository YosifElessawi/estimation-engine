"""Typed wrappers for the few PyTorch calls whose stubs leave a parameter type unknown, so the
rl package stays clean under pyright's strict mode."""

import torch


def backward(loss: torch.Tensor) -> None:
    """Backpropagate a scalar loss."""
    loss.backward()  # pyright: ignore[reportUnknownMemberType]  # torch's stub leaves the optional gradient argument untyped


def step(optimizer: torch.optim.Optimizer) -> None:
    """One optimizer step."""
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]  # torch's stub leaves the optional closure argument untyped


def seed_torch(seed: int) -> None:
    """Seed PyTorch's global generator, which initializes new nets."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]  # torch's stub leaves the seed parameter untyped


def seeded_generator(seed: int) -> torch.Generator:
    """A CPU random generator with its own seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]  # torch's stub leaves the seed parameter untyped
    return generator
