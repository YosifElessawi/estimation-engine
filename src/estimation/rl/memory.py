"""Memory use of training processes, logged so a run shows it fits the dev Mac (SPEC §6 M4)."""

import resource
import sys

import torch


def peak_rss_mb() -> float:
    """This process's peak resident memory in MB (``ru_maxrss`` is bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


def gpu_mb(device: torch.device) -> float:
    """Memory the MPS driver holds for this process in MB, 0 on other devices."""
    if device.type != "mps":
        return 0.0
    return torch.mps.driver_allocated_memory() / (1 << 20)
