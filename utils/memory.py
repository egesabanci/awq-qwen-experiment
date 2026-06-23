"""Memory management utilities for MPS (Apple Silicon).

Provides memory limiting, monitoring, and tracking helpers
to prevent OOM during large calibration runs.
"""

import gc
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch


def get_mps_memory() -> dict:
    """Get current MPS memory stats in GB.

    Returns:
        dict with 'current_gb', 'driver_allocated_gb',
        and 'recommended_max_gb' keys.
    """
    if not torch.backends.mps.is_available():
        return {"error": "MPS not available"}

    stats = {}
    try:
        stats["current_gb"] = torch.mps.current_allocated_memory() / 1e9
    except Exception:
        stats["current_gb"] = 0.0
    try:
        stats["driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
    except Exception:
        stats["driver_allocated_gb"] = 0.0
    try:
        stats["recommended_max_gb"] = torch.mps.recommended_max_memory() / 1e9
    except Exception:
        stats["recommended_max_gb"] = 0.0

    return stats


def get_system_memory() -> dict:
    """Get system RAM stats in GB (macOS)."""
    try:
        import subprocess

        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        page_size = 16384  # macOS default
        stats = {}
        for line in lines:
            if "page size" in line.lower():
                try:
                    page_size = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            if "free" in line.lower():
                try:
                    stats["free_pages"] = int(line.split(":")[1].strip().rstrip("."))
                except (ValueError, IndexError):
                    pass
            if "active" in line.lower() and "file-backed" not in line.lower():
                try:
                    stats["active_pages"] = int(line.split(":")[1].strip().rstrip("."))
                except (ValueError, IndexError):
                    pass
            if "wired" in line.lower():
                try:
                    stats["wired_pages"] = int(line.split(":")[1].strip().rstrip("."))
                except (ValueError, IndexError):
                    pass

        free_gb = stats.get("free_pages", 0) * page_size / 1e9
        active_gb = stats.get("active_pages", 0) * page_size / 1e9
        wired_gb = stats.get("wired_pages", 0) * page_size / 1e9
        return {
            "free_gb": round(free_gb, 1),
            "active_gb": round(active_gb, 1),
            "wired_gb": round(wired_gb, 1),
            "used_gb": round(active_gb + wired_gb, 1),
        }
    except Exception:
        return {"error": "Could not query system memory"}


def log_memory(tag: str = "") -> None:
    """Print current MPS + system memory to stderr."""
    mps = get_mps_memory()
    sys_mem = get_system_memory()

    parts = []
    if tag:
        parts.append(f"[{tag}]")
    if "current_gb" in mps:
        max_gb = mps.get("recommended_max_gb", 0)
        pct = (mps["current_gb"] / max_gb * 100) if max_gb > 0 else 0
        parts.append(f"MPS: {mps['current_gb']:.2f}/{max_gb:.1f} GB ({pct:.0f}%)")
    if "free_gb" in sys_mem:
        parts.append(f"RAM free: {sys_mem['free_gb']:.1f} GB")

    print(" | ".join(parts), flush=True)


def limit_mps_memory(fraction: float = 0.7) -> None:
    """Limit MPS memory to a fraction of recommended max.

    Set this before model loading. The fraction applies to
    ``torch.mps.recommended_max_memory()`` which is typically
    ~75% of total unified memory on Apple Silicon.

    Args:
        fraction: Fraction of recommended max to allow (0.1 - 1.0).
                 Default 0.7 = 70%.
    """
    if not torch.backends.mps.is_available():
        print("[WARN] MPS not available, memory limit not applied")
        return

    fraction = max(0.1, min(1.0, fraction))
    torch.mps.set_per_process_memory_fraction(fraction)
    max_mem = torch.mps.recommended_max_memory()
    print(f"[MEM] MPS limited to {fraction:.0%} = {max_mem / 1e9:.1f} GB")


def empty_cache() -> None:
    """Clear MPS cache and run GC."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@contextmanager
def memory_tracker(tag: str = "track"):
    """Context manager that logs memory before/after a block.

    Also emptys cache on exit.

    Usage:
        with memory_tracker("calibration_pass"):
            run_calibration(...)
    """
    log_memory(f"{tag}_start")
    try:
        yield
    finally:
        empty_cache()
        log_memory(f"{tag}_end")


def batch_generator(
    items: list,
    batch_size: int = 10,
    progress_callback: Callable | None = None,
):
    """Yield items in batches with optional progress callback."""
    total = len(items)
    for i in range(0, total, batch_size):
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
        yield items[i : i + batch_size]
