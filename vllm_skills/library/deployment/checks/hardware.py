"""Hardware detection utilities"""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """GPU information"""

    name: str
    memory_total: int  # in MB
    memory_free: int  # in MB
    cuda_version: str
    count: int


@dataclass
class HardwareInfo:
    """Complete hardware information"""

    cpu_count: int
    ram_total: int  # in GB
    ram_available: int  # in GB
    gpu: Optional[GPUInfo]


def detect_gpu() -> Optional[GPUInfo]:
    """
    Detect GPU information using nvidia-smi

    Returns:
        GPUInfo object or None if no GPU detected
    """
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return None

        # Parse first GPU (for simplicity)
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 3:
            return None

        # Get CUDA version
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        cuda_version = cuda_result.stdout.strip()

        return GPUInfo(
            name=parts[0],
            memory_total=int(float(parts[1])),
            memory_free=int(float(parts[2])),
            cuda_version=cuda_version,
            count=len(lines),
        )

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def detect_cpu() -> int:
    """
    Detect CPU count

    Returns:
        Number of CPU cores
    """
    try:
        import psutil

        return psutil.cpu_count(logical=True)
    except ImportError:
        # Fallback to os
        import os

        return os.cpu_count() or 1


def detect_ram() -> tuple[int, int]:
    """
    Detect RAM information

    Returns:
        Tuple of (total_ram_gb, available_ram_gb)
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return (
            int(mem.total / (1024**3)),  # Convert to GB
            int(mem.available / (1024**3)),
        )
    except ImportError:
        return (0, 0)


def check_hardware() -> HardwareInfo:
    """
    Check complete hardware information

    Returns:
        HardwareInfo object with system details
    """
    gpu = detect_gpu()
    cpu_count = detect_cpu()
    ram_total, ram_available = detect_ram()

    return HardwareInfo(
        cpu_count=cpu_count,
        ram_total=ram_total,
        ram_available=ram_available,
        gpu=gpu,
    )
