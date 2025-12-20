import logging
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger("selo.system_profile")


def _safe_import_psutil() -> Optional[Any]:
    try:
        import psutil  # type: ignore

        return psutil
    except Exception as err:
        logger.debug("psutil not available for system profile detection: %s", err)
        return None


def _safe_import_torch() -> Optional[Any]:
    try:
        import torch  # type: ignore

        return torch
    except Exception as err:
        logger.debug("torch not available for system profile detection: %s", err)
        return None


@lru_cache(maxsize=1)
def detect_system_profile() -> Dict[str, Any]:
    """Detect host capabilities and derive generation budgets.

    Returns a cached dictionary with:
        tier: literal describing overall capacity ("high", "standard")
            - "standard": <12GB GPU (optimized for 8GB GPUs)
            - "high": >=12GB GPU (enhanced depth for 12GB+ GPUs)
        total_ram_gb: rounded system RAM (float)
        gpu_memory_gb: rounded total GPU memory for primary device (float | None)
        gpu_name: CUDA device name when available
        budgets.chat_max_tokens: recommended chat completion budget
        budgets.reflection_max_tokens: recommended reflection completion budget
        features.allow_warmup: whether aggressive warmup loops should run
        features.allow_keepalive: whether continuous keep-alive prompts are safe
    """

    psutil = _safe_import_psutil()
    total_ram_gb = 0.0
    if psutil:
        try:
            total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception as err:
            logger.debug("Failed to read total RAM via psutil: %s", err)

    torch = _safe_import_torch()
    gpu_memory_gb = 0.0
    gpu_name: Optional[str] = None
    
    # Try PyTorch CUDA first (most reliable when available)
    if torch and torch.cuda.is_available():  # type: ignore[attr-defined]
        try:
            props = torch.cuda.get_device_properties(0)  # type: ignore[attr-defined]
            gpu_memory_gb = round(props.total_memory / (1024 ** 3), 2)
            gpu_name = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
            logger.debug("GPU detected via PyTorch CUDA: %s (%.2fGB)", gpu_name, gpu_memory_gb)
        except Exception as err:
            logger.debug("Failed to read CUDA device properties: %s", err)
            gpu_memory_gb = 0.0
            gpu_name = None
    
    # Fallback to nvidia-smi if PyTorch unavailable or failed
    if gpu_memory_gb == 0.0:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",", 1)
                if len(parts) >= 1:
                    gpu_memory_mb = float(parts[0].strip())
                    gpu_memory_gb = round(gpu_memory_mb / 1024, 2)
                if len(parts) >= 2:
                    gpu_name = parts[1].strip()
                logger.debug("GPU detected via nvidia-smi: %s (%.2fGB)", gpu_name or "Unknown", gpu_memory_gb)
        except Exception as err:
            logger.debug("Failed to detect GPU via nvidia-smi: %s", err)

    # Determine tier using 2-tier system (standard/high)
    # Standard tier: <12GB GPU (optimized for 8GB GPUs)
    # High-performance tier: >=12GB GPU (enhanced depth)
    if gpu_memory_gb >= 12:
        tier = "high"
    else:
        tier = "standard"

    budgets_by_tier = {
        "high": {
            "chat_max_tokens": 2048,
            "reflection_max_tokens": 650,
            "analytical_max_tokens": 1536,  # Increased to prevent traits JSON truncation
            "allow_warmup": True,
        },
        "standard": {
            "chat_max_tokens": 1024,
            # qwen2.5:3b supports 8192 token context natively - use full capacity
            "reflection_max_tokens": 640,
            "analytical_max_tokens": 640,
            "allow_warmup": False,
        },
    }

    budgets = budgets_by_tier.get(tier, budgets_by_tier["standard"])

    profile = {
        "tier": tier,
        "total_ram_gb": total_ram_gb,
        "gpu_memory_gb": gpu_memory_gb or None,
        "gpu_name": gpu_name,
        "budgets": {
            "chat_max_tokens": budgets["chat_max_tokens"],
            "reflection_max_tokens": budgets["reflection_max_tokens"],
        },
        "features": {
            "allow_warmup": bool(budgets["allow_warmup"]),
            "allow_keepalive": bool(budgets["allow_warmup"]),
        },
    }

    logger.info(
        "System profile detected: tier=%s, RAM=%.2fGB, GPU=%s (%.2fGB), chat_max=%d, reflection_max=%d",
        profile["tier"],
        profile["total_ram_gb"],
        profile["gpu_name"] or "n/a",
        profile["gpu_memory_gb"] or 0.0,
        profile["budgets"]["chat_max_tokens"],
        profile["budgets"]["reflection_max_tokens"],
    )

    return profile
