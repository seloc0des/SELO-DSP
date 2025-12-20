import logging
import os
import subprocess
import time
from enum import Enum
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may not be installed
    psutil = None  # type: ignore

from .datetime import utc_iso
from .system_profile import detect_system_profile

logger = logging.getLogger("selo.system_metrics")


class DiagnosticMode(str, Enum):
    NONE = "none"
    EXPLICIT_STATUS = "explicit_status"
    PROBLEM_REPORTED = "problem_reported"


class SystemMetricsCollector:
    def __init__(self, cache_ttl_seconds: float = 10.0) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._last_system_status: Optional[Dict[str, Any]] = None
        self._last_gpu_status: Optional[Dict[str, Any]] = None
        self._last_timestamp: float = 0.0

    def get_system_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (
            not force_refresh
            and self._last_system_status is not None
            and (now - self._last_timestamp) < self.cache_ttl_seconds
        ):
            return self._last_system_status

        status: Dict[str, Any] = {}
        try:
            cpu_percent = None
            memory: Dict[str, Any] = {}
            disk: Dict[str, Any] = {}

            if psutil is not None:
                try:
                    cpu_percent = float(psutil.cpu_percent(interval=0.1))
                except Exception as err:
                    logger.debug("CPU percent collection failed: %s", err)

                try:
                    vm = psutil.virtual_memory()
                    memory = {
                        "total_gb": round(vm.total / (1024 ** 3), 2),
                        "used_gb": round(vm.used / (1024 ** 3), 2),
                        "percent": float(vm.percent),
                    }
                except Exception as err:
                    logger.debug("Memory collection failed: %s", err)

                try:
                    du = psutil.disk_usage(os.getcwd())
                    disk = {
                        "total_gb": round(du.total / (1024 ** 3), 2),
                        "used_gb": round(du.used / (1024 ** 3), 2),
                        "percent": float(du.percent),
                    }
                except Exception as err:
                    logger.debug("Disk usage collection failed: %s", err)

            try:
                import os as _os

                load_avg = list(_os.getloadavg())
            except Exception as err:
                logger.debug("Load average collection failed: %s", err)
                load_avg = None

            status = {
                "cpu": {"percent": cpu_percent},
                "memory": memory,
                "disk": disk,
                "load_avg": load_avg,
                "timestamp": utc_iso(),
            }
        except Exception as err:
            logger.debug("System status collection failed: %s", err)
            if not status:
                status = {"error": str(err), "timestamp": utc_iso()}

        self._last_system_status = status
        self._last_timestamp = now
        return status

    def get_gpu_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (
            not force_refresh
            and self._last_gpu_status is not None
            and (now - self._last_timestamp) < self.cache_ttl_seconds
        ):
            return self._last_gpu_status

        profile = detect_system_profile()
        gpu_name = profile.get("gpu_name")
        total_gb = profile.get("gpu_memory_gb")

        gpu_status: Dict[str, Any] = {
            "available": False,
            "name": gpu_name,
            "memory_total_gb": total_gb,
            "memory_used_gb": None,
            "memory_percent": None,
            "utilization_percent": None,
            "temperature_c": None,
        }

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]

                used_mb = float(parts[0]) if len(parts) > 0 and parts[0] else 0.0
                total_mb = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                util = float(parts[2]) if len(parts) > 2 and parts[2] else None
                temp = float(parts[3]) if len(parts) > 3 and parts[3] else None

                used_gb = round(used_mb / 1024, 2)
                total_gb_runtime = round(total_mb / 1024, 2) if total_mb else total_gb

                percent = None
                if total_gb_runtime and total_gb_runtime > 0:
                    try:
                        percent = round(used_gb / float(total_gb_runtime) * 100.0, 2)
                    except Exception as err:
                        logger.debug("GPU percent calculation failed: %s", err)

                gpu_status.update(
                    {
                        "available": True,
                        "memory_total_gb": total_gb_runtime,
                        "memory_used_gb": used_gb,
                        "memory_percent": percent,
                        "utilization_percent": util,
                        "temperature_c": temp,
                    }
                )
        except Exception as err:
            logger.debug("GPU status collection failed: %s", err)

        self._last_gpu_status = gpu_status
        self._last_timestamp = now
        return gpu_status


def detect_diagnostic_trigger(message: Optional[str]) -> DiagnosticMode:
    if not message:
        return DiagnosticMode.NONE

    text = message.strip().lower()
    if not text:
        return DiagnosticMode.NONE

    explicit_commands = [
        "/status",
        "/gpu",
        "/perf",
        "/resources",
    ]
    for cmd in explicit_commands:
        if text.startswith(cmd):
            return DiagnosticMode.EXPLICIT_STATUS

    explicit_keywords = [
        "system status",
        "system health",
        "performance status",
        "cpu usage",
        "gpu usage",
        "vram",
        "ram usage",
        "resource usage",
        "load average",
        "system metrics",
    ]
    for kw in explicit_keywords:
        if kw in text:
            return DiagnosticMode.EXPLICIT_STATUS

    problem_keywords = [
        "slow",
        "laggy",
        "lagging",
        "hanging",
        "freezing",
        "unresponsive",
        "taking forever",
        "too long",
        "overheating",
        "throttling",
        "fans going crazy",
        "fan going crazy",
        "fan is loud",
        "crash",
        "crashed",
        "keeps crashing",
        "error",
        "stack trace",
        "timing out",
        "timeout",
        "out of memory",
        "oom",
        "cuda error",
        "gpu error",
    ]
    for kw in problem_keywords:
        if kw in text:
            return DiagnosticMode.PROBLEM_REPORTED

    return DiagnosticMode.NONE


_global_collector: Optional[SystemMetricsCollector] = None


def get_system_metrics_collector() -> SystemMetricsCollector:
    global _global_collector
    if _global_collector is None:
        _global_collector = SystemMetricsCollector()
    return _global_collector
