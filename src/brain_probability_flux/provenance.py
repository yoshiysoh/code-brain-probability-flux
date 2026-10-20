"""Run directories and provenance attributes for the data-writing scripts."""

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy
from threadpoolctl import threadpool_info

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def run_dir(data_dir: Path = DATA_DIR) -> Path:
    """Today's run directory data/<YYYY-MM-DD>/, created if needed."""
    path = Path(data_dir) / datetime.date.today().isoformat()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def write_provenance(h5file: h5py.File, **parameters) -> None:
    """Store the run parameters and its provenance as file attributes."""
    for key, value in parameters.items():
        h5file.attrs[key] = value
    h5file.attrs["git_commit"] = _git_commit()
    h5file.attrs["python_version"] = platform.python_version()
    h5file.attrs["numpy_version"] = np.__version__
    h5file.attrs["scipy_version"] = scipy.__version__
    h5file.attrs["h5py_version"] = h5py.__version__
    h5file.attrs["created"] = datetime.datetime.now().isoformat(
        timespec="seconds"
    )
    h5file.attrs["command_line"] = " ".join(sys.argv)
    h5file.attrs["cpu_model"] = _cpu_model()
    h5file.attrs["threadpool_info"] = json.dumps(threadpool_info())
