"""Matplotlib runtime setup for sandboxed/local executions."""
from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_cache() -> None:
    """Point Matplotlib/fontconfig caches at writable temp directories."""
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    mpl_dir = tmp_root / "mecdonate-matplotlib"
    xdg_dir = tmp_root / "mecdonate-xdg-cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
