from __future__ import annotations

import os
import warnings
from pathlib import Path


def configure_runtime(repo_root: str | Path) -> Path:
    repo_root = Path(repo_root)
    mpl_dir = repo_root / ".cache" / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "none")
    warnings.filterwarnings("ignore")
    return mpl_dir
