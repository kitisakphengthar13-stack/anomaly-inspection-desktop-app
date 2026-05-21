from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))

    from inspection.main import main as inspection_main

    return inspection_main()


if __name__ == "__main__":
    raise SystemExit(main())
