from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _REPO_ROOT / "src"

if __name__ != "__main__":
    __path__ = [str(_SRC_PATH / "inspection_app")]


def main() -> int:
    sys.path.insert(0, str(_SRC_PATH))

    from inspection_app.app import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
