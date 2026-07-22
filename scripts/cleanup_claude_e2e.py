"""Remove only a verified disposable E2E fixture directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=Path)
    test_root = parser.parse_args().test_root.expanduser().resolve()
    manifest_file = test_root / "manifest.json"
    if not manifest_file.is_file() or not test_root.name.startswith("taskchamber-e2e-"):
        raise SystemExit("refusing to remove a directory that is not an E2E fixture")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_root_value = manifest.get("test_root")
    if not isinstance(manifest_root_value, str):
        raise SystemExit("refusing to remove a fixture with an invalid manifest")
    manifest_root = Path(manifest_root_value).expanduser().resolve()
    if manifest.get("harness_version") not in {2, 3} or manifest_root != test_root:
        raise SystemExit("refusing to remove a fixture with an invalid manifest")
    shutil.rmtree(test_root)
    print(f"Removed {test_root}")


if __name__ == "__main__":
    main()
