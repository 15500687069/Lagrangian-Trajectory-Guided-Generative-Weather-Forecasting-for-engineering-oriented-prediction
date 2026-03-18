from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.cli import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ERA5 stats and skeleton tracks")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    prepare(config=args.config)


if __name__ == "__main__":
    main()
