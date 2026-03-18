from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.cli import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LTG-Net checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--inference_mode",
        default="single",
        choices=["single", "field", "track", "fused"],
    )
    parser.add_argument("--field_checkpoint", default="")
    parser.add_argument("--track_checkpoint", default="")
    parser.add_argument("--fused_encoder_source", default="field", choices=["field", "track"])
    parser.add_argument("--save_fused_checkpoint", default="")
    args = parser.parse_args()
    evaluate(
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        inference_mode=args.inference_mode,
        field_checkpoint=args.field_checkpoint,
        track_checkpoint=args.track_checkpoint,
        fused_encoder_source=args.fused_encoder_source,
        save_fused_checkpoint=args.save_fused_checkpoint,
    )


if __name__ == "__main__":
    main()
