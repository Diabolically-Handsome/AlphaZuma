"""Audit installed Zuma's Revenge level data without modifying game files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from zuma_rl.original_data import OriginalGameCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--level", default="Jungle1")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = OriginalGameCatalog(args.root)
    loaded = catalog.load_level(args.level, hard=args.hard)
    result = {
        "installation": str(catalog.root),
        "level": loaded.definition.id,
        "display_name": loaded.definition.display_name,
        "hard": loaded.hard,
        "gun": {
            "type": loaded.definition.gun.type,
            "positions": loaded.definition.gun.positions,
        },
        "tunnel_layers": [asdict(value) for value in loaded.definition.tunnels],
        "curves": [
            {
                "path": str(curve.source_path),
                "samples": len(curve.points),
                "length_pixels": curve.total_length,
                "start": curve.points[0].tolist(),
                "end": curve.points[-1].tolist(),
                "bounds": {
                    "min": curve.points.min(axis=0).tolist(),
                    "max": curve.points.max(axis=0).tolist(),
                },
                "tunnel_samples": int(curve.in_tunnel.sum()),
                "priorities": sorted(set(map(int, curve.priorities))),
                "absolute_anchors": int(curve.absolute_anchors.sum()),
                "flags": sorted(set(map(int, curve.point_flags))),
                "parameters": asdict(curve.parameters),
            }
            for curve in loaded.curves
        ],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"installation : {result['installation']}")
    print(
        f"level        : {result['level']} / {result['display_name']} "
        f"(hard={result['hard']})"
    )
    print(
        f"gun          : {result['gun']['type']} "
        f"{result['gun']['positions']}"
    )
    for index, curve in enumerate(result["curves"], start=1):
        print(
            f"curve {index:<2}     : {curve['samples']:,} samples, "
            f"{curve['length_pixels']:,.2f}px, "
            f"tunnel_samples={curve['tunnel_samples']}, "
            f"priorities={curve['priorities']}, "
            f"anchors={curve['absolute_anchors']}"
        )
        print(f"parameters   : {curve['parameters']}")


if __name__ == "__main__":
    main()
