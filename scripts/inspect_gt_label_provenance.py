"""Inspect where the GT minutiae labels in a ground-truth root come from.

For a few acquisitions, prints per view: the minutiae count, the `source` field
distribution inside minutiae.json, and the meta.json `minutiae_ground_truth`
config. This answers whether same-acquisition views share a cross-view
consensus label set (reprojected per view) or carry independent per-view
extractions — which decides whether a consensus GT rebuild can help.

Example:
  python scripts/inspect_gt_label_provenance.py \
    --ground-truth-root ground_truth/merged_DS123 --acquisitions 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--acquisitions", type=int, default=3, help="How many acquisitions to inspect.")
    args = parser.parse_args()

    samples_root = args.ground_truth_root / "samples"
    if not samples_root.exists():
        raise SystemExit(f"missing sample directory: {samples_root}")

    by_acquisition: dict[tuple, list[Path]] = defaultdict(list)
    for sample_dir in sorted(p for p in samples_root.iterdir() if p.is_dir()):
        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            view = int(meta.get("raw_view_index", -1))
        except (TypeError, ValueError):
            continue
        if view not in {0, 1, 2}:
            continue
        key = (meta.get("dataset"), meta.get("subject_id"), meta.get("finger_id"), meta.get("acquisition_id"))
        by_acquisition[key].append(sample_dir)

    printed = 0
    gt_config_printed = False
    for key, dirs in by_acquisition.items():
        if len(dirs) < 2:
            continue
        if printed >= int(args.acquisitions):
            break
        printed += 1
        print(f"\n=== acquisition dataset={key[0]} subject={key[1]} finger={key[2]} acq={key[3]} ===")
        for sample_dir in sorted(dirs):
            meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
            view = meta.get("raw_view_index")
            minutiae_path = sample_dir / "minutiae.json"
            records = []
            if minutiae_path.exists():
                payload = json.loads(minutiae_path.read_text(encoding="utf-8"))
                records = payload if isinstance(payload, list) else []
            sources = Counter(str(item.get("source", "<none>")) for item in records)
            keys_seen = sorted({k for item in records[:5] for k in item.keys()})
            print(
                f"  view {view}: {len(records)} minutiae, sources={dict(sources)}, "
                f"record keys={keys_seen}"
            )
            if not gt_config_printed:
                gt_config = meta.get("minutiae_ground_truth")
                if isinstance(gt_config, dict):
                    interesting = {
                        k: gt_config.get(k)
                        for k in (
                            "source",
                            "extractor",
                            "consensus",
                            "consensus_overlap_radius_px",
                            "view_role",
                            "label_minutiae_path",
                            "single_source_candidates_path",
                        )
                        if k in gt_config
                    }
                    print(f"  [meta.minutiae_ground_truth of view {view}]:")
                    print("   ", json.dumps(interesting if interesting else gt_config, indent=2)[:2000])
                    gt_config_printed = True

    if printed == 0:
        raise SystemExit("no multi-view acquisitions found")


if __name__ == "__main__":
    main()
