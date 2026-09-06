import json
import numpy as np
import sys
from pathlib import Path

STILL_BASELINE = {"ax": 0, "ay": 0, "az": 222}
BASELINE_TOLERANCE = 80
MIN_SAMPLES = 6
MAX_CLIP_VALUE = 500
MIN_MOTION_STD = 10

def validate(filepath):
    filepath = Path(filepath)
    issues = []
    warnings = []

    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Cannot parse JSON: {e}"], [], {}

    sensors = data.get("sensors", [])
    tag = data.get("tag", "unknown")

    if not sensors:
        return False, ["No sensor data found"], [], {}

    window = np.array([[s['ax'], s['ay'], s['az']] for s in sensors])

    if len(window) < MIN_SAMPLES:
        issues.append(f"Too few samples: {len(window)} (need at least {MIN_SAMPLES})")

    for i, axis_name in enumerate(['ax', 'ay', 'az']):
        if np.any(np.abs(window[:, i]) >= MAX_CLIP_VALUE):
            issues.append(f"Sensor clipping on {axis_name} — swing less aggressively")

    if tag != "still":
        mag = np.sqrt(np.sum(window**2, axis=1))
        if np.std(mag) < MIN_MOTION_STD:
            issues.append(f"Not enough motion (std={np.std(mag):.1f}) — make a bigger gesture")

    dupes = sum(1 for i in range(1, len(window)) if np.array_equal(window[i], window[i-1]))
    if dupes > len(window) * 0.3:
        warnings.append(f"Many duplicate samples ({dupes}/{len(window)})")

    mag = np.sqrt(np.sum(window**2, axis=1))
    stats = {
        "samples": len(window),
        "tag": tag,
        "ax": f"{np.mean(window[:,0]):.0f} ± {np.std(window[:,0]):.0f}",
        "ay": f"{np.mean(window[:,1]):.0f} ± {np.std(window[:,1]):.0f}",
        "az": f"{np.mean(window[:,2]):.0f} ± {np.std(window[:,2]):.0f}",
        "magnitude": f"{np.mean(mag):.0f} ± {np.std(mag):.0f} (peak: {np.max(mag):.0f})",
    }

    return len(issues) == 0, issues, warnings, stats


def validate_folder(folder):
    folder = Path(folder)
    all_files = list(folder.rglob("*.json"))
    if not all_files:
        print(f"No JSON files found in {folder}")
        return

    passed, failed = 0, 0
    for f in sorted(all_files):
        ok, issues, warnings, stats = validate(f)
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"\n{status}  {f.relative_to(folder)}")
        print(f"       samples={stats['samples']}  tag={stats['tag']}")
        print(f"       ax={stats['ax']}  ay={stats['ay']}  az={stats['az']}")
        print(f"       magnitude={stats['magnitude']}")
        for issue in issues:
            print(f"       ⚠️  {issue}")
        for warning in warnings:
            print(f"       💡 {warning}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'─'*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(all_files)} recordings")
    if failed > 0:
        print("Delete or re-record the failed files before training.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 validate_recording.py data/")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        validate_folder(target)
    else:
        ok, issues, warnings, stats = validate(target)
        print(f"\n{'✅ PASS' if ok else '❌ FAIL'}  {target.name}")
        print(f"  samples={stats['samples']}  tag={stats['tag']}")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        for warning in warnings:
            print(f"  💡 {warning}")
