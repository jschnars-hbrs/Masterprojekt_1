#!/usr/bin/env python3
"""Batch-run approaches.py over all test scenes for all cameras.

Usage:
    python run_all.py                              # run everything
    python run_all.py --dry-run                    # just print commands
    python run_all.py --cameras Schmersal          # one camera only
    python run_all.py --scenes Flat_Wall           # filter scenes by substring
    python run_all.py --approaches 1 3             # only certain approaches
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CAMERAS = {
    "OnSemi": {
        "calibration": ROOT / "Calibrations" / "OnSemi" / "calibration.json",
        "test_dir": ROOT / "Simulation_Pictures" / "PBRT" / "SL_ToF_On" / "Test_Scenes",
        "suffix": "_On",
    },
    "Schmersal": {
        "calibration": ROOT / "Calibrations" / "Schmersal" / "calibration.json",
        "test_dir": ROOT / "Simulation_Pictures" / "PBRT" / "SL_ToF_Schm" / "Test_Scenes",
        "suffix": "_Schm",
    },
}


def discover_pairs(test_dir: Path, suffix: str):
    """Yield (scene_name, sl_path, tof_path) for each matched SL/ToF pair."""
    for sl in sorted(test_dir.glob("SL_*.exr")):
        tof = sl.with_name(sl.name.replace("SL_", "ToF_", 1).replace(".exr", ".pcd"))
        if not tof.exists():
            continue
        # Extract scene name: strip "SL_" prefix and suffix + ".exr"
        # e.g. "SL_Flat_Wall_1.0m_On.exr" -> "Flat_Wall_1.0m"
        scene = sl.stem  # "SL_Flat_Wall_1.0m_On"
        scene = scene[3:]  # "Flat_Wall_1.0m_On"
        if scene.endswith(suffix):
            scene = scene[: -len(suffix)]  # "Flat_Wall_1.0m"
        yield scene, sl, tof


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cameras", nargs="+", choices=list(CAMERAS), default=list(CAMERAS),
                   help="Which cameras to run (default: all)")
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Substring filters for scene names (e.g. Flat_Wall Sp)")
    p.add_argument("--approaches", nargs="+", default=["all"],
                   help="Approaches to pass through (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--no-save", action="store_true",
                   help="Don't pass --save (show plots interactively instead)")
    args = p.parse_args()

    approaches_py = Path(__file__).resolve().parent / "approaches.py"
    total, skipped, failed = 0, 0, 0

    for cam_name in args.cameras:
        cam = CAMERAS[cam_name]
        cal_path = cam["calibration"]

        if not cal_path.exists():
            print(f"[SKIP] {cam_name}: calibration not found at {cal_path}")
            continue

        for scene, sl, tof in discover_pairs(cam["test_dir"], cam["suffix"]):
            if args.scenes and not any(f in scene for f in args.scenes):
                continue

            result_name = f"{cam_name}/{scene}"
            cmd = [
                sys.executable, str(approaches_py),
                "--calibration", str(cal_path),
                "--sl", str(sl),
                "--tof", str(tof),
                "--approaches", *args.approaches,
            ]
            if not args.no_save:
                cmd += ["--save", "--name", result_name]

            total += 1

            if args.dry_run:
                print(f"[{total}] {result_name}")
                print(f"    {' '.join(cmd)}\n")
                continue

            print(f"\n{'#' * 70}")
            print(f"  [{total}] {result_name}")
            print(f"{'#' * 70}\n")

            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                failed += 1
                print(f"[FAIL] {result_name} (exit code {ret.returncode})")

    if args.dry_run:
        print(f"Would run {total} scenario(s).")
    else:
        print(f"\nDone: {total} scenario(s), {failed} failed.")


if __name__ == "__main__":
    main()
