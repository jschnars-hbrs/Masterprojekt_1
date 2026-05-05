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


def parse_avg_err_lines(stdout_text, scene_name, cam_name):
    """Extract AVG_ERR marker lines from subprocess stdout."""
    rows = []
    for line in stdout_text.splitlines():
        if not line.startswith("AVG_ERR|"):
            continue
        parts = line.split("|")
        # AVG_ERR|approach|gt|n_valid|n_total|mean|abs_mean|std|min|max
        if len(parts) != 10:
            continue
        rows.append({
            "camera": cam_name,
            "scene": scene_name,
            "approach": int(parts[1]),
            "gt_m": float(parts[2]),
            "n_valid": int(parts[3]),
            "n_total": int(parts[4]),
            "mean_m": float(parts[5]),
            "abs_mean_m": float(parts[6]),
            "std_m": float(parts[7]),
            "min_m": float(parts[8]),
            "max_m": float(parts[9]),
        })
    return rows


def print_summary_table(all_errors):
    """Print a formatted summary table of average errors."""
    if not all_errors:
        return

    print(f"\n{'=' * 100}")
    print("  AVERAGE ERROR SUMMARY")
    print(f"{'=' * 100}")
    header = (f"{'Camera':<12} {'Scene':<25} {'Appr':>4} {'GT [m]':>7} "
              f"{'Mean [m]':>10} {'|Mean| [m]':>10} {'Std [m]':>9} "
              f"{'Min [m]':>9} {'Max [m]':>9} {'Valid':>7}")
    print(header)
    print("-" * 100)
    for r in all_errors:
        print(f"{r['camera']:<12} {r['scene']:<25} {r['approach']:>4} {r['gt_m']:>7.2f} "
              f"{r['mean_m']:>+10.4f} {r['abs_mean_m']:>10.4f} {r['std_m']:>9.4f} "
              f"{r['min_m']:>+9.4f} {r['max_m']:>+9.4f} "
              f"{r['n_valid']:>3}/{r['n_total']:<3}")
    print("-" * 100)


def save_summary_csv(all_errors, output_path):
    """Save summary table as CSV."""
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "camera", "scene", "approach", "gt_m",
            "mean_m", "abs_mean_m", "std_m", "min_m", "max_m",
            "n_valid", "n_total",
        ])
        writer.writeheader()
        writer.writerows(all_errors)
    print(f"\nSummary CSV saved to {output_path}")


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
    all_errors = []

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

            ret = subprocess.run(cmd, capture_output=True, text=True)
            # Print the original output so it's still visible
            if ret.stdout:
                print(ret.stdout, end="")
            if ret.stderr:
                print(ret.stderr, end="", file=sys.stderr)

            if ret.returncode != 0:
                failed += 1
                print(f"[FAIL] {result_name} (exit code {ret.returncode})")
            else:
                all_errors.extend(parse_avg_err_lines(ret.stdout, scene, cam_name))

    if args.dry_run:
        print(f"Would run {total} scenario(s).")
    else:
        print(f"\nDone: {total} scenario(s), {failed} failed.")
        if all_errors:
            print_summary_table(all_errors)
            csv_path = ROOT / "Results" / "average_error_summary.csv"
            save_summary_csv(all_errors, csv_path)


if __name__ == "__main__":
    main()
