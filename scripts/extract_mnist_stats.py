"""Summarise the first kernel-only gem5 statistics dump from run directories."""
import argparse
from pathlib import Path


def read_first_roi(path: Path) -> dict[str, float]:
    """Exclude the post-program gem5 statistics section explicitly."""
    lines = path.read_text().splitlines()
    inside = False
    found = 0
    values: dict[str, float] = {}
    for line in lines:
        if "Begin Simulation Statistics" in line:
            found += 1
            if found == 1:
                inside = True
            else:
                break
            continue
        if "End Simulation Statistics" in line and inside:
            break
        if inside:
            fields = line.split("#", 1)[0].split()
            if len(fields) >= 2:
                try:
                    values[fields[0]] = float(fields[1])
                except ValueError:
                    pass
    if not values:
        raise RuntimeError(f"No first ROI dump found in {path}")
    return values


def main() -> None:
    """Print minimal architectural counters for easy model-to-model comparison."""
    p = argparse.ArgumentParser()
    p.add_argument("folders", type=Path, nargs="+")
    args = p.parse_args()
    core = "board.processor.cores.core"
    metrics = {
        "cycles": f"{core}.numCycles",
        "instructions": "simInsts",
        "branch_mispredicts": f"{core}.commit.branchMispredicts",
        "conditional_predictions": f"{core}.branchPred.condPredicted",
        "conditional_incorrect": f"{core}.branchPred.condIncorrect",
        "l1d_misses": "board.cache_hierarchy.l1d-cache-0.overallMisses::total",
        "l1d_miss_rate": "board.cache_hierarchy.l1d-cache-0.overallMissRate::total",
        "l2_misses": "board.cache_hierarchy.l2-cache-0.overallMisses::total",
        "l2_data_misses": "board.cache_hierarchy.l2-cache-0.demandMisses::processor.cores.core.data",
        "lsq_full_events": f"{core}.iew.lsqFullEvents",
    }
    print("run," + ",".join(metrics))
    for folder in args.folders:
        path = folder / "stats.txt"
        values = read_first_roi(path)
        print(folder.name + "," + ",".join(
            "NA" if values.get(key) is None else f"{values[key]:.9g}"
            for key in metrics.values()
        ))


if __name__ == "__main__":
    main()
