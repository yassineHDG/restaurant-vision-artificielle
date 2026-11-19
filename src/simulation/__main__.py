"""Run from the repository root: python -m src.simulation --days 90 --seed 42."""

import argparse
from datetime import date
import json
from pathlib import Path

from .config import SimulationConfig
from .dataset import write_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible synthetic restaurant dataset.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config" / "simulation.json")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2026, 11, 1))
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/simulation/baseline-90d"))
    args = parser.parse_args(argv)
    try:
        manifest = write_dataset(SimulationConfig.load(args.config), args.start_date,
                                 args.days, args.seed, args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Dataset generation failed: {exc}\n")
    print(json.dumps(dict(output=str(args.output.resolve()), dataset_id=manifest["dataset_id"],
                          split_days=manifest["split_days"], summary=manifest["summary"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
