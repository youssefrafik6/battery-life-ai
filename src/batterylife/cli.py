"""Reproducible data and experiment commands."""

import argparse
import json
import logging
from pathlib import Path

import joblib

from .data import Settings, download, intervals
from .training import fit, train_models

LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Download, prepare, train, or run the sensitivity study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare", "train", "sensitivity"])
    parser.add_argument("--archive", type=Path, default=Path("data/source.zip"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--xgb", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "download":
        download(args.archive)
        LOGGER.info("Verified official archive at %s", args.archive)
        return
    settings = Settings.model_validate_json(args.config.read_text()) if args.config else Settings()
    rows, audit = intervals(args.archive, settings)
    frame = rows[rows.reason.eq("Discharge")]
    args.output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("%s eligible intervals on %s devices", len(frame), frame.device.nunique())
    if args.command == "prepare":
        write_json(args.output / "data_audit.json", audit)
        rows.to_csv(args.output / "intervals.csv", index=False)
        return
    if args.command == "train":
        bundle, report = train_models(frame, settings, args.xgb)
        report["data_audit"] = audit
        artifact = args.output / "model.joblib"
        temporary = artifact.with_suffix(".part")
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(artifact)
        write_json(args.output / "evaluation.json", report)
        LOGGER.info("Deployment model: %s", report["winner"])
        return
    from sklearn.model_selection import LeaveOneGroupOut

    from .engine import PRESETS, Usage
    from .simulation import time_to_empty
    from .training import metrics, summarize

    variants = {
        "baseline": {},
        "min_600_seconds": {"min_seconds": 600},
        "cap_25": {"cap": 25},
        "winsor_95": {"quantile": 0.95},
        "without_network_type": {"mobile": False},
    }
    comparisons = {}
    for label, changes in variants.items():
        variant = Settings.model_validate(settings.model_dump() | changes)
        observations, _ = intervals(args.archive, variant)
        training_pool = observations[observations.reason.eq("Discharge")]
        folds = []
        # Evaluate against the same baseline population so changes in sample
        # eligibility cannot by themselves make the scores look better.
        for _, test in LeaveOneGroupOut().split(frame, groups=frame.device):
            held_out = frame.iloc[test]
            training = training_pool[training_pool.device.ne(held_out.device.iloc[0])]
            model = fit("Ridge", 10, training, variant)
            folds.append(metrics(held_out, model.predict(held_out)))
        final = fit("Ridge", 10, training_pool, variant)
        tte = {}
        for name, features in PRESETS.items():
            usage = Usage(device=frame.device.iloc[0], **features)
            rate = final.predict(usage.frame())[0]
            tte[name] = time_to_empty(100, rate)
        comparisons[label] = summarize(folds) | {
            "training_intervals": len(training_pool),
            "full_charge_tte_hours": tte,
        }
    write_json(args.output / "sensitivity.json", comparisons)


if __name__ == "__main__":
    main()
