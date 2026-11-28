"""Build the reduced dashboard table from prepared local telemetry."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/intervals.csv"))
    parser.add_argument("--output", type=Path, default=Path("dashboard/artifacts/explorer.csv.gz"))
    args = parser.parse_args()
    columns = ["device", "reason", "soc", "rate", "download", "upload", "screen", "temperature"]
    frame = pd.read_csv(args.input, usecols=[*columns, "time"], parse_dates=["time"])
    start = frame.groupby("device")["time"].transform("min")
    frame["elapsed_hours"] = (frame["time"] - start).dt.total_seconds().div(3600)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[[*columns, "elapsed_hours"]].to_csv(
        args.output, index=False, compression={"method": "gzip", "mtime": 0}
    )


if __name__ == "__main__":
    main()
