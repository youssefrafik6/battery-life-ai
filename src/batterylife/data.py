"""Ingest official snapshot logs and construct independently auditable intervals."""

import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

URL = (
    "https://zenodo.org/records/13982080/files/"
    "Mobile%20devices%20consumption%20analysis.zip?download=1"
)
MD5 = "29531f11edb9767eaa9af797abd381f8"
DEVICES = (
    "OnePlus_LE2113",
    "OnePlus_LE2123",
    "POCO_M2102J20SG",
    "Samsung_SM-A226B",
    "Samsung_SM-G991B",
)
COLUMNS = {
    "Datetime": "time",
    "Battery level (%)": "soc",
    "Charge status": "charge",
    "Battery temperature (degrees Celsius)": "temperature",
    "Network connection": "network",
    "Total screen-on time (seconds)": "screen",
    "Total network traffic received (MB)": "download",
    "Total network traffic sent (MB)": "upload",
}
COUNTERS = ["screen", "download", "upload"]
NUMERIC = COUNTERS + ["temperature"]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_seconds: int = Field(300, gt=0)
    max_seconds: int = Field(7200, gt=0)
    cap: float = Field(30, gt=0)
    quantile: float = Field(0.99, gt=0.5, le=1)
    mobile: bool = True
    seed: int = Field(326, ge=0, lt=2**32)
    bootstrap: int = Field(40, ge=0, le=500)

    @model_validator(mode="after")
    def bounds(self):
        if self.max_seconds < self.min_seconds:
            raise ValueError("max_seconds must be at least min_seconds")
        return self


def verify(payload: bytes) -> None:
    """Check archive identity against the checksum published by Zenodo."""
    if hashlib.md5(payload, usedforsecurity=False).hexdigest() != MD5:
        raise ValueError("Zenodo archive checksum mismatch")


def download(path: Path) -> None:
    """Download and verify data before replacing the destination."""
    with urlopen(URL, timeout=90) as response:
        payload = response.read(10_000_000)
    verify(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def intervals(path: Path, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Keep identity-free interval diagnostics and break sequences at exclusions."""
    verify(path.read_bytes())
    all_rows, audit = [], {}
    with ZipFile(path) as archive:
        for device in DEVICES:
            with archive.open(f"datasets/{device}_en.csv") as member:
                raw = pd.read_csv(member, sep=";", decimal=",", usecols=list(COLUMNS))
            frame = raw.rename(columns=COLUMNS)
            frame["time"] = pd.to_datetime(frame.time, format="%d/%m/%Y %H:%M", errors="coerce")
            frame["soc"] = pd.to_numeric(frame.soc, errors="coerce")
            frame = frame.dropna(subset=["time"]).sort_values("time", kind="stable")
            duplicates = frame.time.duplicated(keep=False)
            frame.loc[duplicates | ~frame.soc.between(0, 100), "soc"] = np.nan
            for name in NUMERIC:
                frame[name] = pd.to_numeric(frame[name], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
            elapsed = frame.time.diff().dt.total_seconds()
            drop = -frame.soc.diff()
            counters = frame[COUNTERS].diff()
            reset = counters.lt(0).any(axis=1)
            flags = frame.charge.str.strip().str.lower()
            charging = flags.eq("yes") | flags.shift().eq("yes") | drop.lt(0)
            valid_status = flags.eq("no") & flags.shift().eq("no")
            rate = drop.div(elapsed.where(elapsed.gt(0))) * 3600
            anomaly = rate.gt(settings.cap) | counters.screen.gt(elapsed)
            reason = np.select(
                [
                    drop.isna() | elapsed.le(0),
                    charging,
                    ~valid_status,
                    reset,
                    elapsed.gt(settings.max_seconds),
                    elapsed.lt(settings.min_seconds),
                ],
                [
                    "Invalid boundary",
                    "Charging or gauge rise",
                    "Unknown charge status",
                    "Counter reset",
                    "Large gap",
                    "Short interval",
                ],
                default="Discharge",
            )
            derived = pd.DataFrame(
                {
                    "device": device,
                    "time": frame.time,
                    "begin": frame.time.shift(),
                    "hours": elapsed / 3600,
                    "start_soc": frame.soc.shift(),
                    "soc": frame.soc,
                    "rate": rate,
                    "reason": reason,
                    "anomaly": anomaly,
                    "sequence": device
                    + ":"
                    + pd.Series(reason != "Discharge", index=frame.index).cumsum().astype(str),
                    "network": frame.network.str.lower()
                    .str.strip()
                    .map({"wi-fi": "wifi", "mobile data": "cellular", "datos moviles": "cellular"}),
                    "temperature": frame.temperature.shift(),
                }
            )
            for name in COUNTERS:
                derived[name] = (
                    counters[name].div(elapsed.where(elapsed.gt(0))).where(counters[name].ge(0))
                )
            derived["screen"] = derived.screen.where(derived.screen.le(1))
            audit[device] = {
                "raw_rows": len(raw),
                "timestamp_rows": len(frame),
                "interval_types": derived.reason.value_counts().to_dict(),
                "flagged_high_drain": int(rate.gt(settings.cap).sum()),
                "observed_networks": frame.network.value_counts().to_dict(),
            }
            all_rows.append(derived)
    return pd.concat(all_rows, ignore_index=True), audit
