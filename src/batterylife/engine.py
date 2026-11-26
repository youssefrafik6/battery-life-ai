"""Shared model inference, scenario uncertainty, and input contracts."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .simulation import percentile_summary, time_to_empty, trajectory

PRESETS = {
    "Idle": {
        "screen": 0.005,
        "download": 0.0005,
        "upload": 0.0002,
        "temperature": 29,
        "network": "wifi",
    },
    "Browsing": {
        "screen": 0.10,
        "download": 0.025,
        "upload": 0.003,
        "temperature": 32,
        "network": "wifi",
    },
    "Streaming (Wi-Fi)": {
        "screen": 0.20,
        "download": 0.10,
        "upload": 0.006,
        "temperature": 34,
        "network": "wifi",
    },
    "Streaming (cellular)": {
        "screen": 0.20,
        "download": 0.10,
        "upload": 0.006,
        "temperature": 34,
        "network": "cellular",
    },
}


class Input(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class Usage(Input):
    """Interval-average usage represented in the original feature space."""

    device: str = Field(min_length=1, max_length=80)
    screen: float = Field(ge=0, le=1)
    download: float = Field(ge=0, le=10000)
    upload: float = Field(ge=0, le=10000)
    temperature: float = Field(ge=-30, le=100)
    network: str = Field(pattern="^(wifi|cellular)$")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump(include=set(Usage.model_fields))])


class Prediction(Usage):
    start_soc: float = Field(default=100, ge=0, le=100)


class Segment(Usage):
    minutes: float = Field(gt=0, le=1440)


class Schedule(Input):
    start_soc: float = Field(default=100, ge=0, le=100)
    segments: list[Segment] = Field(min_length=1, max_length=50)


class Batch(Input):
    items: list[Prediction] = Field(min_length=1, max_length=100)


class Engine:
    """Use only artifacts generated locally by the trusted training pipeline."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.devices = bundle["devices"]

    @classmethod
    def load(cls, path: Path | None = None):
        return cls(
            joblib.load(path or Path(os.environ.get("BATTERYLIFE_MODEL", "results/model.joblib")))
        )

    def warnings(self, usage: Usage) -> list[str]:
        warnings = []
        if usage.device not in self.devices:
            warnings.append("Unseen device uses the learned unknown-device fallback.")
        for name, (low, high) in self.model.bounds.items():
            value = getattr(usage, name)
            if value < low or value > high:
                warnings.append(f"{name} is outside the training support [{low:.4g}, {high:.4g}].")
        return warnings

    def predict(self, request: Prediction) -> dict:
        frame = request.frame()
        rate = float(self.model.predict(frame)[0])
        residuals = self.bundle["residuals"]
        rng = np.random.default_rng(self.bundle["seed"])
        sampled = rng.choice(residuals.get(request.network, residuals["all"]), 3000)
        draws = np.clip(rate + sampled, 0, self.model.cap)
        start = request.start_soc
        tte = (
            np.zeros_like(draws)
            if not start
            else np.divide(start, draws, out=np.full_like(draws, np.inf), where=draws > 0)
        )
        bootstrap = [model.predict(frame)[0] for model in self.bundle["bootstrap"]]
        empty = time_to_empty(start, rate)
        horizon = min(empty if empty is not None and empty > 0 else 24, 72)
        path = trajectory(start, [horizon], [rate], include_crossing=True)
        if empty is not None and empty <= horizon:
            path["empty_hours"] = empty
        return {
            "rate_pph": rate,
            "empty_hours": empty,
            "trajectory": path,
            "drain_interval_pph": percentile_summary(draws),
            "tte_interval_hours": percentile_summary(tte),
            "bootstrap_rate_pph": percentile_summary(bootstrap) if bootstrap else None,
            "bootstrap_draws": len(bootstrap),
            "zero_drain_probability": float(np.mean(draws == 0)),
            "warnings": self.warnings(request),
            "explanation": self.model.explain(frame),
        }

    def simulate(self, request: Schedule) -> dict:
        rows = pd.concat([item.frame() for item in request.segments], ignore_index=True)
        rates = self.model.predict(rows)
        hours = np.array([item.minutes / 60 for item in request.segments])
        path = trajectory(request.start_soc, hours, rates, include_crossing=True)
        residuals = self.bundle["residuals"]
        rng = np.random.default_rng(self.bundle["seed"])
        # Use a shared percentile across all segments as a persistent-error scenario.
        levels = rng.uniform(0, 1, 3000)
        noise = np.column_stack(
            [
                np.quantile(residuals.get(item.network, residuals["all"]), levels)
                for item in request.segments
            ]
        )
        sampled_rates = np.clip(rates + noise, 0, self.model.cap)
        final = np.maximum(0, request.start_soc - sampled_rates @ hours)
        return path | {
            "rates_pph": rates.tolist(),
            "final_soc_interval": percentile_summary(final),
            "warnings": sorted({w for item in request.segments for w in self.warnings(item)}),
        }
