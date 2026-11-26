"""Grouped model selection, held-out evaluation, and deployment refit."""

from dataclasses import dataclass
from importlib.util import find_spec
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NUMERIC, Settings
from .simulation import trajectory


@dataclass
class DrainModel:
    """Fitted transformer and estimator with training-only feature bounds."""

    name: str
    columns: list
    bounds: dict
    estimator: object
    cap: float

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[self.columns].copy()
        for name, (_, upper) in self.bounds.items():
            result[name] = result[name].clip(upper=upper)
        return result

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.estimator.predict(self.features(frame))
        if self.name == "Ridge":
            scores = np.expm1(np.minimum(scores, np.log1p(self.cap)))
        return np.clip(scores, 0, self.cap)

    def explain(self, frame: pd.DataFrame) -> dict:
        """Exact pre-link Ridge decomposition or one-feature reference contrasts."""
        if self.name == "Ridge":
            transform, ridge = self.estimator.steps
            transformed = transform[1].transform(self.features(frame))[0]
            names = transform[1].get_feature_names_out()
            return {
                "scale": "log(1 + drain)",
                "intercept": float(ridge[1].intercept_),
                "coefficients": dict(zip(names, map(float, ridge[1].coef_), strict=True)),
                "contributions": dict(
                    zip(names, map(float, transformed * ridge[1].coef_), strict=True)
                ),
            }
        current = self.predict(frame)[0]
        changes = {}
        reference = frame.copy()
        for column in self.columns:
            reference[column] = (
                self.bounds[column][0]
                if column in self.bounds
                else ("wifi" if column == "network" else "__unseen__")
            )
            changes[column] = float(current - self.predict(reference)[0])
            reference[column] = frame[column]
        return {"scale": "pp/hour versus feature reference", "contrasts": changes}


def fit(name: str, parameter: int, frame: pd.DataFrame, config: Settings) -> DrainModel:
    """Fit clipping, imputation, scaling, and encoding inside each training fold."""
    numeric = NUMERIC
    categorical = ["device"] + (["network"] if config.mobile else [])
    bounds = {
        c: tuple(map(float, frame[c].quantile([0, config.quantile]).fillna(0))) for c in numeric
    }
    transformer = ColumnTransformer(
        [
            (
                "usage",
                make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()),
                numeric,
            ),
            (
                "context",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    if name == "Mean":
        estimator = DummyRegressor()
    elif name == "Ridge":
        estimator = Ridge(alpha=parameter)
    elif name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=parameter, n_jobs=1, random_state=config.seed
        )
    elif name == "Histogram Boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=parameter,
            early_stopping=False,
            random_state=config.seed,
        )
    elif name == "XGBoost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=100, max_depth=parameter, n_jobs=1, random_state=config.seed
        )
    else:
        raise ValueError(f"Unknown model {name}")
    model = DrainModel(
        name, numeric + categorical, bounds, make_pipeline(transformer, estimator), config.cap
    )
    target = frame.rate.clip(upper=config.cap)
    model.estimator.fit(model.features(frame), np.log1p(target) if name == "Ridge" else target)
    return model


def grouped_predictions(
    frame: pd.DataFrame, name: str, parameter: int, config: Settings
) -> np.ndarray:
    """Generate validation predictions for devices excluded from every fit."""
    groups = frame.device.to_numpy()
    splits = min(len(set(groups)), 4)
    predictions = np.empty(len(frame))
    for train, val in GroupKFold(splits).split(frame, groups=groups):
        model = fit(name, parameter, frame.iloc[train], config)
        predictions[val] = model.predict(frame.iloc[val])
    return predictions


def tune(frame: pd.DataFrame, grid: dict, config: Settings) -> tuple:
    """Optimize equally weighted device MAE using grouped inner validation."""
    candidates = {}
    for name, parameters in grid.items():
        best = None
        for value in parameters:
            pred = grouped_predictions(frame, name, value, config)
            score = abs(frame.rate - pred).groupby(frame.device).mean().mean()
            if best is None or score < best[0]:
                best = (float(score), value, pred)
        candidates[name] = best
    return min(candidates, key=lambda name: candidates[name][0]), candidates


def residual_bands(frame: pd.DataFrame, residuals: np.ndarray) -> dict:
    """Pool errors by observed network when at least 30 records are available."""
    bands = {"all": np.asarray(residuals)}
    for name in ("wifi", "cellular"):
        mask = frame.network.eq(name).to_numpy()
        if mask.sum() >= 30:
            bands[name] = residuals[mask]
    return bands


def metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    """Score raw drain targets, contiguous trajectories, and observed zero readings."""
    frame = frame.assign(pred=pred)
    sequence_errors, tte_errors = [], []
    incomplete = 0
    for _, group in frame.groupby("sequence", sort=False):
        if len(group) < 2:
            continue
        path = trajectory(group.start_soc.iloc[0], group.hours, group.pred)
        sequence_errors.append(float(np.mean(abs(group.soc.to_numpy() - path["soc"][1:]))))
        if group.soc.iloc[-1] == 0:
            if path["empty_hours"] is None:
                incomplete += 1
            else:
                tte_errors.append(abs(path["empty_hours"] - path["hours"][-1]))
    return {
        "n": len(frame),
        "mae": float(mean_absolute_error(frame.rate, pred)),
        "rmse": float(root_mean_squared_error(frame.rate, pred)),
        "r2": float(r2_score(frame.rate, pred)) if frame.rate.nunique() > 1 else None,
        "sequence_count": len(sequence_errors),
        "soc_sequence_mae_values": sequence_errors,
        "tte_paired_errors_hours": tte_errors,
        "tte_censored_predictions": incomplete,
    }


def summarize(folds: list[dict]) -> dict:
    """Report macro device drain errors and pooled sequence errors."""
    sequences = [x for fold in folds for x in fold["soc_sequence_mae_values"]]
    tte = [x for fold in folds for x in fold["tte_paired_errors_hours"]]
    return {
        "mae": float(np.mean([f["mae"] for f in folds])),
        "rmse": float(np.mean([f["rmse"] for f in folds])),
        "soc_median_mae": float(np.median(sequences)) if sequences else None,
        "soc_p90_mae": float(np.quantile(sequences, 0.9)) if sequences else None,
        "sequences": len(sequences),
        "tte_paired_count": len(tte),
        "tte_mae_hours": float(np.mean(tte)) if tte else None,
        "tte_censored_predictions": sum(f["tte_censored_predictions"] for f in folds),
    }


def chronological(frame: pd.DataFrame, grid: dict, config: Settings) -> dict:
    """Fit early data, select on middle data, score late data with boundary purge."""
    early, middle, late = [], [], []
    for _, device in frame.groupby("device"):
        ordered = device.sort_values("time")
        start, stop = ordered.time.iloc[[int(len(ordered) * 0.6), int(len(ordered) * 0.8)]]
        early.append(ordered[ordered.time < start])
        middle.append(ordered[(ordered.begin >= start) & (ordered.time < stop)])
        late.append(ordered[ordered.begin >= stop])
    train, val, test = map(pd.concat, (early, middle, late))
    if min(map(len, (train, val, test))) == 0:
        return {}
    output = {}
    for name, parameters in grid.items():
        param = min(
            parameters,
            key=lambda p: mean_absolute_error(val.rate, fit(name, p, train, config).predict(val)),
        )
        model = fit(name, param, pd.concat([train, val]), config)
        output[name] = metrics(test, model.predict(test))
    return output


def train_models(frame: pd.DataFrame, config: Settings, xgb: bool) -> tuple[dict, dict]:
    """Run nested device validation before choosing a deployment model on all data."""
    grid = {
        "Mean": [0],
        "Ridge": [1, 10, 100],
        "Random Forest": [4, 16],
        "Histogram Boosting": [7, 15],
    }
    if xgb:
        if find_spec("xgboost") is None:
            raise RuntimeError("Install the xgb extra to enable XGBoost")
        grid["XGBoost"] = [2, 4]
    outer = {name: [] for name in [*grid, "Selection procedure"]}
    diagnostics = []
    for train, test in LeaveOneGroupOut().split(frame, groups=frame.device):
        training, testing = frame.iloc[train], frame.iloc[test]
        winner, tuned = tune(training, grid, config)
        band = residual_bands(training, training.rate.to_numpy() - tuned[winner][2])
        for name, (_, param, _) in tuned.items():
            started = perf_counter()
            model = fit(name, param, training, config)
            fitted_seconds = perf_counter() - started
            started = perf_counter()
            pred = model.predict(testing)
            prediction_seconds = perf_counter() - started
            row = metrics(testing, pred)
            row.update(
                device=testing.device.iloc[0],
                parameter=param,
                model=name,
                fit_seconds=fitted_seconds,
                predict_seconds=prediction_seconds,
            )
            outer[name].append(row)
            if name == winner:
                outer["Selection procedure"].append(row)
                residual = testing.rate.to_numpy() - pred
                for network in ["all", "wifi", "cellular", "unknown"]:
                    mask = (
                        np.ones(len(testing), dtype=bool)
                        if network == "all"
                        else testing.network.fillna("unknown").eq(network).to_numpy()
                    )
                    if not mask.any():
                        continue
                    errors = residual[mask]
                    low, high = np.quantile(band.get(network, band["all"]), [0.1, 0.9])
                    covered = testing.rate.to_numpy() >= np.clip(pred + low, 0, config.cap)
                    covered &= testing.rate.to_numpy() <= np.clip(pred + high, 0, config.cap)
                    diagnostics.append(
                        {
                            "device": testing.device.iloc[0],
                            "network": network,
                            "n": len(errors),
                            "bias_observed_minus_predicted": float(errors.mean()),
                            "error_std": float(errors.std()),
                            "coverage_80": float(covered[mask].mean())
                            if network != "all"
                            else None,
                        }
                    )
    winner, tuned = tune(frame, grid, config)
    final = fit(winner, tuned[winner][1], frame, config)
    residual = frame.rate.to_numpy() - tuned[winner][2]
    # Resample whole discharge sequences within each device, preserving within-sequence rows.
    groups = [
        [block for _, block in device.groupby("sequence")] for _, device in frame.groupby("device")
    ]
    rng = np.random.default_rng(config.seed)
    bootstrap = []
    for _ in range(config.bootstrap):
        sample = pd.concat(
            [blocks[i] for blocks in groups for i in rng.integers(len(blocks), size=len(blocks))]
        )
        bootstrap.append(fit(winner, tuned[winner][1], sample, config))
    importance = {}
    for name, (_, param, _) in tuned.items():
        if name == "Mean":
            continue
        # Permute inputs within held-out devices, then average importance across folds.
        values = []
        for train, test in LeaveOneGroupOut().split(frame, groups=frame.device):
            training, testing = frame.iloc[train], frame.iloc[test]
            fold = next(f for f in outer[name] if f["device"] == testing.device.iloc[0])
            model = fit(name, fold["parameter"], training, config)
            data = model.features(testing)
            result = permutation_importance(
                model.estimator,
                data,
                np.log1p(testing.rate) if name == "Ridge" else testing.rate,
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=config.seed,
            )
            values.append(result.importances_mean)
        columns = fit(name, param, frame, config).columns
        importance[name] = dict(zip(columns, np.mean(values, axis=0).tolist(), strict=True))
    report = {
        "settings": config.model_dump(),
        "intervals": len(frame),
        "winner": winner,
        "deployment_group_mae": {name: item[0] for name, item in tuned.items()},
        "deployment_parameters": {name: item[1] for name, item in tuned.items()},
        "outer_summary": {name: summarize(folds) for name, folds in outer.items()},
        "outer_folds": outer,
        "chronological": chronological(frame, grid, config),
        "residual_diagnostics": diagnostics,
        "permutation_importance": importance,
    }
    return {
        "model": final,
        "residuals": residual_bands(frame, residual),
        "bootstrap": bootstrap,
        "devices": sorted(frame.device.unique()),
        "seed": config.seed,
    }, report
