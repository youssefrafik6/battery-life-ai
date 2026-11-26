"""FastAPI interface for the shared BatteryLife AI engine."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from batterylife.engine import Batch, Engine, Prediction, Schedule

app = FastAPI(title="BatteryLife AI", version="0.1.0")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    try:
        return Engine.load()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Train the model artifact before serving predictions.") from exc


@app.get("/health")
def health():
    try:
        get_engine()
    except HTTPException:
        return {"status": "untrained", "model_ready": False}
    return {"status": "ok", "model_ready": True}


@app.get("/model-info")
def model_info(engine: Annotated[Engine, Depends(get_engine)]):
    return {
        "model": engine.model.name,
        "devices": engine.devices,
        "bounds": engine.model.bounds,
        "drain_cap_pph": engine.model.cap,
    }


@app.post("/predict")
def predict(request: Prediction, engine: Annotated[Engine, Depends(get_engine)]):
    return engine.predict(request)


@app.post("/simulate")
def simulate(request: Schedule, engine: Annotated[Engine, Depends(get_engine)]):
    return engine.simulate(request)


@app.post("/batch-predict")
async def batch_predict(request: Batch, engine: Annotated[Engine, Depends(get_engine)]):
    # Limit concurrent work within a request to avoid oversubscribing the estimators.
    return await run_in_threadpool(
        lambda: {"predictions": [engine.predict(item) for item in request.items]}
    )


@app.get("/evaluation")
def evaluation():
    try:
        return json.loads(Path("results/evaluation.json").read_text())
    except FileNotFoundError as exc:
        raise HTTPException(503, "Evaluation report is unavailable.") from exc
