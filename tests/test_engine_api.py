import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_engine
from batterylife.data import Settings, verify
from batterylife.engine import Engine, Prediction
from batterylife.training import fit, grouped_predictions, metrics

PAYLOAD = {
    "device": "A",
    "screen": 0.4,
    "download": 0.02,
    "upload": 0.01,
    "network": "wifi",
    "temperature": 32,
    "start_soc": 75,
}


@pytest.fixture()
def observations():
    rows = []
    for device, shift in [("A", 0), ("B", 2), ("C", 4)]:
        for step in range(8):
            rows.append(
                {
                    "device": device,
                    "screen": step / 10,
                    "download": step / 100,
                    "upload": step / 200,
                    "temperature": 30 + step,
                    "network": "wifi",
                    "rate": 3 + shift + step,
                    "sequence": device,
                    "hours": 0.25,
                    "start_soc": 80 - step,
                    "soc": 79 - step,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def engine(observations):
    model = fit("Ridge", 1, observations, Settings())
    return Engine(
        {
            "model": model,
            "devices": ["A", "B", "C"],
            "seed": 1,
            "residuals": {"all": np.array([-1, 0, 1])},
            "bootstrap": [model],
        }
    )


def test_api_endpoints_and_input_validation(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        client = TestClient(app)
        predict = client.post("/predict", json=PAYLOAD)
        assert predict.status_code == 200, predict.text
        assert predict.json() == engine.predict(Prediction(**PAYLOAD))
        assert client.post("/batch-predict", json={"items": [PAYLOAD]}).json() == {
            "predictions": [predict.json()]
        }
        segment = {key: value for key, value in PAYLOAD.items() if key != "start_soc"}
        response = client.post(
            "/simulate",
            json={
                "start_soc": 75,
                "segments": [segment | {"minutes": 60}],
            },
        )
        assert response.status_code == 200
        assert response.json()["soc"][-1] == pytest.approx(75 - predict.json()["rate_pph"])
        assert client.get("/model-info").json()["model"] == "Ridge"
        assert client.get("/openapi.json").status_code == 200
        assert client.post("/predict", json=PAYLOAD | {"screen": 1.1}).status_code == 422
        assert client.post("/simulate", json={"segments": []}).status_code == 422
        assert client.post("/batch-predict", json={"items": []}).status_code == 422
        assert client.post("/predict", json=PAYLOAD | {"device": "unknown"}).json()["warnings"]
    finally:
        app.dependency_overrides.clear()


def test_untrained_api(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BATTERYLIFE_MODEL", str(tmp_path / "absent"))
    get_engine.cache_clear()
    client = TestClient(app)
    assert client.get("/health").json()["model_ready"] is False
    assert client.post("/predict", json=PAYLOAD).status_code == 503
    assert client.get("/evaluation").status_code == 503


def test_grouped_mean_excludes_entire_device(observations):
    predictions = grouped_predictions(observations, "Mean", 0, Settings())
    assert np.allclose(predictions[:8], observations.loc[8:, "rate"].mean())


def test_raw_outlier_counts_in_evaluation(observations):
    sample = observations.iloc[:1].assign(rate=40)
    output = metrics(sample, np.array([30]))
    assert output["mae"] == 10 and output["r2"] is None


def test_integrity_and_configuration():
    with pytest.raises(ValueError, match="checksum"):
        verify(b"wrong")
    with pytest.raises(ValueError):
        Settings(min_seconds=8000)
