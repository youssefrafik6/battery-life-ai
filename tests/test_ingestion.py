from zipfile import ZipFile

import pandas as pd
import pytest

from batterylife import data


def test_counter_intervals_and_sequence_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "verify", lambda payload: None)
    monkeypatch.setattr(data, "DEVICES", ("Test_device",))
    archive = tmp_path / "fixture.zip"
    raw = pd.DataFrame(
        [
            ["01/05/2024 00:00", 90, "no", 32.5, "wi-fi", 0, 0, 0],
            ["01/05/2024 00:10", 89, "no", 33.5, "mobile data", 60, 6, 3],
            ["01/05/2024 00:20", 88, "no", 34.5, "datos moviles", 90, 9, 6],
            ["01/05/2024 00:30", 87, "no", 35.5, "wi-fi", 0, 0, 0],
            ["01/05/2024 00:40", 89, "no", 35.5, "wi-fi", 0, 0, 0],
            ["01/05/2024 04:40", 88, "no", 35.5, "wi-fi", 30, 3, 0],
            ["01/05/2024 04:50", 87, "no", 35.5, "wi-fi", 40, 4, 0],
        ],
        columns=list(data.COLUMNS),
    )
    with ZipFile(archive, "w") as stream:
        stream.writestr(
            "datasets/Test_device_en.csv", raw.to_csv(index=False, sep=";", decimal=",")
        )

    rows, audit = data.intervals(archive, data.Settings())
    assert rows.reason.tolist() == [
        "Invalid boundary",
        "Discharge",
        "Discharge",
        "Counter reset",
        "Charging or gauge rise",
        "Large gap",
        "Discharge",
    ]
    assert rows.loc[1:2, "network"].tolist() == ["cellular", "cellular"]
    assert rows.loc[1, "rate"] == pytest.approx(6)
    assert rows.loc[1, "screen"] == pytest.approx(0.1)
    assert rows.loc[1, "download"] == pytest.approx(0.01)
    assert rows.loc[1, "temperature"] == pytest.approx(32.5)
    assert rows.loc[1, "sequence"] == rows.loc[2, "sequence"]
    assert rows.loc[6, "sequence"] != rows.loc[2, "sequence"]
    assert audit["Test_device"]["interval_types"]["Discharge"] == 3
