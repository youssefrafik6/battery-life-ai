from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_overview_and_research():
    path = Path(__file__).resolve().parents[1] / "dashboard/app.py"
    dashboard = AppTest.from_file(str(path), default_timeout=60).run()
    assert not dashboard.exception
    assert dashboard.title[0].value == "BatteryLife AI"
    dashboard.sidebar.radio[0].set_value("Research & Methodology").run()
    assert not dashboard.exception
    assert any("Lucas Carreno" in item.value for item in dashboard.markdown)


def test_bundled_dashboard_views(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "dashboard/app.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BATTERYLIFE_MODEL", raising=False)
    app = AppTest.from_file(str(path), default_timeout=60).run()
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception and not app.error, page
        assert not any("Run the" in message.value for message in app.info), page
        if page == "Battery Life Predictor":
            assert any(metric.label == "Drain rate" for metric in app.metric)
        elif page == "Interactive Simulation":
            assert any(metric.label == "Final charge" for metric in app.metric)
        elif page == "Data Explorer":
            assert len(app.multiselect[0].options) == 5
