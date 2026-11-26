from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_overview_and_research():
    path = Path(__file__).resolve().parents[1] / "dashboard/app.py"
    dashboard = AppTest.from_file(str(path)).run()
    assert not dashboard.exception
    assert dashboard.title[0].value == "BatteryLife AI"
    dashboard.sidebar.radio[0].set_value("Research & Methodology").run()
    assert not dashboard.exception
    assert any("Lucas Carreno" in item.value for item in dashboard.markdown)
