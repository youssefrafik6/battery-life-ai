"""Interactive BatteryLife AI dashboard."""

import json
import os
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from batterylife.engine import PRESETS, Engine, Prediction, Schedule, Segment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAGES = [
    "Overview",
    "Battery Life Predictor",
    "Interactive Simulation",
    "Model Comparison",
    "Data Explorer",
    "Uncertainty Analysis",
    "Research & Methodology",
]
TEAL = "#42dec4"


@st.cache_resource(show_spinner="Preparing the dataset and fitting the CPU models…")
def initialize_cloud_model() -> None:
    """Build local artifacts once per hosted process when enabled by the operator."""
    environment = os.environ | {"OMP_NUM_THREADS": "1"}
    commands = ("download", "prepare", "sensitivity", "train")
    for command in commands:
        subprocess.run(
            [sys.executable, "-m", "batterylife.cli", command, "--config", "configs/default.json"],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=180,
        )


@st.cache_resource
def load_engine():
    return Engine.load()


@st.cache_data
def read_json(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def read_intervals():
    path = RESULTS / "intervals.csv"
    return pd.read_csv(path, parse_dates=["time"]) if path.exists() else pd.DataFrame()


def hours_text(value):
    return f"{value:.2f} h" if value is not None else "Unbounded / outside horizon"


def plot_path(path):
    frame = pd.DataFrame({"Hours": path["hours"], "Charge (%)": path["soc"]})
    empty = path["empty_hours"]
    if empty is not None and empty not in path["hours"]:
        frame = pd.concat([frame, pd.DataFrame({"Hours": [empty], "Charge (%)": [0]})])
    chart = (
        alt.Chart(frame)
        .mark_line(point=True, color=TEAL)
        .encode(
            x="Hours:Q",
            y=alt.Y("Charge (%):Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Hours:Q", "Charge (%):Q"],
        )
        .properties(height=290)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def display_warnings(output):
    for warning in output["warnings"]:
        st.warning(warning)


def overview(report):
    st.title("BatteryLife AI")
    st.subheader("Predict, Simulate, and Understand Smartphone Battery Life.")
    st.markdown(
        "Explore how screen activity, network traffic, connection type, and temperature "
        "relate to battery drain. Build a usage schedule and see how charge changes over time."
    )
    if report:
        procedure = report["outer_summary"]["Selection procedure"]
        cols = st.columns(3)
        cols[0].metric("Eligible intervals", f"{report['intervals']:,}")
        cols[1].metric("Android devices", len(report["data_audit"]))
        cols[2].metric("Device holdout MAE", f"{procedure['mae']:.2f} pp/h")
        st.write(f"Deployment model · **{report['winner']}**")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("How the model works")
        st.write(
            "The model estimates drain from interval-average usage. The continuous-time "
            "simulator integrates that rate to produce a charge trajectory."
        )
        st.latex(r"\frac{dS}{dt}=-r(t)")
        st.write("Data from five Android device logs published with Zenodo record 13982080.")
    with right:
        st.subheader("Original COMAP research")
        st.write(
            "Youssef Rafik and Lucas Carreno reported 7.86 h for Wi-Fi streaming "
            "and 6.90 h for cellular streaming from full charge. Their median "
            "sequence simulation error was 4.82 percentage points."
        )


def predictor(engine):
    st.title("Battery Life Predictor")
    device = st.selectbox("Device", engine.devices)
    preset = st.selectbox("Usage scenario", list(PRESETS))
    defaults = PRESETS[preset]
    st.caption("Presets use illustrative feature values.")
    left, right = st.columns(2)
    with left:
        start = st.slider("Starting charge (%)", 0, 100, 80)
        screen = st.slider(
            "Screen-on share (%)", 0.0, 100.0, defaults["screen"] * 100, key=f"screen:{preset}"
        )
        temperature = st.number_input(
            "Temperature (°C)", -30.0, 100.0, float(defaults["temperature"]), key=f"temp:{preset}"
        )
    with right:
        network = st.selectbox(
            "Connection",
            ["wifi", "cellular"],
            index=int(defaults["network"] == "cellular"),
            key=f"net:{preset}",
        )
        down = st.number_input(
            "Download (MB/s)",
            0.0,
            10000.0,
            defaults["download"],
            step=0.001,
            format="%.4f",
            key=f"down:{preset}",
        )
        up = st.number_input(
            "Upload (MB/s)",
            0.0,
            10000.0,
            defaults["upload"],
            step=0.001,
            format="%.4f",
            key=f"up:{preset}",
        )
    output = engine.predict(
        Prediction(
            device=device,
            start_soc=start,
            screen=screen / 100,
            temperature=temperature,
            network=network,
            download=down,
            upload=up,
        )
    )
    cols = st.columns(3)
    cols[0].metric("Drain rate", f"{output['rate_pph']:.2f} pp/h")
    cols[1].metric("Time to empty", hours_text(output["empty_hours"]))
    cols[2].metric("Monte Carlo median", hours_text(output["tte_interval_hours"]["median"]))
    display_warnings(output)
    plot_path(output["trajectory"])
    st.caption("The constant-drain chart is limited to a 72-hour horizon.")
    st.write("Time-to-empty percentiles and width (hours)", output["tte_interval_hours"])
    with st.expander("Feature contributions"):
        st.write(output["explanation"])
        st.caption(
            "Feature contributions describe model associations. Reference contrasts "
            "vary one input at a time and need not add up."
        )


def simulation(engine):
    st.title("Interactive Simulation")
    st.write("Build a schedule using illustrative screen, network, and temperature inputs.")
    device = st.selectbox("Device", engine.devices)
    start = st.slider("Starting charge (%)", 0, 100, 80)
    count = st.number_input("Schedule segments", 1, 12, 3)
    segments = []
    for index in range(count):
        left, right = st.columns([2, 1])
        preset = left.selectbox(
            f"Activity {index + 1}",
            list(PRESETS),
            index=index % len(PRESETS),
            key=f"preset:{index}",
        )
        minutes = right.number_input(f"Minutes {index + 1}", 1, 1440, 30, key=f"mins:{index}")
        segments.append(Segment(device=device, minutes=minutes, **PRESETS[preset]))
    with st.expander("Scenario input assumptions"):
        st.dataframe(pd.DataFrame([item.model_dump() for item in segments]))
        st.write("Gaming, app-specific workloads, brightness, and GPS have no dedicated features.")
    output = engine.simulate(Schedule(start_soc=start, segments=segments))
    display_warnings(output)
    plot_path(output)
    cols = st.columns(2)
    cols[0].metric("Final charge", f"{output['soc'][-1]:.1f}%")
    cols[1].metric("Time to empty within schedule", hours_text(output["empty_hours"]))
    st.write("Final charge percentiles and width (percentage points)", output["final_soc_interval"])


def comparison(report):
    st.title("Model Comparison")
    if not report:
        st.info("Run the training pipeline to generate model scores.")
        return
    st.caption("Outer leave-one-device-out metrics. Drain MAE and RMSE are device macro averages.")
    models = pd.DataFrame(report["outer_summary"]).T
    st.dataframe(models, use_container_width=True)
    st.bar_chart(models[["mae", "rmse"]])
    folds = pd.DataFrame([f for fits in report["outer_folds"].values() for f in fits])
    st.subheader("Held-out devices")
    st.dataframe(
        folds[
            ["model", "device", "n", "mae", "rmse", "r2", "fit_seconds", "predict_seconds"]
        ].drop_duplicates(),
        use_container_width=True,
    )
    st.subheader("Chronological test on known devices")
    st.dataframe(
        pd.DataFrame(report["chronological"]).T[["n", "mae", "rmse", "r2", "sequence_count"]]
    )
    st.subheader("Held-out permutation importance")
    importance = st.selectbox("Model", list(report["permutation_importance"]))
    st.bar_chart(pd.Series(report["permutation_importance"][importance], name="MAE increase"))
    st.caption("Ridge importance uses log-drain MAE. Tree models use drain-rate MAE.")


def explorer():
    st.title("Data Explorer")
    frame = read_intervals()
    if frame.empty:
        st.info("Run the prepare command to generate the local explorer data.")
        return
    devices = st.multiselect(
        "Devices", sorted(frame.device.unique()), default=sorted(frame.device.unique())
    )
    frame = frame[frame.device.isin(devices)]
    feature = st.selectbox("Distribution", ["rate", "download", "upload", "screen", "temperature"])
    discharging = frame[frame.reason.eq("Discharge")]
    chart = (
        alt.Chart(discharging)
        .mark_bar(opacity=0.55)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=35)),
            y="count():Q",
            color="device:N",
            tooltip=["device:N", "count():Q"],
        )
        .properties(height=275)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    st.subheader("Charging and interval quality")
    st.dataframe(pd.crosstab(frame.device, frame.reason))
    st.subheader("State of charge over time")
    chart = (
        alt.Chart(frame)
        .mark_circle(size=20)
        .encode(
            x="time:T",
            y="soc:Q",
            color="device:N",
            tooltip=["time:T", "soc:Q", "reason:N"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def uncertainty(engine, report):
    st.title("Uncertainty Analysis")
    device = st.selectbox("Device", engine.devices)
    charge = st.slider("Starting charge (%)", 0, 100, 100)
    records = []
    for name, features in PRESETS.items():
        result = engine.predict(Prediction(device=device, start_soc=charge, **features))
        records.append({"scenario": name, **result["tte_interval_hours"]})
    data = pd.DataFrame(records)
    st.subheader("Residual Monte Carlo time to empty (hours)")
    base = alt.Chart(data).encode(y="scenario:N")
    st.altair_chart(
        (
            base.mark_rule(strokeWidth=5, color=TEAL).encode(x="p10:Q", x2="p90:Q")
            + base.mark_point(filled=True, color="#ffd166", size=90).encode(
                x="median:Q", tooltip=["scenario:N", "p10:Q", "median:Q", "p90:Q"]
            )
        ).interactive(),
        use_container_width=True,
    )
    st.dataframe(data)
    st.caption(
        "Blank upper bounds represent infinite draws. Percentiles describe empirical "
        "scenario uncertainty. The held-out coverage below concerns drain only."
    )
    bootstrap = engine.predict(Prediction(device=device, start_soc=charge, **PRESETS["Browsing"]))
    st.write(
        "Browsing · sequence bootstrap drain percentiles (pp/h)", bootstrap["bootstrap_rate_pph"]
    )
    if report:
        st.subheader("Residuals and coverage by device and connection")
        st.dataframe(pd.DataFrame(report["residual_diagnostics"]))
    sensitivity = read_json("sensitivity.json")
    if sensitivity:
        st.subheader("Ridge preprocessing sensitivity")
        st.dataframe(pd.DataFrame(sensitivity).T.drop(columns=["full_charge_tte_hours"]))
        st.dataframe(
            pd.DataFrame(
                {name: row["full_charge_tte_hours"] for name, row in sensitivity.items()}
            ).T
        )


def main():
    st.set_page_config(page_title="BatteryLife AI", page_icon="🔋", layout="wide")
    setup = os.environ.get("BATTERYLIFE_CLOUD_SETUP")
    if setup is None:
        try:
            setup = st.secrets.get("BATTERYLIFE_CLOUD_SETUP")
        except FileNotFoundError:
            pass
    if setup == "1" and not (RESULTS / "model.joblib").exists():
        try:
            initialize_cloud_model()
        except (subprocess.SubprocessError, OSError):
            st.error("Model setup failed. Check the application logs and retry.")
            st.stop()
    with st.sidebar:
        st.markdown("## 🔋 BatteryLife AI")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Android battery drain and charge simulation")
    report = read_json("evaluation.json")
    if page == "Overview":
        overview(report)
    elif page == "Research & Methodology":
        st.title(page)
        st.markdown((ROOT / "docs/original_research.md").read_text())
        st.markdown((ROOT / "docs/methodology.md").read_text())
    elif page == "Data Explorer":
        explorer()
    elif page == "Model Comparison":
        comparison(report)
    else:
        try:
            engine = load_engine()
        except FileNotFoundError:
            st.info("Run the training pipeline to enable predictions and simulations.")
            return
        if page == "Battery Life Predictor":
            predictor(engine)
        elif page == "Interactive Simulation":
            simulation(engine)
        else:
            uncertainty(engine, report)


if __name__ == "__main__":
    main()
