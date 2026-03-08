import io
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EV–V2G–PV Digital Twin", page_icon="⚡", layout="wide")

DT = 0.5
T = 48
TIME = range(T)
DEFAULT_BASE_PROFILE = np.array([
    0.30, 0.29, 0.28, 0.27, 0.27, 0.28, 0.30, 0.32,
    0.35, 0.39, 0.44, 0.50, 0.55, 0.57, 0.52, 0.47,
    0.43, 0.40, 0.38, 0.37, 0.39, 0.42, 0.46, 0.50,
    0.54, 0.58, 0.60, 0.58, 0.55, 0.52, 0.50, 0.49,
    0.51, 0.56, 0.63, 0.72, 0.80, 0.85, 0.88, 0.86,
    0.80, 0.74, 0.66, 0.58, 0.50, 0.44, 0.38, 0.34,
], dtype=float)
CANDIDATE_COLS = ["Wed", "Wednesday", "WED", "wed", "WEDNESDAY"]
MILES_BINS = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], float)
PDF_UK = np.array([0.31, 0.25, 0.16, 0.11, 0.08, 0.06, 0.045, 0.03, 0.018, 0.01, 0.006, 0.003], float)
PDF_UK /= PDF_UK.sum()


def slot(hh: float) -> int:
    return int(round(hh / DT))


def default_gamma() -> np.ndarray:
    gamma = np.zeros(T)
    gamma[slot(10.0):slot(16.0)] = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2])
    return gamma


def default_alpha() -> np.ndarray:
    return np.array([0.3] * 20 + [0.8] * 12 + [0.3] * 16, dtype=float)


def load_base_profile(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        return DEFAULT_BASE_PROFILE.copy()
    df = pd.read_csv(uploaded_file)
    col = None
    for c in CANDIDATE_COLS:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"Cannot find Wednesday column in CSV. Available columns: {list(df.columns)}")
    values = df[col].to_numpy(dtype=float)
    if len(values) != T:
        raise ValueError(f"The input CSV must contain exactly {T} half-hour rows.")
    return values


def build_connected_counts(ev_count: int, midday_fraction: float) -> np.ndarray:
    midday = max(1, int(round(ev_count * midday_fraction)))
    counts = np.zeros(T, dtype=int)
    counts[:slot(7.0)] = ev_count
    counts[slot(18.0):slot(24.0)] = ev_count
    counts[slot(11.0):slot(14.0)] = midday
    return counts


def allocate_scaled(capacity_kw: np.ndarray, required_net_kwh: float, eta: float = 1.0) -> np.ndarray:
    max_net_kwh = np.sum(capacity_kw * DT * eta)
    if required_net_kwh <= 1e-9 or max_net_kwh <= 1e-9:
        return np.zeros_like(capacity_kw)
    return capacity_kw * min(1.0, required_net_kwh / max_net_kwh)


def allocate_priority(capacity_kw: np.ndarray, priority: np.ndarray, required_gross_kwh: float) -> np.ndarray:
    out = np.zeros_like(capacity_kw)
    order = sorted(range(len(capacity_kw)), key=lambda i: (priority[i], capacity_kw[i]), reverse=True)
    remaining = required_gross_kwh
    for i in order:
        if remaining <= 1e-9:
            break
        max_energy = capacity_kw[i] * DT
        used = min(max_energy, remaining)
        out[i] = used / DT
        remaining -= used
    return out


def binary_search(low: float, high: float, feasible, iters: int = 50) -> float:
    l, h = low, high
    for _ in range(iters):
        mid = 0.5 * (l + h)
        if feasible(mid):
            h = mid
        else:
            l = mid
    return h


def heuristic_s1(p_t: np.ndarray, cap_t: np.ndarray, trip_energy_kwh: float, eta_c: float):
    def feasible(z):
        charge_cap = np.minimum(cap_t, np.maximum(0.0, z - p_t))
        return np.sum(charge_cap * DT * eta_c) >= trip_energy_kwh

    z = binary_search(float(np.max(p_t)), float(np.max(p_t) + np.max(cap_t) + 50), feasible)
    charge_cap = np.minimum(cap_t, np.maximum(0.0, z - p_t))
    x = allocate_scaled(charge_cap, trip_energy_kwh, eta_c)
    y = np.zeros_like(x)
    s = np.zeros_like(x)
    pgrid = p_t + x
    return float(np.max(pgrid)), x, y, s, pgrid


def heuristic_s2_like(p_t: np.ndarray, cap_t: np.ndarray, trip_energy_kwh: float, eta_c: float, eta_d: float, priority: np.ndarray | None):
    min_z = float(np.max(np.maximum(0.0, p_t - cap_t)))
    max_z = float(np.max(p_t) + np.max(cap_t) + 50)

    def feasible(z):
        req_dis = np.minimum(cap_t, np.maximum(0.0, p_t - z))
        if np.any(np.maximum(0.0, p_t - z) - cap_t > 1e-8):
            return False
        charge_cap = np.minimum(cap_t, np.maximum(0.0, z - p_t))
        max_net = np.sum(charge_cap * DT * eta_c) - np.sum(req_dis * DT / eta_d)
        return max_net >= trip_energy_kwh

    z = binary_search(min_z, max_z, feasible)
    y = np.minimum(cap_t, np.maximum(0.0, p_t - z))
    charge_cap = np.minimum(cap_t, np.maximum(0.0, z - p_t))
    gross_need = (trip_energy_kwh + np.sum(y * DT / eta_d)) / eta_c
    if priority is None:
        x = allocate_scaled(charge_cap, trip_energy_kwh + np.sum(y * DT / eta_d), eta_c)
    else:
        x = allocate_priority(charge_cap, priority, gross_need)
    s = np.zeros_like(x)
    pgrid = np.maximum(0.0, p_t + x - y)
    return float(np.max(pgrid)), x, y, s, pgrid


def heuristic_s4(p_t: np.ndarray, cap_t: np.ndarray, trip_energy_kwh: float, eta_c: float, eta_d: float, gamma_t: np.ndarray, pv_cap: float, ev_count: int, alpha_t: np.ndarray, rho: float):
    pv_avail = gamma_t * pv_cap * ev_count
    s_raw = pv_avail * (0.65 + 0.35 * rho * (gamma_t > 0).astype(float))
    max_solar = np.sum(s_raw * DT)
    solar_scale = min(1.0, trip_energy_kwh / max(max_solar, 1e-9)) if max_solar > 0 else 0.0
    s = s_raw * solar_scale
    remaining = max(0.0, trip_energy_kwh - np.sum(s * DT))
    p_t_net = np.maximum(0.0, p_t - s)
    _, x, y, _, _ = heuristic_s2_like(p_t_net, cap_t, remaining, eta_c, eta_d, alpha_t + 0.25 * gamma_t)
    pgrid = np.maximum(0.0, p_t + x - y - s)
    return float(np.max(pgrid)), x, y, s, pgrid


def expand_to_matrix(series: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((T, 1))
    return np.tile((series / n)[:, None], (1, n))


def build_soc_series(x_mat: np.ndarray, y_mat: np.ndarray, s_mat: np.ndarray, soc0_j: np.ndarray, eta_c: float, eta_d: float) -> np.ndarray:
    inv_eta_d = 1.0 / eta_d
    current = soc0_j.copy().astype(float)
    out = []
    for t in TIME:
        current = current + eta_c * x_mat[t, :] * DT - y_mat[t, :] * inv_eta_d * DT + s_mat[t, :] * DT
        out.append(current.mean())
    return np.array(out)


def charging_losses_kwh(x_mat: np.ndarray, y_mat: np.ndarray, eta_c: float, eta_d: float):
    ex = x_mat.sum() * DT
    ey = y_mat.sum() * DT
    loss_c = (1.0 - eta_c) * ex
    loss_d = (1.0 / eta_d - 1.0) * ey
    return loss_c + loss_d, loss_c, loss_d


def feeder_i2r_losses_kwh(pgrid: np.ndarray, r_phase: float, pf: float):
    v_ll = 400.0
    current = (pgrid * 1e3) / (np.sqrt(3) * v_ll * pf)
    ploss_w = 3.0 * r_phase * (current ** 2)
    return (ploss_w * DT).sum() / 1000.0


def estimate_gr_gf(pgrid: np.ndarray, alpha_t: np.ndarray):
    return alpha_t * pgrid, (1 - alpha_t) * pgrid


def make_ts_df(name: str, p_t: np.ndarray, pgrid: np.ndarray, x_mat: np.ndarray, y_mat: np.ndarray, s_mat: np.ndarray, gr: np.ndarray, gf: np.ndarray, soc: np.ndarray) -> pd.DataFrame:
    hours = np.arange(T) * DT
    return pd.DataFrame({
        "hour": hours,
        "time_label": [f"{int(h):02d}:{'30' if h % 1 else '00'}" for h in hours],
        "scenario": name,
        "base_load_kw": p_t,
        "grid_kw": pgrid,
        "charge_kw": x_mat.sum(axis=1),
        "discharge_kw": y_mat.sum(axis=1),
        "pv_kw": s_mat.sum(axis=1),
        "grid_renewable_kw": gr,
        "grid_fossil_kw": gf,
        "avg_soc_kwh": soc,
    })


def make_metrics(name: str, z: float, x_mat: np.ndarray, y_mat: np.ndarray, s_mat: np.ndarray, pgrid: np.ndarray, gr: np.ndarray, gf: np.ndarray, eta_c: float, eta_d: float, r_phase: float, pf: float) -> dict:
    conv, charge_loss, discharge_loss = charging_losses_kwh(x_mat, y_mat, eta_c, eta_d)
    i2r = feeder_i2r_losses_kwh(pgrid, r_phase, pf)
    return {
        "scenario": name,
        "peak_kw": float(z),
        "charge_energy_kwh": float(x_mat.sum() * DT),
        "discharge_energy_kwh": float(y_mat.sum() * DT),
        "pv_used_kwh": float(s_mat.sum() * DT),
        "renewable_grid_kwh": float(np.sum(gr) * DT),
        "fossil_grid_kwh": float(np.sum(gf) * DT),
        "conversion_loss_kwh": float(conv),
        "charge_loss_kwh": float(charge_loss),
        "discharge_loss_kwh": float(discharge_loss),
        "i2r_loss_kwh": float(i2r),
    }


@st.cache_data(show_spinner=False)
def run_twin(base_profile_tuple: Tuple[float, ...], n_house: int, base_scale: float, ev_pen: float, soc0: float, battery_kwh: float, p_max: float, eta_c: float, eta_d: float, pv_cap: float, midday_fraction: float, rho: float, r_phase: float, pf: float, seed: int):
    rng = np.random.default_rng(seed)
    base_load = np.array(base_profile_tuple, dtype=float)
    p_t = base_load * n_house * base_scale
    ev_count = max(1, int(round(n_house * ev_pen)))
    d_miles = rng.choice(MILES_BINS, size=ev_count, p=PDF_UK)
    e_per_mile = np.clip(rng.normal(0.30, 0.03, size=ev_count), 0.25, 0.4)
    e_j = d_miles * e_per_mile
    total_trip_energy = float(np.sum(e_j))
    soc0_j = np.full(ev_count, soc0)
    connected_counts = build_connected_counts(ev_count, midday_fraction)
    cap_t = connected_counts * p_max
    alpha_t = default_alpha()
    gamma_t = default_gamma()

    z1, x1, y1, s1, p1 = heuristic_s1(p_t, cap_t, total_trip_energy, eta_c)
    z2, x2, y2, s2, p2 = heuristic_s2_like(p_t, cap_t, total_trip_energy, eta_c, eta_d, None)
    z3, x3, y3, s3, p3 = heuristic_s2_like(p_t, cap_t, total_trip_energy, eta_c, eta_d, alpha_t)
    z4, x4, y4, s4, p4 = heuristic_s4(p_t, cap_t, total_trip_energy, eta_c, eta_d, gamma_t, pv_cap, ev_count, alpha_t, rho)

    x1m, y1m, s1m = expand_to_matrix(x1, ev_count), expand_to_matrix(y1, ev_count), np.zeros((T, ev_count))
    x2m, y2m, s2m = expand_to_matrix(x2, ev_count), expand_to_matrix(y2, ev_count), np.zeros((T, ev_count))
    x3m, y3m, s3m = expand_to_matrix(x3, ev_count), expand_to_matrix(y3, ev_count), np.zeros((T, ev_count))
    x4m, y4m, s4m = expand_to_matrix(x4, ev_count), expand_to_matrix(y4, ev_count), expand_to_matrix(s4, ev_count)

    gr1, gf1 = estimate_gr_gf(p1, alpha_t)
    gr2, gf2 = estimate_gr_gf(p2, alpha_t)
    gr3, gf3 = estimate_gr_gf(p3, alpha_t)
    gr4, gf4 = estimate_gr_gf(p4, alpha_t)

    soc1 = build_soc_series(x1m, y1m, s1m, soc0_j, eta_c, eta_d)
    soc2 = build_soc_series(x2m, y2m, s2m, soc0_j, eta_c, eta_d)
    soc3 = build_soc_series(x3m, y3m, s3m, soc0_j, eta_c, eta_d)
    soc4 = build_soc_series(x4m, y4m, s4m, soc0_j, eta_c, eta_d)

    ts = pd.concat([
        make_ts_df("S1", p_t, p1, x1m, y1m, s1m, gr1, gf1, soc1),
        make_ts_df("S2", p_t, p2, x2m, y2m, s2m, gr2, gf2, soc2),
        make_ts_df("S3", p_t, p3, x3m, y3m, s3m, gr3, gf3, soc3),
        make_ts_df("S4", p_t, p4, x4m, y4m, s4m, gr4, gf4, soc4),
    ], ignore_index=True)

    metrics = pd.DataFrame([
        make_metrics("S1", z1, x1m, y1m, s1m, p1, gr1, gf1, eta_c, eta_d, r_phase, pf),
        make_metrics("S2", z2, x2m, y2m, s2m, p2, gr2, gf2, eta_c, eta_d, r_phase, pf),
        make_metrics("S3", z3, x3m, y3m, s3m, p3, gr3, gf3, eta_c, eta_d, r_phase, pf),
        make_metrics("S4", z4, x4m, y4m, s4m, p4, gr4, gf4, eta_c, eta_d, r_phase, pf),
    ])
    s1_peak = metrics.loc[metrics["scenario"] == "S1", "peak_kw"].iloc[0]
    metrics["peak_reduction_vs_s1_pct"] = (s1_peak - metrics["peak_kw"]) / s1_peak * 100.0
    metrics.loc[metrics["scenario"] == "S1", "peak_reduction_vs_s1_pct"] = 0.0

    summary = {
        "houses": n_house,
        "ev_fleet": ev_count,
        "total_trip_energy_kwh": round(total_trip_energy, 2),
        "aggregate_initial_soc_kwh": round(float(np.sum(soc0_j)), 2),
    }
    return metrics, ts, summary


def fig_power(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time_label"], y=df["base_load_kw"], mode="lines", name="Base load"))
    fig.add_trace(go.Scatter(x=df["time_label"], y=df["grid_kw"], mode="lines", name="Grid net load"))
    fig.add_trace(go.Scatter(x=df["time_label"], y=df["charge_kw"], mode="lines", name="Charge"))
    fig.add_trace(go.Scatter(x=df["time_label"], y=df["discharge_kw"], mode="lines", name="Discharge"))
    if df["pv_kw"].sum() > 0:
        fig.add_trace(go.Scatter(x=df["time_label"], y=df["pv_kw"], mode="lines", name="PV used"))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Time", yaxis_title="kW")
    return fig


def fig_soc(df: pd.DataFrame):
    fig = px.area(df, x="time_label", y="avg_soc_kwh", labels={"time_label": "Time", "avg_soc_kwh": "Average SOC (kWh)"})
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def fig_peaks(metrics: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics["scenario"], y=metrics["peak_kw"], name="Peak demand (kW)"))
    fig.add_trace(go.Bar(x=metrics["scenario"], y=metrics["peak_reduction_vs_s1_pct"], name="Peak reduction vs S1 (%)", yaxis="y2"))
    fig.update_layout(height=360, barmode="group", yaxis=dict(title="kW"), yaxis2=dict(title="%", overlaying="y", side="right"), margin=dict(l=20, r=20, t=40, b=20))
    return fig


def fig_losses(metrics: pd.DataFrame):
    df = metrics.melt(id_vars=["scenario"], value_vars=["conversion_loss_kwh", "i2r_loss_kwh"], var_name="loss_type", value_name="value")
    fig = px.bar(df, x="scenario", y="value", color="loss_type", barmode="group", labels={"value": "kWh"})
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def fig_mix(metrics: pd.DataFrame):
    df = metrics[["scenario", "renewable_grid_kwh", "fossil_grid_kwh", "pv_used_kwh"]].melt(id_vars=["scenario"], var_name="energy_type", value_name="kwh")
    fig = px.bar(df, x="scenario", y="kwh", color="energy_type", barmode="stack", labels={"kwh": "Energy (kWh)"})
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig


st.title("⚡ EV–V2G–PV Digital Twin App")
st.caption("Deployable web app based on the article structure and scenario logic of your EV/V2G/PV study.")

with st.sidebar:
    st.header("Input layer")
    uploaded = st.file_uploader("Optional baseline CSV", type=["csv"], help="Upload weekday_averages_30min1.csv with a Wednesday column.")
    houses = st.slider("Number of houses", 20, 150, 55, 1)
    base_scale = st.slider("Base load scale", 0.5, 1.5, 1.0, 0.05)
    ev_pen = st.slider("EV penetration", 0.1, 1.0, 0.5, 0.05)
    soc0 = st.slider("Initial energy per EV (kWh)", 0.0, 30.0, 10.0, 1.0)
    battery_kwh = st.slider("Battery capacity (kWh)", 30.0, 100.0, 60.0, 5.0)
    p_max = st.slider("Maximum charge/discharge power (kW)", 2.0, 11.0, 4.0, 0.5)
    eta_c = st.slider("Charging efficiency", 0.80, 0.99, 0.92, 0.01)
    eta_d = st.slider("Discharging efficiency", 0.80, 0.99, 0.92, 0.01)
    pv_cap = st.slider("PV capacity per EV/home (kW)", 0.0, 6.0, 3.0, 0.25)
    midday_fraction = st.slider("Midday at-home EV fraction", 0.05, 0.50, 0.14, 0.01)
    rho = st.slider("Solar priority coefficient ρ", 0.0, 1.0, 0.8, 0.05)
    r_phase = st.slider("Equivalent feeder resistance (Ω)", 0.03, 0.25, 0.10, 0.01)
    pf = st.slider("Power factor", 0.90, 1.00, 0.98, 0.01)
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=2025, step=1)
    run = st.button("Run digital twin", type="primary", use_container_width=True)

try:
    base_profile = load_base_profile(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

if "results" not in st.session_state or run:
    with st.spinner("Running scenario engine..."):
        st.session_state.results = run_twin(tuple(base_profile.tolist()), houses, base_scale, ev_pen, soc0, battery_kwh, p_max, eta_c, eta_d, pv_cap, midday_fraction, rho, r_phase, pf, int(seed))

metrics, timeseries, summary = st.session_state.results
selected = st.radio("Scenario layer", ["S1", "S2", "S3", "S4"], horizontal=True)
selected_df = timeseries[timeseries["scenario"] == selected].copy()
selected_metrics = metrics[metrics["scenario"] == selected].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Houses", summary["houses"])
c2.metric("EV fleet", summary["ev_fleet"])
c3.metric("Trip energy", f"{summary['total_trip_energy_kwh']:.1f} kWh")
c4.metric("Selected peak", f"{selected_metrics['peak_kw']:.2f} kW")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak reduction vs S1", f"{selected_metrics['peak_reduction_vs_s1_pct']:.2f}%")
m2.metric("Conversion loss", f"{selected_metrics['conversion_loss_kwh']:.2f} kWh")
m3.metric("Feeder I²R loss", f"{selected_metrics['i2r_loss_kwh']:.2f} kWh")
m4.metric("PV used", f"{selected_metrics['pv_used_kwh']:.2f} kWh")

left, right = st.columns([1.4, 1.0])
with left:
    st.subheader("Power profile")
    st.plotly_chart(fig_power(selected_df), use_container_width=True)
with right:
    st.subheader("Fleet SOC evolution")
    st.plotly_chart(fig_soc(selected_df), use_container_width=True)

l2, r2 = st.columns(2)
with l2:
    st.subheader("Peak comparison")
    st.plotly_chart(fig_peaks(metrics), use_container_width=True)
with r2:
    st.subheader("Loss comparison")
    st.plotly_chart(fig_losses(metrics), use_container_width=True)

st.subheader("Energy source composition")
st.plotly_chart(fig_mix(metrics), use_container_width=True)

st.subheader("Metrics table")
st.dataframe(metrics, use_container_width=True)

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    metrics.to_excel(writer, sheet_name="summary_metrics", index=False)
    for s in ["S1", "S2", "S3", "S4"]:
        timeseries[timeseries["scenario"] == s].to_excel(writer, sheet_name=f"{s}_timeseries", index=False)
excel_buffer.seek(0)

st.subheader("Download layer")
col_a, col_b = st.columns(2)
with col_a:
    st.download_button("Download summary_metrics.csv", metrics.to_csv(index=False).encode("utf-8"), file_name="summary_metrics.csv", mime="text/csv")
with col_b:
    st.download_button("Download results_timeseries.xlsx", excel_buffer.getvalue(), file_name="results_timeseries.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("Deployment"):
    st.markdown(
        """
        Put `app.py` and `requirements.txt` into a GitHub repository, then connect it to Streamlit Community Cloud.
        The hosting platform will generate a public link that anyone can open in a browser.
        """
    )
