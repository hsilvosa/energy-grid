from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from energy_grid.domain import EventType
from energy_grid.features import build_horizon_features
from energy_grid.forecasting import create_forecaster
from energy_grid.sources.dataset_reader import DEFAULT_DATASET_PATH, EntsoeDatasetReader

COUNTRY_OPTIONS = [
    ("Spain (ES)", "ES"),
    ("France (FR)", "FR"),
    ("Germany (DE)", "DE"),
    ("Italy (IT)", "IT"),
    ("Portugal (PT)", "PT"),
    ("Netherlands (NL)", "NL"),
    ("Belgium (BE)", "BE"),
    ("Poland (PL)", "PL"),
    ("Austria (AT)", "AT"),
    ("Norway (NO)", "NO"),
    ("Sweden (SE)", "SE"),
    ("Greece (GR)", "GR"),
    ("Switzerland (CH)", "CH"),
]

MODEL_CHOICES = [
    ("Dual-Horizon Ensemble (Short + Long Lead Time)", "dual_horizon"),
    ("Stacked Multi-Model (LightGBM + XGBoost + CatBoost)", "stacked"),
    ("LightGBM Quantile Forecaster", "lightgbm"),
    ("XGBoost Quantile Forecaster", "xgboost"),
    ("CatBoost Quantile Forecaster", "catboost"),
]


def create_interactive_forecast_dashboard(
    timestamps: pd.DatetimeIndex,
    p50: np.ndarray[Any, Any],
    p10: np.ndarray[Any, Any],
    p90: np.ndarray[Any, Any],
    actuals: np.ndarray[Any, Any] | None = None,
    history_times: pd.DatetimeIndex | None = None,
    history_vals: np.ndarray[Any, Any] | None = None,
    lead_times: pd.DatetimeIndex | None = None,
    lead_vals: np.ndarray[Any, Any] | None = None,
    country_label: str = "Spain",
    target_name: str = "Demand",
    unit: str = "MW",
) -> go.Figure:
    """Create a power market dashboard with range slider and residual error."""
    # Create 2 subplots: Main forecast (top 75%) and Residual Error (bottom 25%)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=[
            f"{country_label} Electricity {target_name} ({unit}) - Interactive Pan & Zoom",
            "Point Forecast Residual Error (Actual - P50)",
        ],
    )

    # 1. Historical context trace
    if history_times is not None and history_vals is not None and len(history_times):
        fig.add_trace(
            go.Scatter(
                x=history_times,
                y=history_vals,
                mode="lines",
                name="Historical Context",
                line=dict(color="#475569", width=1.5),
                hoverinfo="x+y",
            ),
            row=1,
            col=1,
        )

    # Observations inside the lead window are useful for retrospective context,
    # but they were unavailable at the model input cutoff and are never features.
    if lead_times is not None and lead_vals is not None and len(lead_times):
        fig.add_trace(
            go.Scatter(
                x=lead_times,
                y=lead_vals,
                mode="lines",
                name="Lead Window Actuals (Not Used)",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                hoverinfo="x+y",
            ),
            row=1,
            col=1,
        )
        fig.add_vrect(
            x0=lead_times.min(),
            x1=timestamps.min(),
            fillcolor="#f59e0b",
            opacity=0.08,
            line_width=0,
            annotation_text="24 h lead window",
            annotation_position="top left",
            row=1,
            col=1,
        )

    # 2. Conformal Prediction Interval (P10 to P90 ribbon)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=p90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(14, 165, 233, 0.25)",
            name="80% Calibrated Interval (P10-P90)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # 3. Point Forecast (P50)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=p50,
            mode="lines+markers",
            name="Point Forecast (P50)",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # 4. Actual observation trace
    if actuals is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actuals,
                mode="lines+markers",
                name="Actual Observation",
                line=dict(color="#10b981", width=2, dash="dot"),
                marker=dict(size=4),
            ),
            row=1,
            col=1,
        )

        # 5. Residual error bar chart in row 2
        residuals = actuals - p50
        colors = ["#ef4444" if r < 0 else "#3b82f6" for r in residuals]
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=residuals,
                name="Residual (MW/EUR)",
                marker=dict(color=colors),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Add Range Slider and Range Selector Buttons to X-Axis
    fig.update_xaxes(
        row=2,
        col=1,
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=3, label="3D", step="day", stepmode="backward"),
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=14, label="14D", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
            bgcolor="#f8fafc",
            activecolor="#0284c7",
        ),
        type="date",
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


def create_seasonality_heatmap(series: pd.Series, target_name: str, unit: str) -> go.Figure:
    """Create hour-of-day vs day-of-week average profile heatmap."""
    df = pd.DataFrame({"time": series.index, "val": series.values})
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.pivot_table(index="dow", columns="hour", values="val", aggfunc="mean").reindex(order)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(range(24)),
            y=pivot.index,
            colorscale="Viridis",
            colorbar=dict(title=unit),
        )
    )
    fig.update_layout(
        title=f"Average {target_name} Profile by Day of Week & Hour ({unit})",
        xaxis_title="Hour of Day (UTC)",
        yaxis_title="Day of Week",
        template="plotly_white",
        height=380,
    )
    return fig


def build_gradio_app(data_dir: Path = DEFAULT_DATASET_PATH) -> Any:
    reader = EntsoeDatasetReader(data_dir)

    def forecast_interface(
        country_code: str,
        zone_key: str,
        target_choice: str,
        model_type: str,
        test_date_str: str,
        history_days: int,
        apply_cal: bool,
    ) -> tuple[go.Figure | None, str, dict[str, float]]:
        target = EventType.DEMAND if "Demand" in target_choice else EventType.PRICE
        steps_per_day = 96 if target == EventType.DEMAND else 24
        unit = "MW" if target == EventType.DEMAND else "EUR/MWh"

        # Load series for selected European country
        try:
            series = reader.load_series(
                target, country_code=country_code, zone_key=zone_key.strip() or None,
                start_year=2023, end_year=2026,
                resample_freq="15min" if target == EventType.DEMAND else "60min",
            )
        except (FileNotFoundError, ValueError) as exc:
            return None, f"Error: {exc}", {}
        if series.empty:
            return None, f"Error: No data found for {country_code} in dataset directory.", {}

        # Feature engineering
        df = build_horizon_features(
            series, horizon_steps=steps_per_day, steps_per_day=steps_per_day,
            country_code=country_code,
        )
        feature_names = [c for c in df.columns if c != "target"]

        # Parse test date
        try:
            target_dt = pd.to_datetime(test_date_str, utc=True)
        except Exception:
            target_dt = df.index[-steps_per_day]

        # Find test window
        test_mask = (df.index >= target_dt) & (df.index < target_dt + timedelta(days=1))
        if not test_mask.any():
            test_df = df.tail(steps_per_day)
            train_df = df.iloc[:-steps_per_day]
        else:
            test_idx_start = np.where(test_mask)[0][0]
            train_df = df.iloc[:test_idx_start]
            test_df = df.iloc[test_idx_start : test_idx_start + steps_per_day]

        if len(train_df) < steps_per_day * 10:
            train_df = df.iloc[:-steps_per_day]
            test_df = df.tail(steps_per_day)

        # Labels inside the lead-time gap are unknown at the first test origin.
        first_test_pos = df.index.get_loc(test_df.index[0])
        train_df = df.iloc[: max(0, first_test_pos - steps_per_day)]
        if len(train_df) < steps_per_day * 10:
            return None, "Error: too little history before the selected forecast origins.", {}
        history_df = train_df
        lead_context = series[
            (series.index > history_df.index.max()) & (series.index < test_df.index.min())
        ]
        if model_type == "dual_horizon":
            short_df = build_horizon_features(
                series, horizon_steps=1, steps_per_day=steps_per_day,
                country_code=country_code,
            )
            short_df = short_df[short_df.index <= train_df.index.max()]
            train_df = pd.concat([train_df, short_df]).sort_index(kind="stable")

        # Train model
        model_options = (
            {"switch_horizon_steps": 6 * steps_per_day // 24}
            if model_type == "dual_horizon" else {}
        )
        model = create_forecaster(model_type, **model_options)  # type: ignore[arg-type]
        model.fit(
            train_df,
            train_df["target"],
            feature_names=feature_names,
            calibration_fraction=0.15,
        )

        p10, p50, p90 = model.predict(test_df, apply_calibration=apply_cal)
        actuals = test_df["target"].to_numpy()

        # Historical context window requested by user (e.g. 7, 14, 30 days)
        hist_steps = steps_per_day * history_days
        hist_df = history_df.tail(hist_steps)

        country_name = dict((code, label) for label, code in COUNTRY_OPTIONS).get(
            country_code, country_code
        )
        if zone_key.strip():
            country_name += f" / {zone_key.strip()}"
        fig = create_interactive_forecast_dashboard(
            timestamps=test_df.index,
            p50=p50,
            p10=p10,
            p90=p90,
            actuals=actuals,
            history_times=hist_df.index,
            history_vals=hist_df["target"].to_numpy(),
            lead_times=pd.DatetimeIndex(lead_context.index),
            lead_vals=lead_context.to_numpy(),
            country_label=country_name,
            target_name=target.value.capitalize(),
            unit=unit,
        )

        mae = float(np.mean(np.abs(actuals - p50)))
        rmse = float(np.sqrt(np.mean(np.square(actuals - p50))))
        denom = float(np.sum(np.abs(actuals)))
        wape = float(np.sum(np.abs(actuals - p50)) / denom * 100.0) if denom else 0.0
        cov = float(np.mean((actuals >= p10) & (actuals <= p90))) * 100.0
        baseline = series.shift(steps_per_day).reindex(test_df.index).to_numpy()
        baseline_mae = float(np.mean(np.abs(actuals - baseline)))
        first_origin = test_df.index.min() - timedelta(days=1)
        last_origin = test_df.index.max() - timedelta(days=1)
        cal_status = (
            "Split-Conformal Calibrated (80% Nominal Target)" if apply_cal else "Raw Uncalibrated"
        )

        summary_text = (
            f"### Evaluation Results for {country_name} on {test_df.index.min().date()}\n\n"
            f"| Metric | Result |\n"
            f"|---|---:|\n"
            f"| **Model Strategy** | `{model_type.upper()}` |\n"
            f"| **Forecast Lead Time** | 24 hours |\n"
            f"| **Forecast Origins (UTC)** | {first_origin} to {last_origin} |\n"
            f"| **Latest Training Observation (UTC)** | {history_df.index.max()} |\n"
            f"| **Lead-Window Actuals** | Displayed for context; excluded from model inputs |\n"
            f"| **Mean Absolute Error (MAE)** | **{mae:.2f} {unit}** |\n"
            f"| **Previous-Day Baseline MAE** | **{baseline_mae:.2f} {unit}** |\n"
            f"| **Root Mean Squared Error (RMSE)** | **{rmse:.2f} {unit}** |\n"
            f"| **WAPE (Weighted Absolute Percentage Error)** | **{wape:.2f}%** |\n"
            f"| **P10-P90 Empirical Interval Coverage** | **{cov:.1f}%** |\n"
            f"| **Uncertainty Calibration Status** | {cal_status} |\n"
        )

        importances = model.get_feature_importances()
        top_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

        return fig, summary_text, top_importances

    def analytics_interface(
        country_code: str, zone_key: str, target_choice: str
    ) -> go.Figure | None:
        target = EventType.DEMAND if "Demand" in target_choice else EventType.PRICE
        unit = "MW" if target == EventType.DEMAND else "EUR/MWh"
        try:
            series = reader.load_series(
                target, country_code=country_code, zone_key=zone_key.strip() or None,
                start_year=2024, end_year=2026, resample_freq="60min",
            )
        except (FileNotFoundError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        if series.empty:
            return None
        return create_seasonality_heatmap(series, target.value.capitalize(), unit)

    with gr.Blocks(title="Pan-European Energy Grid AI Forecaster") as demo:
        gr.Markdown(
            """
            # Pan-European Energy Grid Intelligence & Forecaster Studio
            ### Multi-Model Ensembles (Dual-Horizon, Stacked, LightGBM, XGBoost, CatBoost)
            ### Conformal Prediction Uncertainty Calibration Covering 34 European Countries
            """
        )

        with gr.Tabs():
            with gr.Tab("Forecast Studio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        country_dropdown = gr.Dropdown(
                            choices=[(label, code) for label, code in COUNTRY_OPTIONS],
                            value="ES",
                            label="European Market",
                        )
                        zone_input = gr.Textbox(
                            value="", label="Bidding Zone",
                            info="Required when a country has multiple zones (for example DE_LU)",
                        )
                        target_radio = gr.Radio(
                            choices=["Electricity Demand (MW)", "Day-Ahead Price (EUR/MWh)"],
                            value="Electricity Demand (MW)",
                            label="Target Variable",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=[(label, code) for label, code in MODEL_CHOICES],
                            value="lightgbm",
                            label="Model Strategy",
                        )
                        date_input = gr.Textbox(
                            value="2025-02-20",
                            label="Evaluation Date (YYYY-MM-DD)",
                            info="Select an out-of-sample date (e.g. 2024 to 2026)",
                        )
                        history_slider = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=7,
                            step=1,
                            label="Historical Context Window (Days)",
                            info="Number of historical days to display in the chart",
                        )
                        cal_check = gr.Checkbox(
                            value=True,
                            label="Apply Conformal Calibration (80% Interval)",
                            info="Apply interval calibration fitted on later training data",
                        )
                        predict_btn = gr.Button("Generate Forecast", variant="primary")

                    with gr.Column(scale=3):
                        plot_output = gr.Plot(label="Forecast Dashboard & Range Slider")
                        metrics_output = gr.Markdown()
                        importances_output = gr.Label(label="Top 10 Feature Importances")

                predict_btn.click(
                    fn=forecast_interface,
                    inputs=[
                        country_dropdown,
                        zone_input,
                        target_radio,
                        model_dropdown,
                        date_input,
                        history_slider,
                        cal_check,
                    ],
                    outputs=[plot_output, metrics_output, importances_output],
                )

            with gr.Tab("Market Seasonality & Heatmap"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ana_country = gr.Dropdown(
                            choices=[(label, code) for label, code in COUNTRY_OPTIONS],
                            value="ES",
                            label="European Market",
                        )
                        ana_zone = gr.Textbox(
                            value="", label="Bidding Zone",
                            info="Required when a country has multiple zones",
                        )
                        ana_target = gr.Radio(
                            choices=["Electricity Demand (MW)", "Day-Ahead Price (EUR/MWh)"],
                            value="Electricity Demand (MW)",
                            label="Target Variable",
                        )
                        ana_btn = gr.Button("Compute Seasonality Profile", variant="primary")
                    with gr.Column(scale=3):
                        heatmap_output = gr.Plot(label="Seasonality Profile Heatmap")

                ana_btn.click(
                    fn=analytics_interface,
                    inputs=[ana_country, ana_zone, ana_target],
                    outputs=[heatmap_output],
                )

            with gr.Tab("Model Methodology & Conformal Calibration"):
                gr.Markdown(
                    """
                    ### Multi-Model Forecasting Strategies

                    1. **Dual-Horizon Ensemble (`dual_horizon`)**:
                       - **Short-Horizon (1-6 Hours)**: Captures immediate grid inertia,
                         high-frequency autoregressive momentum, and 4-step rolling volatility.
                       - **Long-Horizon (24-48 Hours)**: Captures diurnal load cycles
                         (`lag_24h`, `lag_48h`, `lag_7d`), calendar harmonics, and holiday shifts.
                       - **Dynamic Horizon Blending**: Smooth linear decay transition from
                         short-term to long-term forecaster across lead time steps.

                    2. **Stacked Multi-Model Ensemble (`stacked`)**:
                       - Blends LightGBM, XGBoost, and CatBoost with fixed weights.

                    3. **Split-Conformal Prediction Calibration**:
                       - Calibration targets 80% P10-P90 coverage. Measured coverage is
                         displayed with each evaluation.
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_port=7860)
