from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
import requests
import yaml

from dash import (
    Dash,
    Input,
    Output,
    dcc,
    html,
    dash_table,
)


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://gauge24h-api-360668288184.europe-west3.run.app",
).rstrip("/")

API_TIMEOUT_SECONDS = int(
    os.getenv(
        "API_TIMEOUT_SECONDS",
        "15",
    )
)

THRESHOLDS_FILE = (
    Path(__file__).parent
    / "threshold.yaml"
)


def normalize_station_name(
    name: str,
) -> str:
    return (
        str(name)
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("/", "")
    )


def load_thresholds(
    path: Path,
) -> dict[str, float]:
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle) or {}

    raw_thresholds = config.get(
        "low_water_thresholds_cm",
        {},
    )

    thresholds: dict[str, float] = {}

    for station, value in raw_thresholds.items():
        thresholds[
            normalize_station_name(station)
        ] = float(value)

    return thresholds


THRESHOLDS_CM = load_thresholds(
    THRESHOLDS_FILE
)


app = Dash(
    __name__,
    title="Rhine Corridor Water Risk",
    suppress_callback_exceptions=True,
)

server = app.server


COLORS = {
    "background": "#f5f7fb",
    "card": "#ffffff",
    "text": "#1f2937",
    "muted": "#6b7280",
    "border": "#dbe2ea",
    "blue": "#2563eb",
    "green": "#15803d",
    "yellow": "#ca8a04",
    "red": "#dc2626",
    "gray": "#64748b",
}


def api_get(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> Any:
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        params=params,
        timeout=API_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    return response.json()


def empty_message(
    message: str,
    color: str = COLORS["muted"],
) -> html.Div:
    return html.Div(
        message,
        style={
            "padding": "24px",
            "textAlign": "center",
            "color": color,
        },
    )


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = COLORS["blue"],
) -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "color": COLORS["muted"],
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "25px",
                    "fontWeight": "700",
                    "color": color,
                },
            ),
            html.Div(
                subtitle,
                style={
                    "fontSize": "12px",
                    "color": COLORS["muted"],
                    "marginTop": "5px",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": (
                f"1px solid {COLORS['border']}"
            ),
            "borderRadius": "10px",
            "padding": "18px",
            "minWidth": "190px",
            "flex": "1",
            "boxShadow": (
                "0 1px 3px rgba(15, 23, 42, 0.06)"
            ),
        },
    )


def section_title(
    title: str,
    subtitle: str = "",
) -> html.Div:
    return html.Div(
        [
            html.H2(
                title,
                style={
                    "fontSize": "20px",
                    "marginBottom": "4px",
                    "color": COLORS["text"],
                },
            ),
            html.Div(
                subtitle,
                style={
                    "fontSize": "13px",
                    "color": COLORS["muted"],
                    "marginBottom": "16px",
                },
            ),
        ]
    )


def classify_risk(
    station_name: str,
    prediction: Any,
) -> tuple[str, str]:
    try:
        prediction_value = float(prediction)
    except (
        TypeError,
        ValueError,
    ):
        return "Unknown", COLORS["gray"]

    threshold = THRESHOLDS_CM.get(
        normalize_station_name(station_name)
    )

    if threshold is None:
        return "Unclassified", COLORS["gray"]

    if prediction_value <= threshold:
        return "High", COLORS["red"]

    if prediction_value <= threshold + 20:
        return "Elevated", COLORS["yellow"]

    return "Normal", COLORS["green"]


def format_timestamp(
    value: Any,
) -> str:
    if value is None:
        return "—"

    parsed = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return str(value)

    return parsed.strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}"
    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def build_risk_table(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        station = record.get(
            "station_name",
            "Unknown",
        )

        risk, _ = classify_risk(
            station,
            record.get("prediction"),
        )

        rows.append(
            {
                "Station": station,
                "Risk": risk,
                "Prediction (cm)": format_number(
                    record.get("prediction")
                ),
                "Forecast time (UTC)": (
                    format_timestamp(
                        record.get(
                            "forecast_timestamp_utc"
                        )
                    )
                ),
                "Issued (UTC)": (
                    format_timestamp(
                        record.get(
                            "prediction_ready_utc"
                        )
                    )
                ),
                "Model version": record.get(
                    "model_version",
                    "—",
                ),
                "Actual": (
                    format_number(
                        record.get(
                            "actual_if_available"
                        )
                    )
                    if record.get(
                        "actual_available_now",
                        False,
                    )
                    else "Pending"
                ),
            }
        )

    return rows


def build_figure(
    records: list[dict[str, Any]],
    station_name: str | None,
    days: int,
) -> go.Figure:
    figure = go.Figure()

    if not records:
        figure.update_layout(
            title="No forecast history available",
            template="plotly_white",
            height=480,
        )
        return figure

    df = pd.DataFrame(records)

    if station_name:
        df = df[
            df["station_name"].astype(str)
            == station_name
        ].copy()

    if df.empty:
        figure.update_layout(
            title="No data for selected station",
            template="plotly_white",
            height=480,
        )
        return figure

    df["forecast_timestamp_utc"] = (
        pd.to_datetime(
            df["forecast_timestamp_utc"],
            utc=True,
            errors="coerce",
        )
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "forecast_timestamp_utc",
            "prediction",
        ]
    ).sort_values(
        "forecast_timestamp_utc"
    )

    station = str(
        df["station_name"].iloc[0]
    )

    figure.add_trace(
        go.Scatter(
            x=df["forecast_timestamp_utc"],
            y=df["prediction"],
            mode="lines+markers",
            name="Prediction",
            line={
                "color": COLORS["blue"],
                "width": 3,
            },
            marker={
                "size": 7,
            },
            hovertemplate=(
                "Forecast: %{x|%Y-%m-%d %H:%M UTC}"
                "<br>Prediction: %{y:.2f} cm"
                "<extra></extra>"
            ),
        )
    )

    if "actual_if_available" in df.columns:
        df["actual_if_available"] = (
            pd.to_numeric(
                df["actual_if_available"],
                errors="coerce",
            )
        )

        actual_df = df.dropna(
            subset=["actual_if_available"]
        )

        if not actual_df.empty:
            figure.add_trace(
                go.Scatter(
                    x=actual_df[
                        "forecast_timestamp_utc"
                    ],
                    y=actual_df[
                        "actual_if_available"
                    ],
                    mode="lines+markers",
                    name="Actual",
                    line={
                        "color": COLORS["green"],
                        "width": 2,
                    },
                    marker={
                        "size": 6,
                    },
                    hovertemplate=(
                        "Time: %{x|%Y-%m-%d %H:%M UTC}"
                        "<br>Actual: %{y:.2f} cm"
                        "<extra></extra>"
                    ),
                )
            )

    threshold = THRESHOLDS_CM.get(
        normalize_station_name(station)
    )

    if threshold is not None:
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=COLORS["red"],
            annotation_text=(
                f"Low-water threshold: "
                f"{threshold:.0f} cm"
            ),
            annotation_position="top left",
        )

    figure.update_layout(
        title=(
            f"{station} — {days}-day forecast history"
        ),
        template="plotly_white",
        height=480,
        margin={
            "l": 50,
            "r": 30,
            "t": 70,
            "b": 50,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
        },
        xaxis_title="Forecast target time (UTC)",
        yaxis_title="Gauge level (cm)",
    )

    return figure


app.layout = html.Div(
    [
        dcc.Interval(
            id="refresh-interval",
            interval=5 * 60 * 1000,
            n_intervals=0,
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "Rhine Corridor Water Risk",
                            style={
                                "margin": "0",
                                "fontSize": "30px",
                                "color": "#ffffff",
                            },
                        ),
                        html.Div(
                            (
                                "Operational 24-hour gauge "
                                "forecast dashboard"
                            ),
                            style={
                                "marginTop": "6px",
                                "color": (
                                    "rgba(255,255,255,0.8)"
                                ),
                                "fontSize": "14px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    id="dashboard-refresh-status",
                    style={
                        "textAlign": "right",
                        "color": (
                            "rgba(255,255,255,0.85)"
                        ),
                        "fontSize": "12px",
                    },
                ),
            ],
            style={
                "background": (
                    "linear-gradient("
                    "135deg, #123b68, #2563eb"
                    ")"
                ),
                "padding": "24px 32px",
                "display": "flex",
                "justifyContent": (
                    "space-between"
                ),
                "alignItems": "center",
            },
        ),

        html.Div(
            [
                html.Div(
                    id="metric-cards",
                    style={
                        "display": "flex",
                        "gap": "14px",
                        "flexWrap": "wrap",
                        "marginBottom": "24px",
                    },
                ),

                html.Div(
                    id="system-status-banner",
                    style={
                        "marginBottom": "24px",
                    },
                ),

                html.Div(
                    [
                        section_title(
                            "Latest corridor forecasts",
                            (
                                "Current production predictions "
                                "from the latest model run."
                            ),
                        ),
                        html.Div(
                            id="latest-predictions-table"
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": (
                            f"1px solid {COLORS['border']}"
                        ),
                        "borderRadius": "10px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                ),

                html.Div(
                    [
                        section_title(
                            "Station forecast analysis",
                            (
                                "Select a station to inspect its "
                                "forecast history."
                            ),
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Station",
                                            style={
                                                "fontWeight": (
                                                    "600"
                                                ),
                                                "fontSize": (
                                                    "13px"
                                                ),
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id=(
                                                "station-dropdown"
                                            ),
                                            placeholder=(
                                                "Select a station"
                                            ),
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "2",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "History window",
                                            style={
                                                "fontWeight": (
                                                    "600"
                                                ),
                                                "fontSize": (
                                                    "13px"
                                                ),
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id=(
                                                "history-days"
                                            ),
                                            options=[
                                                {
                                                    "label": (
                                                        "7 days"
                                                    ),
                                                    "value": 7,
                                                },
                                                {
                                                    "label": (
                                                        "14 days"
                                                    ),
                                                    "value": 14,
                                                },
                                                {
                                                    "label": (
                                                        "30 days"
                                                    ),
                                                    "value": 30,
                                                },
                                                {
                                                    "label": (
                                                        "90 days"
                                                    ),
                                                    "value": 90,
                                                },
                                            ],
                                            value=7,
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "16px",
                                "marginBottom": "16px",
                                "flexWrap": "wrap",
                            },
                        ),
                        dcc.Graph(
                            id="station-forecast-chart",
                            config={
                                "displaylogo": False,
                                "responsive": True,
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": (
                            f"1px solid {COLORS['border']}"
                        ),
                        "borderRadius": "10px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                ),

                html.Div(
                    [
                        section_title(
                            "Data and model status",
                            (
                                "Operational status of the "
                                "prediction service."
                            ),
                        ),
                        html.Pre(
                            id="system-status-details",
                            style={
                                "backgroundColor": (
                                    "#f8fafc"
                                ),
                                "border": (
                                    f"1px solid "
                                    f"{COLORS['border']}"
                                ),
                                "borderRadius": "8px",
                                "padding": "14px",
                                "fontSize": "12px",
                                "overflowX": "auto",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": (
                            f"1px solid {COLORS['border']}"
                        ),
                        "borderRadius": "10px",
                        "padding": "20px",
                    },
                ),
            ],
            style={
                "maxWidth": "1500px",
                "margin": "0 auto",
                "padding": "24px 32px 40px",
            },
        ),
    ],
    style={
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "fontFamily": (
            "Inter, Arial, sans-serif"
        ),
        "color": COLORS["text"],
    },
)


@app.callback(
    [
        Output(
            "latest-predictions-table",
            "children",
        ),
        Output(
            "station-dropdown",
            "options",
        ),
        Output(
            "station-dropdown",
            "value",
        ),
        Output(
            "metric-cards",
            "children",
        ),
        Output(
            "system-status-banner",
            "children",
        ),
        Output(
            "system-status-details",
            "children",
        ),
        Output(
            "dashboard-refresh-status",
            "children",
        ),
    ],
    Input(
        "refresh-interval",
        "n_intervals",
    ),
)
def refresh_dashboard(
    n_intervals: int,
):
    del n_intervals

    try:
        latest_records = api_get(
            "/predictions/latest"
        )
    except Exception as exc:
        error = empty_message(
            f"Prediction API unavailable: {exc}",
            COLORS["red"],
        )

        return (
            error,
            [],
            None,
            [
                metric_card(
                    "API status",
                    "Offline",
                    color=COLORS["red"],
                )
            ],
            error,
            "Unable to load system status.",
            "Refresh failed",
        )

    try:
        system_status = api_get(
            "/system/status"
        )
    except Exception as exc:
        system_status = {
            "status": "degraded",
            "error": str(exc),
        }

    if not latest_records:
        table = empty_message(
            "No production predictions available."
        )
        station_options = []
        station_value = None
    else:
        table_records = build_risk_table(
            latest_records
        )

        table = dash_table.DataTable(
            data=table_records,
            columns=[
                {
                    "name": column,
                    "id": column,
                }
                for column in [
                    "Station",
                    "Risk",
                    "Prediction (cm)",
                    "Forecast time (UTC)",
                    "Issued (UTC)",
                    "Model version",
                    "Actual",
                ]
            ],
            style_table={
                "overflowX": "auto",
            },
            style_header={
                "backgroundColor": "#eef2f7",
                "fontWeight": "700",
                "border": (
                    f"1px solid {COLORS['border']}"
                ),
            },
            style_cell={
                "padding": "11px",
                "fontSize": "13px",
                "textAlign": "left",
                "border": (
                    f"1px solid {COLORS['border']}"
                ),
            },
            style_data_conditional=[
                {
                    "if": {
                        "filter_query": (
                            '{Risk} = "High"'
                        ),
                        "column_id": "Risk",
                    },
                    "color": COLORS["red"],
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "filter_query": (
                            '{Risk} = "Elevated"'
                        ),
                        "column_id": "Risk",
                    },
                    "color": COLORS["yellow"],
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "filter_query": (
                            '{Risk} = "Normal"'
                        ),
                        "column_id": "Risk",
                    },
                    "color": COLORS["green"],
                },
            ],
            page_size=25,
            sort_action="native",
            filter_action="native",
        )

        stations = sorted(
            {
                str(
                    record.get(
                        "station_name"
                    )
                )
                for record in latest_records
                if record.get("station_name")
            }
        )

        station_options = [
            {
                "label": station,
                "value": station,
            }
            for station in stations
        ]

        station_value = (
            stations[0]
            if stations
            else None
        )

    prediction_count = len(
        latest_records
    )

    station_count = len(
        {
            record.get("station_name")
            for record in latest_records
            if record.get("station_name")
        }
    )

    high_risk_count = 0
    elevated_risk_count = 0

    for record in latest_records:
        risk, _ = classify_risk(
            record.get(
                "station_name",
                "Unknown",
            ),
            record.get("prediction"),
        )

        if risk == "High":
            high_risk_count += 1

        elif risk == "Elevated":
            elevated_risk_count += 1

    prediction_status = system_status.get(
        "prediction_status",
        {},
    )

    data_quality_status = system_status.get(
        "data_quality_status",
        "unknown",
    )

    api_status = system_status.get(
        "status",
        "unknown",
    )

    metric_cards = [
        metric_card(
            "Stations forecast",
            str(station_count),
            "Latest production run",
            COLORS["blue"],
        ),
        metric_card(
            "Predictions",
            str(prediction_count),
            "Rows in latest run",
            COLORS["blue"],
        ),
        metric_card(
            "High risk",
            str(high_risk_count),
            "Below configured threshold",
            COLORS["red"]
            if high_risk_count
            else COLORS["green"],
        ),
        metric_card(
            "Elevated risk",
            str(elevated_risk_count),
            "Near configured threshold",
            COLORS["yellow"]
            if elevated_risk_count
            else COLORS["green"],
        ),
        metric_card(
            "Data quality",
            str(data_quality_status).upper(),
            "Latest quality status",
            COLORS["green"]
            if data_quality_status == "pass"
            else COLORS["red"],
        ),
    ]

    banner_color = (
        COLORS["green"]
        if (
            api_status == "ok"
            and data_quality_status == "pass"
        )
        else COLORS["yellow"]
    )

    status_banner = html.Div(
        [
            html.Strong(
                "System status: "
                f"{str(api_status).upper()}"
            ),
            html.Span(
                "  |  Data quality: "
                f"{str(data_quality_status).upper()}",
                style={
                    "marginLeft": "18px",
                },
            ),
            html.Span(
                "  |  Latest model: "
                f"{prediction_status.get(
                    'model_version',
                    'unknown',
                )}",
                style={
                    "marginLeft": "18px",
                },
            ),
        ],
        style={
            "backgroundColor": (
                f"{banner_color}18"
            ),
            "border": (
                f"1px solid {banner_color}66"
            ),
            "borderRadius": "8px",
            "padding": "13px 16px",
            "color": banner_color,
            "fontSize": "13px",
        },
    )

    details = {
        "latest_prediction_ready_utc": (
            prediction_status.get(
                "latest_prediction_ready_utc"
            )
        ),
        "latest_forecast_timestamp_utc": (
            prediction_status.get(
                "latest_forecast_timestamp_utc"
            )
        ),
        "run_id": prediction_status.get(
            "run_id"
        ),
        "model_version": prediction_status.get(
            "model_version"
        ),
        "prediction_rows": prediction_status.get(
            "prediction_rows"
        ),
        "station_count": prediction_status.get(
            "station_count"
        ),
        "data_quality_status": (
            data_quality_status
        ),
    }

    status_details = (
        str(details)
        .replace("'", '"')
    )

    refresh_status = (
        "Last dashboard refresh: "
        f"{pd.Timestamp.now(tz='UTC').strftime(
            '%Y-%m-%d %H:%M UTC'
        )}"
    )

    return (
        table,
        station_options,
        station_value,
        metric_cards,
        status_banner,
        status_details,
        refresh_status,
    )


@app.callback(
    Output(
        "station-forecast-chart",
        "figure",
    ),
    [
        Input(
            "refresh-interval",
            "n_intervals",
        ),
        Input(
            "station-dropdown",
            "value",
        ),
        Input(
            "history-days",
            "value",
        ),
    ],
)
def update_station_chart(
    n_intervals: int,
    station_name: str | None,
    days: int,
):
    del n_intervals

    if not station_name:
        return build_figure(
            [],
            station_name,
            days,
        )

    try:
        history_records = api_get(
            "/predictions/history",
            params={
                "days": days,
                "station": station_name,
            },
        )
    except Exception as exc:
        figure = go.Figure()
        figure.update_layout(
            title=(
                "Forecast history unavailable: "
                f"{exc}"
            ),
            template="plotly_white",
            height=480,
        )
        return figure

    return build_figure(
        history_records,
        station_name,
        days,
    )


@app.server.route("/debug")
def debug_endpoint():
    try:
        status = api_get(
            "/system/status"
        )
        return status
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "api_base_url": API_BASE_URL,
        }


if __name__== "__main__":
    port = int(
        os.getenv(
            "PORT",
            "8050",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )