from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import yaml
from dash import Dash, Input, Output, dcc, html, dash_table

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://gauge24h-api-360668288184.europe-west3.run.app",
).rstrip("/")
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "15"))


def find_config_file(filename: str) -> Path:
    candidates = [
        Path(__file__).parent / filename,
        Path(__file__).parent / "config" / filename,
        Path(__file__).parent.parent / "config" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


THRESHOLDS_FILE = Path(
    os.getenv("GAUGE24H_THRESHOLDS_PATH", str(find_config_file("thresholds.yaml")))
)
SEGMENTS_FILE = Path(
    os.getenv("GAUGE24H_SEGMENTS_PATH", str(find_config_file("segments.yaml")))
)


def normalize_station_name(name: Any) -> str:
    return str(name).strip().upper().replace(" ", "-").replace("/", "-")


def load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    raw = config.get("low_water_thresholds_cm", {})
    return {
        normalize_station_name(station): float(value)
        for station, value in raw.items()
        if value is not None
    }


def load_segments(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    raw_segments = config.get("segments", {})
    raw_primary = config.get("primary_segment_by_station", {})
    segments = raw_segments if isinstance(raw_segments, dict) else {}
    primary = raw_primary if isinstance(raw_primary, dict) else {}
    return (
        {
            str(segment_id): segment
            for segment_id, segment in segments.items()
            if isinstance(segment, dict)
        },
        {
            normalize_station_name(station): str(segment_id)
            for station, segment_id in primary.items()
        },
    )


THRESHOLDS_CM = load_thresholds(THRESHOLDS_FILE)
SEGMENTS, PRIMARY_SEGMENT_BY_STATION = load_segments(SEGMENTS_FILE)

STATION_TO_SEGMENTS: dict[str, list[dict[str, Any]]] = {}
for segment_id, segment in SEGMENTS.items():
    decision_gauge = normalize_station_name(segment.get("decision_gauge", ""))
    gauges = {decision_gauge}
    gauges.update(
        normalize_station_name(gauge)
        for gauge in segment.get("support_gauges", [])
        if gauge
    )
    gauges.discard("")
    for gauge in gauges:
        STATION_TO_SEGMENTS.setdefault(gauge, []).append(
            {
                "segment_id": segment_id,
                "segment_label": segment.get("label", segment_id),
                "is_decision_gauge": gauge == decision_gauge,
            }
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


def api_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        params=params,
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def empty_message(message: str, color: str = COLORS["muted"]) -> html.Div:
    return html.Div(
        message,
        style={"padding": "24px", "textAlign": "center", "color": color},
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
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "10px",
            "padding": "18px",
            "minWidth": "190px",
            "flex": "1",
            "boxShadow": "0 1px 3px rgba(15, 23, 42, 0.06)",
        },
    )


def section_title(title: str, subtitle: str = "") -> html.Div:
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


def classify_risk(station_name: Any, prediction: Any) -> tuple[str, str]:
    try:
        value = float(prediction)
    except (TypeError, ValueError):
        return "Unknown", COLORS["gray"]
    threshold = THRESHOLDS_CM.get(normalize_station_name(station_name))
    if threshold is None:
        return "Unclassified", COLORS["gray"]
    if value <= threshold:
        return "High", COLORS["red"]
    if value <= threshold + 20:
        return "Elevated", COLORS["yellow"]
    return "Normal", COLORS["green"]


def station_segment_entries(
    station_name: Any,
) -> list[dict[str, Any]]:
    return STATION_TO_SEGMENTS.get(
        normalize_station_name(station_name),
        [],
    )


def primary_segment(
    station_name: Any,
) -> str | None:
    station_key = normalize_station_name(
        station_name
    )

    if station_key in PRIMARY_SEGMENT_BY_STATION:
        return PRIMARY_SEGMENT_BY_STATION[station_key]

    entries = station_segment_entries(station_name)

    return (
        entries[0]["segment_id"]
        if entries
        else None
    )


def record_segment_ids(
    record: dict[str, Any],
) -> list[str]:
    api_values = record.get("segment_ids")

    if isinstance(api_values, list) and api_values:
        return [
            str(segment_id)
            for segment_id in api_values
        ]

    return [
        entry["segment_id"]
        for entry in station_segment_entries(
            record.get("station_name")
        )
    ]


def format_timestamp(value: Any) -> str:
    if value is None:
        return "—"
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return str(value) if pd.isna(parsed) else parsed.strftime("%Y-%m-%d %H:%M UTC")


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def risk_distance(record: dict[str, Any]) -> float | None:
    threshold = THRESHOLDS_CM.get(normalize_station_name(record.get("station_name")))
    try:
        prediction = float(record.get("prediction"))
    except (TypeError, ValueError):
        return None
    if threshold is None:
        return None
    return prediction - threshold


def build_risk_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        station = record.get("station_name", "Unknown")
        risk, _ = classify_risk(station, record.get("prediction"))
        threshold = THRESHOLDS_CM.get(normalize_station_name(station))
        rows.append(
            {
                "Station": station,
                "Segment": primary_segment(station) or "—",
                "Risk": risk,
                "Prediction (cm)": format_number(record.get("prediction")),
                "Threshold (cm)": format_number(threshold),
                "Margin (cm)": format_number(risk_distance(record)),
                "Forecast time (UTC)": format_timestamp(record.get("forecast_timestamp_utc")),
                "Issued (UTC)": format_timestamp(record.get("prediction_ready_utc")),
                "Model version": record.get("model_version", "—"),
                "Actual": format_number(record.get("actual_if_available")) if record.get("actual_available_now", False) else "Pending",
            }
        )
    return rows


def build_event_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for record in records:
        distance = risk_distance(record)
        if distance is None:
            continue
        risk, _ = classify_risk(record.get("station_name"), record.get("prediction"))
        if risk not in {"High", "Elevated"}:
            continue
        station = record.get("station_name", "Unknown")
        ranked.append(
            {
                "Station": station,
                "Segment": primary_segment(station) or "—",
                "Risk": risk,
                "Prediction (cm)": format_number(record.get("prediction")),
                "Threshold (cm)": format_number(
                    THRESHOLDS_CM.get(normalize_station_name(station))
                ),
                "Margin (cm)": format_number(distance),
                "Decision gauge": "Yes"
                if any(
                    entry["is_decision_gauge"]
                    for entry in station_segment_entries(station)
                )
                else "No",
                "Forecast time (UTC)": format_timestamp(
                    record.get("forecast_timestamp_utc")
                ),
                "Model version": record.get("model_version", "—"),
            }
        )
    ranked.sort(
        key=lambda row: float(row["Margin (cm)"])
        if row["Margin (cm)"] != "—"
        else float("inf")
    )
    for index, row in enumerate(ranked, start=1):
        row["Rank"] = index
    return ranked


def build_segment_cards(records: list[dict[str, Any]]) -> list[html.Div]:
    cards: list[html.Div] = []
    for segment_id, segment in SEGMENTS.items():
        members = [
            record
            for record in records
            if segment_id in (record.get("segment_ids") or [])
        ]
        events = build_event_rows(members)
        worst = events[0] if events else None
        cards.append(
            metric_card(
                segment.get("label", segment_id),
                str(len(events)),
                f"Decision: {segment.get('decision_gauge', '—')} | Worst: {worst['Station'] if worst else 'None'}",
                COLORS["red"]
                if events and events[0]["Risk"] == "High"
                else COLORS["yellow"]
                if events
                else COLORS["green"],
            )
        )
    return cards


def build_map_figure(
    records: list[dict[str, Any]],
    station_metadata: list[dict[str, Any]],
    selected_segment: str | None,
) -> go.Figure:
    prediction_by_station = {
        normalize_station_name(record.get("station_name")): record
        for record in records
    }
    selected = None if selected_segment in (None, "all") else selected_segment
    lons: list[Any] = []
    lats: list[Any] = []
    texts: list[str] = []
    colors: list[str] = []
    sizes: list[int] = []

    for station in station_metadata:
        name = station.get("station_name")
        ids = station.get("segment_ids") or [
            entry["segment_id"] for entry in station_segment_entries(name)
        ]
        if selected and selected not in ids:
            continue
        if station.get("longitude") is None or station.get("latitude") is None:
            continue
        record = prediction_by_station.get(normalize_station_name(name), {})
        risk, color = classify_risk(name, record.get("prediction"))
        lons.append(station["longitude"])
        lats.append(station["latitude"])
        texts.append(
            f"{name}<br>Risk: {risk}<br>Prediction: {format_number(record.get('prediction'))} cm"
        )
        colors.append(color)
        sizes.append(
            16
            if any(
                entry["is_decision_gauge"] for entry in station_segment_entries(name)
            )
            else 12
        )

    figure = go.Figure()
    if lons:
        figure.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                text=texts,
                mode="markers",
                marker={"size": sizes, "color": colors, "line": {"width": 0.5, "color": "#111827"}},
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure.update_geos(
        scope="europe",
        showcountries=True,
        showland=True,
        landcolor="#f8fafc",
        fitbounds="locations",
    )
    figure.update_layout(
        title="Corridor risk map" if not selected else f"Corridor risk map — {selected}",
        template="plotly_white",
        height=520,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    return figure


def build_history_figure(
    records: list[dict[str, Any]],
    station_name: str | None,
    days: int,
) -> go.Figure:
    figure = go.Figure()
    if not records or not station_name:
        figure.update_layout(
            title="No forecast history available",
            template="plotly_white",
            height=480,
        )
        return figure

    frame = pd.DataFrame(records)
    frame = frame[frame["station_name"].astype(str) == station_name].copy()
    frame["forecast_timestamp_utc"] = pd.to_datetime(
        frame["forecast_timestamp_utc"], utc=True, errors="coerce"
    )
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.dropna(subset=["forecast_timestamp_utc", "prediction"]).sort_values(
        "forecast_timestamp_utc"
    )
    if frame.empty:
        figure.update_layout(
            title="No forecast history available",
            template="plotly_white",
            height=480,
        )
        return figure

    figure.add_trace(
        go.Scatter(
            x=frame["forecast_timestamp_utc"],
            y=frame["prediction"],
            mode="lines+markers",
            name="Prediction",
            line={"color": COLORS["blue"], "width": 3},
            marker={"size": 7},
            hovertemplate="Forecast: %{x|%Y-%m-%d %H:%M UTC}<br>Prediction: %{y:.2f} cm<extra></extra>",
        )
    )

    if "actual_if_available" in frame.columns:
        frame["actual_if_available"] = pd.to_numeric(
            frame["actual_if_available"], errors="coerce"
        )
        actual = frame.dropna(subset=["actual_if_available"])
        if not actual.empty:
            figure.add_trace(
                go.Scatter(
                    x=actual["forecast_timestamp_utc"],
                    y=actual["actual_if_available"],
                    mode="lines+markers",
                    name="Actual",
                    line={"color": COLORS["green"], "width": 2},
                    marker={"size": 6},
                    hovertemplate="Time: %{x|%Y-%m-%d %H:%M UTC}<br>Actual: %{y:.2f} cm<extra></extra>",
                )
            )

    threshold = THRESHOLDS_CM.get(normalize_station_name(station_name))
    if threshold is not None:
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=COLORS["red"],
            annotation_text=f"Low-water threshold: {threshold:.0f} cm",
            annotation_position="top left",
        )

    figure.update_layout(
        title=f"{station_name} — {days}-day forecast history",
        template="plotly_white",
        height=480,
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis_title="Forecast target time (UTC)",
        yaxis_title="Gauge level (cm)",
    )
    return figure


app.layout = html.Div(
    [
        dcc.Interval(id="refresh-interval", interval=5 * 60 * 1000, n_intervals=0),
        dcc.Download(id="download-risk"),
        dcc.Download(id="download-evaluations"),
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
                            "Operational 24-hour gauge forecast dashboard",
                            style={
                                "marginTop": "6px",
                                "color": "rgba(255,255,255,0.8)",
                                "fontSize": "14px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    id="dashboard-refresh-status",
                    style={
                        "textAlign": "right",
                        "color": "rgba(255,255,255,0.85)",
                        "fontSize": "12px",
                    },
                ),
            ],
            style={
                "background": "linear-gradient(135deg, #123b68, #2563eb)",
                "padding": "24px 32px",
                "display": "flex",
                "justifyContent": "space-between",
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
                        "marginBottom": "20px",
                    },
                ),
                html.Div(
                    id="segment-cards",
                    style={
                        "display": "flex",
                        "gap": "14px",
                        "flexWrap": "wrap",
                        "marginBottom": "24px",
                    },
                ),
                html.Div(
                    id="system-status-banner",
                    style={"marginBottom": "24px"},
                ),
                html.Div(
                    [
                        section_title(
                            "Corridor map",
                            "Segment-aware overview of current production risk.",
                        ),
                        dcc.Dropdown(
                            id="segment-dropdown",
                            options=[{"label": "All segments", "value": "all"}]
                            + [
                                {
                                    "label": segment.get("label", segment_id),
                                    "value": segment_id,
                                }
                                for segment_id, segment in SEGMENTS.items()
                            ],
                            value="all",
                            clearable=False,
                            style={"marginBottom": "12px"},
                        ),
                        dcc.Graph(
                            id="corridor-map",
                            config={"displaylogo": False, "responsive": True},
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "10px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                ),
                html.Div(
                    [
                        section_title(
                            "Latest corridor forecasts",
                            "Current production predictions from the latest model run.",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Download current risk CSV",
                                    id="download-risk-button",
                                    style={"marginRight": "10px"},
                                ),
                                html.Button(
                                    "Download evaluation history CSV",
                                    id="download-evaluations-button",
                                ),
                            ],
                            style={"marginBottom": "12px"},
                        ),
                        html.Div(id="latest-predictions-table"),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "10px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                ),
                html.Div(
                    [
                        section_title(
                            "Ranked low-water events",
                            "Highest-priority current risks first.",
                        ),
                        html.Div(id="event-ranking-table"),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "10px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                ),
                html.Div(
                    [
                        section_title(
                            "Station forecast analysis",
                            "Select a station to inspect its forecast history.",
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="station-dropdown",
                                    placeholder="Select a station",
                                    clearable=False,
                                    style={"flex": "2"},
                                ),
                                dcc.Dropdown(
                                    id="history-days",
                                    options=[
                                        {"label": "7 days", "value": 7},
                                        {"label": "14 days", "value": 14},
                                        {"label": "30 days", "value": 30},
                                        {"label": "90 days", "value": 90},
                                    ],
                                    value=7,
                                    clearable=False,
                                    style={"flex": "1"},
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
                            config={"displaylogo": False, "responsive": True},
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "10px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                ),
                html.Div(
                    [
                        section_title(
                            "Data and model status",
                            "Operational status of the prediction service.",
                        ),
                        html.Pre(
                            id="system-status-details",
                            style={
                                "backgroundColor": "#f8fafc",
                                "border": f"1px solid {COLORS['border']}",
                                "borderRadius": "8px",
                                "padding": "14px",
                                "fontSize": "12px",
                                "overflowX": "auto",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
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
        "fontFamily": "Inter, Arial, sans-serif",
        "color": COLORS["text"],
    },
)


@app.callback(
    [
        Output("latest-predictions-table", "children"),
        Output("event-ranking-table", "children"),
        Output("station-dropdown", "options"),
        Output("station-dropdown", "value"),
        Output("metric-cards", "children"),
        Output("segment-cards", "children"),
        Output("system-status-banner", "children"),
        Output("system-status-details", "children"),
        Output("dashboard-refresh-status", "children"),
        Output("corridor-map", "figure"),
    ],
    [Input("refresh-interval", "n_intervals"), Input("segment-dropdown", "value")],
)
def refresh_dashboard(n_intervals: int, segment_value: str | None):
    del n_intervals
    try:
        latest_records = api_get("/predictions/latest")
        station_metadata = api_get("/metadata/stations")
        system_status = api_get("/system/status")
    except Exception as exc:
        error = empty_message(f"Dashboard API unavailable: {exc}", COLORS["red"])
        return (
            error,
            error,
            [],
            None,
            [],
            [],
            error,
            str(exc),
            "Refresh failed",
            go.Figure(),
        )

    selected_segment = None if segment_value in (None, "all") else segment_value
    filtered_records = [
        record
        for record in latest_records
        if selected_segment is None or selected_segment in record_segment_ids(record)
    ]

    if not filtered_records:
        risk_table = empty_message(
            "No production predictions available for this segment."
        )
        event_table = empty_message("No current events to rank.")
        station_options = []
        station_value = None
    else:
        risk_rows = build_risk_rows(filtered_records)
        risk_table = dash_table.DataTable(
            data=risk_rows,
            columns=[{"name": column, "id": column} for column in risk_rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#eef2f7", "fontWeight": "700"},
            style_cell={
                "padding": "11px",
                "fontSize": "13px",
                "textAlign": "left",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": '{Risk} = "High"', "column_id": "Risk"},
                    "color": COLORS["red"],
                    "fontWeight": "700",
                },
                {
                    "if": {"filter_query": '{Risk} = "Elevated"', "column_id": "Risk"},
                    "color": COLORS["yellow"],
                    "fontWeight": "700",
                },
                {
                    "if": {"filter_query": '{Risk} = "Normal"', "column_id": "Risk"},
                    "color": COLORS["green"],
                },
            ],
            page_size=25,
            sort_action="native",
            filter_action="native",
        )

        event_rows = build_event_rows(filtered_records)
        event_table = (
            empty_message("No high or elevated events at the moment.", COLORS["green"])
            if not event_rows
            else dash_table.DataTable(
                data=event_rows,
                columns=[{"name": column, "id": column} for column in event_rows[0].keys()],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#eef2f7", "fontWeight": "700"},
                style_cell={
                    "padding": "11px",
                    "fontSize": "13px",
                    "textAlign": "left",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{Risk} = "High"', "column_id": "Risk"},
                        "color": COLORS["red"],
                        "fontWeight": "700",
                    },
                    {
                        "if": {
                            "filter_query": '{Risk} = "Elevated"',
                            "column_id": "Risk",
                        },
                        "color": COLORS["yellow"],
                        "fontWeight": "700",
                    },
                ],
                page_size=15,
                sort_action="native",
            )
        )

        stations = sorted(
            {
                str(record.get("station_name"))
                for record in filtered_records
                if record.get("station_name")
            }
        )
        station_options = [{"label": station, "value": station} for station in stations]
        station_value = stations[0] if stations else None

    high_risk = sum(
        classify_risk(record.get("station_name"), record.get("prediction"))[0] == "High"
        for record in filtered_records
    )
    elevated_risk = sum(
        classify_risk(record.get("station_name"), record.get("prediction"))[0]
        == "Elevated"
        for record in filtered_records
    )
    data_quality = system_status.get("data_quality_status", "unknown")
    api_status = system_status.get("status", "unknown")
    prediction_status = system_status.get("prediction_status", {})

    cards = [
        metric_card(
            "Stations forecast",
            str(
                len(
                    {
                        record.get("station_name")
                        for record in filtered_records
                        if record.get("station_name")
                    }
                )
            ),
            "Selected segment" if selected_segment else "Latest production run",
        ),
        metric_card("Predictions", str(len(filtered_records)), "Rows in selected view"),
        metric_card(
            "High risk",
            str(high_risk),
            "Below configured threshold",
            COLORS["red"] if high_risk else COLORS["green"],
        ),
        metric_card(
            "Elevated risk",
            str(elevated_risk),
            "Near configured threshold",
            COLORS["yellow"] if elevated_risk else COLORS["green"],
        ),
        metric_card(
            "Data quality",
            str(data_quality).upper(),
            "Latest quality status",
            COLORS["green"] if data_quality == "pass" else COLORS["red"],
        ),
    ]

    segment_cards = build_segment_cards(filtered_records)
    banner_color = (
        COLORS["green"] if api_status == "ok" and data_quality == "pass" else COLORS["yellow"]
    )
    banner = html.Div(
        [
            html.Strong(f"System status: {str(api_status).upper()}"),
            html.Span(
                f" | Data quality: {str(data_quality).upper()}",
                style={"marginLeft": "18px"},
            ),
            html.Span(
                f" | Latest model: {prediction_status.get('model_version', 'unknown')}",
                style={"marginLeft": "18px"},
            ),
        ],
        style={
            "backgroundColor": f"{banner_color}18",
            "border": f"1px solid {banner_color}66",
            "borderRadius": "8px",
            "padding": "13px 16px",
            "color": banner_color,
            "fontSize": "13px",
        },
    )

    details = {
        "latest_prediction_ready_utc": prediction_status.get(
            "latest_prediction_ready_utc"
        ),
        "latest_forecast_timestamp_utc": prediction_status.get(
            "latest_forecast_timestamp_utc"
        ),
        "run_id": prediction_status.get("run_id"),
        "model_version": prediction_status.get("model_version"),
        "prediction_rows": prediction_status.get("prediction_rows"),
        "station_count": prediction_status.get("station_count"),
        "data_quality_status": data_quality,
    }
    status_details = str(details).replace("'", '"')
    refresh_status = (
        f"Last dashboard refresh: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}"
    )
    map_figure = build_map_figure(filtered_records, station_metadata, selected_segment)

    return (
        risk_table,
        event_table,
        station_options,
        station_value,
        cards,
        segment_cards,
        banner,
        status_details,
        refresh_status,
        map_figure,
    )


@app.callback(
    Output("station-forecast-chart", "figure"),
    [Input("refresh-interval", "n_intervals"), Input("station-dropdown", "value"), Input("history-days", "value")],
)
def update_station_chart(n_intervals: int, station_name: str | None, days: int):
    del n_intervals
    if not station_name:
        return build_history_figure([], station_name, days)
    try:
        records = api_get(
            "/predictions/history",
            params={"days": days, "station": station_name},
        )
    except Exception as exc:
        figure = go.Figure()
        figure.update_layout(
            title=f"Forecast history unavailable: {exc}",
            template="plotly_white",
            height=480,
        )
        return figure
    return build_history_figure(records, station_name, days)


@app.callback(
    Output("download-risk", "data"),
    Input("download-risk-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_current_risk(n_clicks: int):
    del n_clicks
    records = api_get("/predictions/latest")
    rows = []
    for record in records:
        station = record.get("station_name")
        rows.append(
            {
                "station_name": station,
                "primary_segment": primary_segment(station),
                "risk": classify_risk(station, record.get("prediction"))[0],
                "prediction_cm": record.get("prediction"),
                "threshold_cm": THRESHOLDS_CM.get(
                    normalize_station_name(station)
                ),
                "risk_distance_cm": risk_distance(record),
                "forecast_timestamp_utc": record.get("forecast_timestamp_utc"),
                "prediction_ready_utc": record.get("prediction_ready_utc"),
                "model_version": record.get("model_version"),
            }
        )
    return dcc.send_data_frame(
        pd.DataFrame(rows).to_csv,
        "current_risk.csv",
        index=False,
    )


@app.callback(
    Output("download-evaluations", "data"),
    Input("download-evaluations-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_evaluations(n_clicks: int):
    del n_clicks
    records = api_get("/evaluations/history", params={"days": 365})
    return dcc.send_data_frame(
        pd.DataFrame(records).to_csv,
        "evaluation_history.csv",
        index=False,
    )


@app.server.route("/debug")
def debug_endpoint():
    try:
        return api_get("/system/status")
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "api_base_url": API_BASE_URL,
        }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8050")),
        debug=False,
    )