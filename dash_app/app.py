from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
import requests
import yaml
from dash import Dash, Input, Output, dcc, html, dash_table
from dash_iconify import DashIconify


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://gauge24h-api-360668288184.europe-west3.run.app",
).rstrip("/")

API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "15"))
DISPLAY_MODEL_VERSION = os.getenv("DISPLAY_MODEL_VERSION", "Model 1.1")


def find_config_file(filename: str) -> Path:
    app_dir = Path(__file__).resolve().parent
    root = app_dir.parent

    candidates = [
        app_dir / filename,
        app_dir / "config" / filename,
        root / "config" / filename,
        root / filename,
    ]

    return next(
        (path for path in candidates if path.exists()),
        candidates[0],
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}

    return value if isinstance(value, dict) else {}


def normalize_station_name(name: Any) -> str:
    return str(name).strip().upper().replace(" ", "-").replace("/", "-")


THRESHOLDS_FILE = Path(
    os.getenv(
        "GAUGE24H_THRESHOLDS_PATH",
        str(find_config_file("thresholds.yaml")),
    )
)

SEGMENTS_FILE = Path(
    os.getenv(
        "GAUGE24H_SEGMENTS_PATH",
        str(find_config_file("segments.yaml")),
    )
)

OPERATIONAL_THRESHOLDS_FILE = Path(
    os.getenv(
        "GAUGE24H_OPERATIONAL_THRESHOLDS_PATH",
        str(find_config_file("operational_thresholds.yaml")),
    )
)

THRESHOLDS = {
    normalize_station_name(key): float(value)
    for key, value in load_yaml(THRESHOLDS_FILE)
    .get("low_water_thresholds_cm", {})
    .items()
    if value is not None
}

segment_config = load_yaml(SEGMENTS_FILE)
raw_segments = segment_config.get("segments", {})

SEGMENTS = (
    {
        str(key): value
        for key, value in raw_segments.items()
        if isinstance(value, dict)
    }
    if isinstance(raw_segments, dict)
    else {}
)

raw_primary = segment_config.get("primary_segment_by_station", {})

PRIMARY_SEGMENTS = (
    {
        normalize_station_name(key): str(value)
        for key, value in raw_primary.items()
    }
    if isinstance(raw_primary, dict)
    else {}
)

OPERATIONAL_CONFIG = load_yaml(
    OPERATIONAL_THRESHOLDS_FILE
).get("operational_thresholds", {})

if not isinstance(OPERATIONAL_CONFIG, dict):
    OPERATIONAL_CONFIG = {}


COLORS = {
    "background": "#090E17",
    "card": "#111A27",
    "text": "#F8FAFC",
    "muted": "#64748B",
    "border": "#1E293B",
    "blue": "#38BDF8",
    "cyan": "#2DD4BF",
    "green": "#10B981",
    "yellow": "#F59E0B",
    "orange": "#F97316",
    "red": "#EF4444",
    "darkred": "#991B1B",
    "gray": "#475569",
}


STATION_SEGMENTS: dict[str, list[dict[str, Any]]] = {}

for segment_id, segment in SEGMENTS.items():
    decision = normalize_station_name(
        segment.get("decision_gauge", "")
    )

    gauges = {decision} | {
        normalize_station_name(value)
        for value in segment.get("support_gauges", [])
        if value
    }

    gauges.discard("")

    for gauge in gauges:
        STATION_SEGMENTS.setdefault(gauge, []).append(
            {
                "segment_id": segment_id,
                "segment_label": segment.get(
                    "label",
                    segment_id,
                ),
                "is_decision_gauge": gauge == decision,
            }
        )


app = Dash(
    __name__,
    title="Rhine Corridor Forecaster",
    update_title=None,
    suppress_callback_exceptions=True,
)

server = app.server


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


def normalize_status(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def empty_message(
    message: str,
    detail: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Div(
                DashIconify(
                    icon="carbon:warning-square",
                    width=32,
                    color=COLORS["gray"],
                ),
                className="empty-icon",
            ),
            html.Div(message, className="empty-title"),
            html.Div(detail, className="empty-detail"),
        ],
        className="empty-state",
    )


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = COLORS["blue"],
    icon: str = "carbon:analytics",
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        DashIconify(
                            icon=icon,
                            width=16,
                            color=color,
                        ),
                        className="metric-icon",
                        style={
                            "backgroundColor": f"{color}1A"
                        },
                    ),
                    html.Span(
                        title,
                        className="metric-label",
                    ),
                ],
                className="metric-heading",
            ),
            html.Div(
                value,
                className="metric-value",
                style={"color": color},
            ),
            html.Div(
                subtitle,
                className="metric-detail",
            ),
        ],
        className="metric-card",
    )


def station_segments(name: Any) -> list[dict[str, Any]]:
    return STATION_SEGMENTS.get(
        normalize_station_name(name),
        [],
    )


def primary_segment_label(name: Any) -> str:
    key = normalize_station_name(name)

    segment_id = PRIMARY_SEGMENTS.get(key) or (
        station_segments(name)[0]["segment_id"]
        if station_segments(name)
        else None
    )

    return str(
        SEGMENTS.get(segment_id, {}).get(
            "label",
            segment_id or "—",
        )
    )


def record_segment_ids(record: dict[str, Any]) -> list[str]:
    values = record.get("segment_ids")

    if isinstance(values, list) and values:
        return [str(value) for value in values]

    return [
        item["segment_id"]
        for item in station_segments(record.get("station_name"))
    ]


def band_for_level(
    name: Any,
    value: Any,
) -> dict[str, Any]:
    try:
        level = float(value)
    except (TypeError, ValueError):
        return {
            "label": "Unknown",
            "severity": "unknown",
            "color": "gray",
            "risk": "unknown",
            "level": None,
        }

    config = (
        OPERATIONAL_CONFIG.get(normalize_station_name(name))
        or OPERATIONAL_CONFIG.get("DEFAULT")
    )

    if isinstance(config, dict):
        for band in config.get("bands", []):
            lower = band.get("min_cm")
            upper = band.get("max_cm")

            if (
                (lower is None or level >= float(lower))
                and (upper is None or level <= float(upper))
            ):
                severity = normalize_status(
                    band.get("severity")
                )

                risk = (
                    "normal"
                    if severity == "normal"
                    else "elevated"
                    if severity in {"warning", "elevated"}
                    else "high"
                )

                return {
                    "label": str(
                        band.get(
                            "label",
                            band.get("id", "Unknown"),
                        )
                    ),
                    "severity": severity,
                    "color": str(band.get("color", "gray")),
                    "risk": risk,
                    "level": level,
                }

    threshold = THRESHOLDS.get(normalize_station_name(name))

    if threshold is None:
        return {
            "label": "Unclassified",
            "severity": "unknown",
            "color": "gray",
            "risk": "unknown",
            "level": level,
        }

    risk = (
        "high"
        if level <= threshold
        else "elevated"
        if level <= threshold + 20
        else "normal"
    )

    return {
        "label": risk.title(),
        "severity": risk,
        "color": (
            "red"
            if risk == "high"
            else "yellow"
            if risk == "elevated"
            else "green"
        ),
        "risk": risk,
        "level": level,
    }


def operational_status(
    name: Any,
    value: Any,
) -> tuple[str, str, str]:
    band = band_for_level(name, value)

    color = {
        "green": COLORS["green"],
        "yellow": COLORS["yellow"],
        "orange": COLORS["orange"],
        "red": COLORS["red"],
        "darkred": COLORS["darkred"],
    }.get(band["color"], COLORS["gray"])

    return band["label"], color, band["risk"]


def fmt_number(value: Any) -> str:
    if value is None:
        return "—"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_time(value: Any) -> str:
    if value is None:
        return "—"

    parsed = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return str(value)

    return parsed.strftime("%d %b %Y · %H:%M UTC")


def risk_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for record in records:
        station = record.get("station_name", "Unknown")

        label, _, risk = operational_status(
            station,
            record.get("prediction"),
        )

        rows.append(
            {
                "Station": station,
                "Segment": primary_segment_label(station),
                "Risk": risk.title(),
                "Operational status": label,
                "Prediction (cm)": fmt_number(
                    record.get("prediction")
                ),
                "Forecast time (UTC)": fmt_time(
                    record.get("forecast_timestamp_utc")
                ),
                "Issued (UTC)": fmt_time(
                    record.get("prediction_ready_utc")
                ),
                "Model version": DISPLAY_MODEL_VERSION,
            }
        )

    return rows


def make_table(
    rows: list[dict[str, Any]],
    page_size: int = 5,
) -> Any:
    if not rows:
        return empty_message(
            "No forecast records",
            "No prediction rows are currently available.",
        )

    tooltip_data = [
        {
            column: {
                "value": f"**{column}**\n\n{str(row[column])}",
                "type": "markdown",
            }
            for column in row
        }
        for row in rows
    ]

    return dash_table.DataTable(
        data=rows,
        columns=[
            {"name": key, "id": key}
            for key in rows[0]
        ],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        tooltip_data=tooltip_data,
        tooltip_duration=None,
        style_table={
            "overflowX": "auto",
            "minWidth": "100%",
            "maxHeight": "500px",
            "overflowY": "auto",
            "borderRadius": "8px",
        },
        style_header={
            "backgroundColor": "#0B111A",
            "color": "#64748B",
            "fontWeight": "600",
            "fontSize": "11px",
            "fontFamily": "Inter, sans-serif",
            "letterSpacing": "0.08em",
            "textTransform": "uppercase",
            "border": "none",
            "borderBottom": "1px solid #1E293B",
            "padding": "12px 16px",
        },
        style_cell={
            "backgroundColor": "transparent",
            "color": "#E2E8F0",
            "fontSize": "13px",
            "fontFamily": "Inter, sans-serif",
            "padding": "14px 16px",
            "textAlign": "left",
            "border": "none",
            "borderBottom": (
                "1px solid rgba(30, 41, 59, 0.6)"
            ),
            "minWidth": "110px",
            "whiteSpace": "nowrap",
        },
        style_data_conditional=[
            {
                "if": {
                    "filter_query": '{Risk} = "High"',
                    "column_id": "Risk",
                },
                "color": COLORS["red"],
                "fontWeight": "600",
            },
            {
                "if": {
                    "filter_query": '{Risk} = "Elevated"',
                    "column_id": "Risk",
                },
                "color": COLORS["yellow"],
                "fontWeight": "600",
            },
            {
                "if": {
                    "filter_query": '{Risk} = "Normal"',
                    "column_id": "Risk",
                },
                "color": COLORS["green"],
                "fontWeight": "600",
            },
            {
                "if": {
                    "column_id": "Prediction (cm)"
                },
                "fontFamily": "'JetBrains Mono', monospace",
                "fontWeight": "500",
            },
        ],
    )


def history_figure(
    records: list[dict[str, Any]],
    station: str | None,
    days: int,
) -> go.Figure:
    figure = go.Figure()

    if not records or not station:
        figure.update_layout(
            title="Forecast history unavailable",
            template="plotly_dark",
            height=430,
        )
        return figure

    frame = pd.DataFrame(records)

    frame = frame[
        frame["station_name"].astype(str) == station
    ].copy()

    frame["forecast_timestamp_utc"] = pd.to_datetime(
        frame["forecast_timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    frame["prediction"] = pd.to_numeric(
        frame["prediction"],
        errors="coerce",
    )

    frame["actual_if_available"] = pd.to_numeric(
        frame["actual_if_available"],
        errors="coerce",
    )

    frame["actual_available_now"] = (
        frame["actual_available_now"].astype(bool)
        if "actual_available_now" in frame.columns
        else False
    )

    frame = frame.dropna(
        subset=["forecast_timestamp_utc"]
    ).sort_values("forecast_timestamp_utc")

    if frame.empty:
        figure.update_layout(
            title=f"No data points found for station {station}",
            template="plotly_dark",
            height=430,
        )
        return figure

    actuals_frame = frame[
        (frame["actual_available_now"] == True)
        & frame["actual_if_available"].notna()
    ].sort_values("forecast_timestamp_utc")

    forecast_frame = frame[
        frame["prediction"].notna()
    ].sort_values("forecast_timestamp_utc")

    if not actuals_frame.empty:
        figure.add_trace(
            go.Scatter(
                x=actuals_frame["forecast_timestamp_utc"],
                y=actuals_frame["actual_if_available"],
                mode="lines+markers",
                name="Actual Level",
                line={
                    "color": COLORS["green"],
                    "width": 2.5,
                },
                marker={
                    "size": 6,
                    "color": COLORS["text"],
                    "line": {
                        "width": 1,
                        "color": COLORS["green"],
                    },
                },
                hovertemplate=(
                    "<b>%{x|%d %b %H:%M} UTC</b>"
                    "<br>Actual: %{y:.2f} cm"
                    "<extra></extra>"
                ),
            )
        )

    if not forecast_frame.empty:
        forecast_x = forecast_frame["forecast_timestamp_utc"]
        forecast_y = forecast_frame["prediction"]

        if not actuals_frame.empty:
            last_actual_row = actuals_frame.iloc[[-1]]

            forecast_x = pd.concat(
                [
                    last_actual_row[
                        "forecast_timestamp_utc"
                    ],
                    forecast_x,
                ]
            )

            forecast_y = pd.concat(
                [
                    last_actual_row[
                        "actual_if_available"
                    ],
                    forecast_y,
                ]
            )

        figure.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Forecast",
                line={
                    "color": COLORS["blue"],
                    "width": 2.5,
                    "dash": "dot",
                },
                marker={
                    "size": 6,
                    "color": COLORS["text"],
                    "line": {
                        "width": 1,
                        "color": COLORS["blue"],
                    },
                },
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.05)",
                hovertemplate=(
                    "<b>%{x|%d %b %H:%M} UTC</b>"
                    "<br>Prediction: %{y:.2f} cm"
                    "<extra></extra>"
                ),
            )
        )

    transition_x = (
        actuals_frame["forecast_timestamp_utc"].max()
        if not actuals_frame.empty
        else frame["forecast_timestamp_utc"].min()
    )

    figure.add_vline(
        x=transition_x.timestamp() * 1000,
        line_dash="dash",
        line_color=COLORS["muted"],
        line_width=1.5,
        annotation_text="Cutoff",
        annotation_position="top right",
        annotation_font={
            "size": 10,
            "color": COLORS["muted"],
        },
    )

    x_min = frame["forecast_timestamp_utc"].min()
    x_max = frame["forecast_timestamp_utc"].max()

    figure.update_layout(
        title=None,
        template="plotly_dark",
        height=430,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#111A27",
            "bordercolor": "#1E293B",
            "font": {
                "family": "Inter",
                "size": 13,
                "color": "#F8FAFC",
            },
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, sans-serif",
            "color": "#94A3B8",
        },
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        xaxis={
            "showgrid": False,
            "linecolor": "#1E293B",
            "title": "Forecast Target (UTC)",
            "title_font": {"size": 11},
            "range": (
                [x_min, x_max]
                if x_min != x_max
                else None
            ),
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "rgba(30, 41, 59, 0.6)",
            "linecolor": "#1E293B",
            "title": "Gauge Level (cm)",
            "title_font": {"size": 11},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure


def map_figure(
    records: list[dict[str, Any]],
    stations: list[dict[str, Any]],
    selected: str | None,
) -> go.Figure:
    selected = None if selected in {None, "all"} else selected

    by_station = {
        normalize_station_name(item.get("station_name")): item
        for item in records
    }

    lon: list[float] = []
    lat: list[float] = []
    text: list[str] = []
    colors: list[str] = []
    sizes: list[int] = []

    for station in stations:
        ids = station.get("segment_ids") or [
            item["segment_id"]
            for item in station_segments(
                station.get("station_name")
            )
        ]

        if (
            (selected and selected not in ids)
            or station.get("longitude") is None
            or station.get("latitude") is None
        ):
            continue

        name = station.get("station_name")

        record = by_station.get(
            normalize_station_name(name),
            {},
        )

        label, color, risk = operational_status(
            name,
            record.get("prediction"),
        )

        lon.append(station["longitude"])
        lat.append(station["latitude"])
        colors.append(color)

        sizes.append(
            12
            if any(
                item["is_decision_gauge"]
                for item in station_segments(name)
            )
            else 7
        )

        text.append(
            f"<b>{name}</b>"
            f"<br>Risk: {risk.title()}"
            f"<br>Status: {label}"
            f"<br>Prediction: "
            f"{fmt_number(record.get('prediction'))} cm"
        )

    figure = go.Figure()

    if lon:
        figure.add_trace(
            go.Scattermapbox(
                lon=lon,
                lat=lat,
                text=text,
                mode="markers",
                marker={
                    "size": sizes,
                    "color": colors,
                    "opacity": 0.9,
                },
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure.update_layout(
        mapbox={
            "style": "carto-darkmatter",
            "center": {"lat": 50.3, "lon": 7.5},
            "zoom": 5.2,
            "pitch": 0,
        },
        title=None,
        template="plotly_dark",
        height=430,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={
            "bgcolor": "#111A27",
            "bordercolor": "#1E293B",
            "font": {
                "family": "Inter",
                "size": 13,
                "color": "#F8FAFC",
            },
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return figure


def section_header(
    eyebrow: str,
    title: str,
    action: Any = None,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        eyebrow.upper(),
                        className="eyebrow",
                    ),
                    html.H2(
                        title,
                        className="section-title",
                    ),
                ]
            ),
            action,
        ],
        className="section-header",
    )


def dmc_select(
    component_id: str,
    data: list[dict[str, str]],
    value: str | None = None,
    class_name: str = "",
    placeholder: str | None = None,
    width: int | str | None = None,
) -> dmc.Select:
    return dmc.Select(
        id=component_id,
        data=data,
        value=value,
        placeholder=placeholder,
        searchable=False,
        clearable=False,
        checkIconPosition="right",
        className=f"mantine-select {class_name}".strip(),
        style={"width": width} if width else None,
        styles={
            "input": {
                "backgroundColor": "#0B111A",
                "borderColor": "#1E293B",
                "color": "#F8FAFC",
                "fontFamily": "Inter, sans-serif",
                "fontSize": "13px",
                "fontWeight": 500,
                "minHeight": "38px",
            },
            "dropdown": {
                "backgroundColor": "#0B111A",
                "borderColor": "#1E293B",
                "boxShadow": "0 12px 28px rgba(0, 0, 0, 0.48)",
            },
            "option": {
                "color": "#94A3B8",
                "fontFamily": "Inter, sans-serif",
                "fontSize": "13px",
            },
        },
    )


segment_options = [
    {
        "label": "All segments",
        "value": "all",
    },
    *[
        {
            "label": str(
                item.get("label", segment_id)
            ),
            "value": str(segment_id),
        }
        for segment_id, item in SEGMENTS.items()
    ],
]

history_day_options = [
    {
        "label": f"{value} days",
        "value": str(value),
    }
    for value in (7, 14, 30, 90)
]

MANTINE_THEME = {
    "fontFamily": "Inter, sans-serif",
    "primaryColor": "cyan",
    "colors": {
        "dark": [
            "#C1C2C5",
            "#A6A7AB",
            "#909296",
            "#5C5F66",
            "#373A40",
            "#2C2E33",
            "#25262B",
            "#1A1B1E",
            "#141517",
            "#101113",
        ],
        "cyan": [
            "#E0F7FF",
            "#B9ECFF",
            "#8BDFFF",
            "#62D3FC",
            "#46CAF9",
            "#38BDF8",
            "#26AEDA",
            "#1B97BF",
            "#0F7C9D",
            "#07566E",
        ],
    },
}


dashboard_layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.Div(
                            "R",
                            className="brand-mark",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "PRODUCTION ANALYTICS",
                                    className="brand-overline",
                                ),
                                html.H1(
                                    "Rhine Corridor Forecaster",
                                    className="brand-title",
                                ),
                            ]
                        ),
                    ],
                    className="brand-lockup",
                )
            ],
            className="topbar",
        ),
        html.Main(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "EXECUTIVE OVERVIEW",
                                    className="eyebrow",
                                ),
                                html.H2(
                                    "Corridor Water Risk Exposure",
                                    className="hero-title",
                                ),
                                html.P(
                                    (
                                        "Decision-ready ML forecasting "
                                        "for logistics, infrastructure, "
                                        "and waterway operations."
                                    ),
                                    className="hero-copy",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "LAST PIPELINE RUN",
                                    className="eyebrow",
                                ),
                                html.Div(
                                    id="last-refresh-time",
                                    className="refresh-value",
                                ),
                                html.Div(
                                    "RhineNav MLOps pipeline",
                                    className="refresh-detail",
                                ),
                            ],
                            className="refresh-box",
                        ),
                    ],
                    className="hero-row",
                ),
                dcc.Loading(
                    id="loading-banner",
                    type="circle",
                    color=COLORS["blue"],
                    children=html.Div(
                        id="system-status-banner",
                        className="status-banner-wrap",
                    ),
                ),
                dcc.Loading(
                    id="loading-metrics",
                    type="circle",
                    color=COLORS["blue"],
                    children=html.Div(
                        id="metric-cards",
                        className="metrics-grid",
                    ),
                ),
                dcc.Loading(
                    id="loading-segments",
                    type="circle",
                    color=COLORS["blue"],
                    children=html.Div(
                        id="segment-cards",
                        className="segment-grid",
                    ),
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                section_header(
                                    "Geospatial intelligence",
                                    "Corridor exposure",
                                    html.Div(
                                        dmc_select(
                                            component_id=(
                                                "segment-dropdown"
                                            ),
                                            data=segment_options,
                                            value="all",
                                            width=200,
                                        ),
                                        className=(
                                            "filter-control"
                                        ),
                                    ),
                                ),
                                dcc.Loading(
                                    id="loading-map",
                                    type="circle",
                                    color=COLORS["blue"],
                                    children=dcc.Graph(
                                        id="corridor-map",
                                        style={
                                            "height": "430px",
                                            "width": "100%",
                                        },
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                            "scrollZoom": True,
                                        },
                                    ),
                                ),
                            ],
                            className="panel panel-map",
                        ),
                        html.Div(
                            [
                                section_header(
                                    "Operational intelligence",
                                    "Operational forecast",
                                ),
                                dcc.Loading(
                                    id="loading-table",
                                    type="circle",
                                    color=COLORS["blue"],
                                    children=html.Div(
                                        id=(
                                            "operational-forecast-table"
                                        ),
                                        className="table-wrapper",
                                    ),
                                ),
                            ],
                            className="panel panel-events",
                        ),
                    ],
                    className="main-grid",
                ),
                html.Div(
                    [
                        section_header(
                            "Station intelligence",
                            "Forecast trajectory",
                            html.Div(
                                [
                                    dmc_select(
                                        component_id=(
                                            "station-dropdown"
                                        ),
                                        data=[],
                                        value=None,
                                        placeholder=(
                                            "Select a station"
                                        ),
                                        class_name="station-select",
                                        width=240,
                                    ),
                                    dmc_select(
                                        component_id="history-days",
                                        data=history_day_options,
                                        value="7",
                                        class_name="days-select",
                                        width=120,
                                    ),
                                ],
                                className="chart-controls",
                            ),
                        ),
                        dcc.Loading(
                            id="loading-chart",
                            type="circle",
                            color=COLORS["blue"],
                            children=dcc.Graph(
                                id="station-forecast-chart",
                                config={
                                    "displayModeBar": False,
                                    "responsive": True,
                                },
                            ),
                        ),
                    ],
                    className="panel chart-panel",
                ),
            ],
            className="content-shell",
        ),
        html.Footer(
            "RHINE CORRIDOR FORECASTER · "
            "GAUGE24H ANALYTICS",
            className="footer",
        ),
    ],
    id="app-shell",
    className="app-shell",
)

app.layout = dmc.MantineProvider(
    forceColorScheme="dark",
    theme=MANTINE_THEME,
    children=dashboard_layout,
)


@app.callback(
    Output("operational-forecast-table", "children"),
    Output("station-dropdown", "data"),
    Output("station-dropdown", "value"),
    Output("metric-cards", "children"),
    Output("segment-cards", "children"),
    Output("system-status-banner", "children"),
    Output("corridor-map", "figure"),
    Output("last-refresh-time", "children"),
    Input("segment-dropdown", "value"),
)
def refresh_dashboard(segment: str | None):
    try:
        predictions = api_get("/predictions/latest")
        stations = api_get("/metadata/stations")
        system = api_get("/system/status")

    except Exception as exc:
        error = empty_message(
            "Dashboard API unavailable",
            str(exc),
        )

        return (
            error,
            [],
            None,
            [],
            [],
            error,
            go.Figure(),
            "Error Fetching Data",
        )

    selected = (
        None
        if segment in {None, "", "all"}
        else segment
    )

    records = [
        record
        for record in predictions
        if selected is None
        or selected in record_segment_ids(record)
    ]

    rows = risk_rows(records)

    station_names = sorted(
        {
            str(record.get("station_name"))
            for record in records
            if record.get("station_name")
        }
    )

    data_quality = normalize_status(
        system.get("data_quality_status")
    )

    summary = system.get("latest_quality_summary") or {}

    if (
        data_quality == "unknown"
        and int(summary.get("failed_metrics", 0) or 0) == 0
        and int(summary.get("passed_metrics", 0) or 0) > 0
    ):
        data_quality = "pass"

    overall = normalize_status(system.get("status"))
    pipeline = normalize_status(system.get("pipeline_status"))
    stage = normalize_status(system.get("stage_status"))
    evaluation = normalize_status(
        system.get("evaluation_status")
    )

    healthy = all(
        value in {"ok", "pass", "available"}
        for value in (
            overall,
            pipeline,
            stage,
            evaluation,
            data_quality,
        )
    )

    high_count = sum(row["Risk"] == "High" for row in rows)

    elevated_count = sum(
        row["Risk"] == "Elevated"
        for row in rows
    )

    last_run_time = (
        fmt_time(records[0].get("prediction_ready_utc"))
        if records
        else "Unknown"
    )

    cards = [
        metric_card(
            "Stations active",
            str(len(station_names)),
            "Selected view",
            COLORS["blue"],
            "carbon:location-current",
        ),
        metric_card(
            "Predictions",
            str(len(records)),
            "Rows in selected view",
            COLORS["cyan"],
            "carbon:machine-learning-model",
        ),
        metric_card(
            "High risk",
            str(high_count),
            "Critical low water",
            COLORS["red"],
            "carbon:warning-alt",
        ),
        metric_card(
            "Elevated risk",
            str(elevated_count),
            "Conditional operations",
            COLORS["yellow"],
            "carbon:warning",
        ),
        metric_card(
            "Pipeline",
            "PASS" if healthy else "DEGRADED",
            "Data quality & sync",
            (
                COLORS["green"]
                if healthy
                else COLORS["yellow"]
            ),
            (
                "carbon:data-check"
                if healthy
                else "carbon:data-error"
            ),
        ),
    ]

    banner_color = (
        COLORS["green"]
        if healthy
        else COLORS["yellow"]
    )

    banner = html.Div(
        [
            html.Strong(
                f"System status: {overall.upper()}"
            ),
            html.Span(
                f" | Pipeline: {pipeline.upper()}",
                className="banner-item",
            ),
            html.Span(
                f" | Quality: {data_quality.upper()}",
                className="banner-item",
            ),
            html.Span(
                f" | Stage: {stage.upper()}",
                className="banner-item",
            ),
            html.Span(
                f" | Evaluation: {evaluation.upper()}",
                className="banner-item",
            ),
        ],
        className="status-banner",
        style={"--status-color": banner_color},
    )

    segment_cards = [
        metric_card(
            title=item.get("label", segment_id),
            value=str(
                len(
                    risk_rows(
                        [
                            record
                            for record in records
                            if segment_id
                            in record_segment_ids(record)
                        ]
                    )
                )
            ),
            subtitle=(
                "Decision: "
                f"{item.get('decision_gauge', '—')}"
            ),
            color=COLORS["text"],
            icon="carbon:chart-waterfall",
        )
        for segment_id, item in SEGMENTS.items()
    ]

    station_options = [
        {
            "label": station_name,
            "value": station_name,
        }
        for station_name in station_names
    ]

    station_value = (
        station_names[0]
        if station_names
        else None
    )

    return (
        make_table(rows, 6),
        station_options,
        station_value,
        cards,
        segment_cards,
        banner,
        map_figure(records, stations, selected),
        last_run_time,
    )


@app.callback(
    Output("station-forecast-chart", "figure"),
    Input("station-dropdown", "value"),
    Input("history-days", "value"),
)
def update_station_chart(
    station: str | None,
    days: str | None,
):
    if not station:
        return history_figure([], None, 7)

    try:
        parsed_days = int(days or "7")

        history_records = api_get(
            "/predictions/history",
            {
                "days": parsed_days,
                "station": station,
            },
        )

        latest_records = api_get("/predictions/latest")

        station_latest = [
            record
            for record in latest_records
            if record.get("station_name") == station
        ]

        return history_figure(
            history_records + station_latest,
            station,
            parsed_days,
        )

    except Exception as exc:
        figure = go.Figure()

        figure.update_layout(
            title=f"Forecast history unavailable: {exc}",
            template="plotly_dark",
            height=430,
        )

        return figure


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