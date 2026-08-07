import os
from pathlib import Path
import dash
from dash import html, dcc, Input, Output, State
import requests
import pandas as pd
import plotly.graph_objects as go
import yaml
from flask import Response
import json

API_BASE = os.getenv("API_BASE_URL", "https://gauge24h-api-360668288184.europe-west3.run.app")

# ---------------- Thresholds ----------------

THRESHOLDS_FILE = Path(__file__).parent / "threshold.yaml"

def normalize_station_name(name: str) -> str:
    return name.upper().replace(" ", "").replace("-", "")

def load_thresholds(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("low_water_thresholds_cm", {})
    out = {}
    for k, v in raw.items():
        key = normalize_station_name(str(k))
        out[key] = int(v)
    return out

THRESHOLDS_CM = load_thresholds(THRESHOLDS_FILE)

# ---------------- Dash app ----------------

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Rhine Gauge 24h Risk Forecast", style={"textAlign": "center"}),
    html.Div(id="last-updated", style={"textAlign": "center", "color": "#666"}),
    html.Br(),
    dcc.Interval(id="interval", interval=60_000, n_intervals=0),
    
    html.H2("Latest Predictions", style={"textAlign": "center"}),
    html.Div(id="predictions-table"),
    html.Br(),
    
    html.H2("Forecast Time Series", style={"textAlign": "center"}),
    html.Div([
        html.Label("Select history window: ", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="days-dropdown",
            options=[
                {"label": "7 days", "value": 7},
                {"label": "14 days", "value": 14},
                {"label": "30 days", "value": 30},
            ],
            value=7,
            clearable=False,
            style={"width": "200px", "display": "inline-block"},
        ),
    ], style={"textAlign": "center", "marginBottom": "20px"}),
    
    html.Div(id="predictions-charts"),
])

@app.server.route("/debug")
def debug():
    import os
    import requests as req

    api_base = os.getenv("API_BASE_URL", "https://gauge24h-api-360668288184.europe-west3.run.app")
    try:
        r = req.get(f"{api_base}/predictions/latest", timeout=10)
        body = r.text
        status = r.status_code
    except Exception as e:
        body = f"ERROR: {e}"
        status = 500

    return Response(
        json.dumps({
            "API_BASE": api_base,
            "status_code": status,
            "body": body,
        }, indent=2),
        mimetype="application/json",
    )

@app.callback(
    [Output("predictions-table", "children"),
     Output("last-updated", "children")],
    Input("interval", "n_intervals"),
)
def update_predictions(n):
    try:
        resp = requests.get(f"{API_BASE}/predictions/latest", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [html.Div("Failed to load predictions.", style={"color": "red"})], ""

    if not data:
        return [html.Div("No predictions available yet.", style={"textAlign": "center"})], ""

    df = pd.DataFrame(data)

    children = [
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Station"),
                    html.Th("Forecast time (UTC)"),
                    html.Th("Prediction"),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(row["station_name"]),
                    html.Td(row["forecast_timestamp_utc"]),
                    html.Td(f"{row['prediction']:.2f}"),
                ])
                for _, row in df.iterrows()
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ]

    last_updated = f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    return children, last_updated


@app.callback(
    Output("predictions-charts", "children"),
    [Input("interval", "n_intervals"),
     Input("days-dropdown", "value")],
)
def update_charts(n, days):
    try:
        resp = requests.get(f"{API_BASE}/predictions/history", params={"days": days}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [html.Div("Failed to load forecast history.", style={"color": "red"})]

    if not data:
        return [html.Div("No forecast history available yet.", style={"textAlign": "center"})]

    df = pd.DataFrame(data)
    df["forecast_timestamp_utc"] = pd.to_datetime(df["forecast_timestamp_utc"], utc=True)

    charts = []
    for station, g in df.groupby("station_name"):
        g = g.sort_values("forecast_timestamp_utc")
        norm_name = normalize_station_name(station)
        threshold = THRESHOLDS_CM.get(norm_name)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=g["forecast_timestamp_utc"],
            y=g["prediction"],
            mode="lines+markers",
            name="Prediction",
        ))

        if threshold is not None:
            fig.add_trace(go.Scatter(
                x=[g["forecast_timestamp_utc"].min(), g["forecast_timestamp_utc"].max()],
                y=[threshold, threshold],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Low-water threshold ({threshold} cm)",
            ))

        fig.update_layout(
            title=f"{station} – 24h forecast",
            xaxis_title="Forecast time (UTC)",
            yaxis_title="Prediction (cm)",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        charts.append(dcc.Graph(figure=fig, style={"marginBottom": "20px"}))

    return charts


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=True)