# =============================================================================
# 🚕 Big Data Analytics: Urban Mobility Trends
# Dataset : NYC Taxi Fare (Kaggle)
# Tools   : pandas, plotly
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Import Libraries
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("NYC_Taxi_Fare.csv")
print(f"   Raw rows: {len(df):,}  |  Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔧 Preprocessing...")

# 3-a  Convert pickup_datetime to proper datetime (strip trailing ' UTC')
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"].str.replace(" UTC", "", regex=False),
    errors="coerce",
)

# 3-b  Extract time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"]  = df["pickup_datetime"].dt.day_name()
df["year"] = df["pickup_datetime"].dt.year

# 3-c  Drop rows with missing / obviously invalid coordinates or fare
df.dropna(subset=["pickup_latitude", "pickup_longitude",
                  "dropoff_latitude", "dropoff_longitude",
                  "fare_amount", "pickup_datetime"], inplace=True)

# 3-d  Keep only realistic NYC bounding box & positive fares
df = df[
    (df["pickup_latitude"].between(40.4, 41.0)) &
    (df["pickup_longitude"].between(-74.3, -73.6)) &
    (df["fare_amount"] > 0)
]

print(f"   Rows after cleaning: {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Big-Data Optimisation Techniques
# ─────────────────────────────────────────────────────────────────────────────
print("\n⚡ Applying optimisation techniques...")

# ── 4-a  SAMPLING ────────────────────────────────────────────────────────────
# Why: Rendering 500 K+ points in a browser is slow and memory-heavy.
#      A random 50 K–100 K sample preserves the statistical distribution
#      while keeping the visualisation snappy.
SAMPLE_SIZE = min(100_000, len(df))
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
print(f"   Sampled {SAMPLE_SIZE:,} rows for map rendering.")

# ── 4-b  GEOSPATIAL BINNING ──────────────────────────────────────────────────
# Why: Rounding lat/lon to 2 decimal places (~1 km grid) collapses thousands
#      of near-identical points into a single cell, making density patterns
#      visible and reducing the number of unique locations dramatically.
df_sample["lat_bin"] = df_sample["pickup_latitude"].round(2)
df_sample["lon_bin"] = df_sample["pickup_longitude"].round(2)

# ── 4-c  AGGREGATION ─────────────────────────────────────────────────────────
# Why: Instead of plotting every raw row, we count trips per bin / time unit.
#      This reduces millions of data points to a few thousand summary rows,
#      cutting render time and memory usage by orders of magnitude.

# Heatmap aggregation – trips per geo-bin
geo_agg = (
    df_sample
    .groupby(["lat_bin", "lon_bin"])
    .size()
    .reset_index(name="trip_count")
)
print(f"   Geo bins: {len(geo_agg):,}")

# Hourly aggregation
hourly_agg = (
    df_sample
    .groupby("hour")
    .size()
    .reset_index(name="trip_count")
    .sort_values("hour")
)

# Yearly aggregation
yearly_agg = (
    df_sample
    .groupby("year")
    .size()
    .reset_index(name="trip_count")
    .sort_values("year")
)

# Hourly aggregation per year (for animated heatmap)
hourly_geo_agg = (
    df_sample
    .groupby(["hour", "lat_bin", "lon_bin"])
    .size()
    .reset_index(name="trip_count")
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Visualisations
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Building visualisations...")

NYC_CENTER = {"lat": 40.73, "lon": -73.99}

# ── 5-1  STATIC PICKUP HEATMAP ───────────────────────────────────────────────
fig_heatmap = px.density_mapbox(
    geo_agg,
    lat="lat_bin",
    lon="lon_bin",
    z="trip_count",
    radius=18,
    center=NYC_CENTER,
    zoom=10,
    mapbox_style="open-street-map",
    color_continuous_scale="Inferno",
    title="🗺️ NYC Taxi Pickup Density Heatmap",
    labels={"trip_count": "Trip Count"},
    height=650,
)
fig_heatmap.update_layout(
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title="Trips"),
)
fig_heatmap.write_html("heatmap_static.html")
print("   ✅ heatmap_static.html saved.")

# ── 5-2  HOURLY TRIPS LINE CHART ─────────────────────────────────────────────
fig_hourly = px.line(
    hourly_agg,
    x="hour",
    y="trip_count",
    markers=True,
    title="🕐 NYC Taxi Trips by Hour of Day",
    labels={"hour": "Hour of Day (0–23)", "trip_count": "Number of Trips"},
    template="plotly_dark",
    height=450,
)
fig_hourly.update_traces(line_color="#FFA500", line_width=2.5)
fig_hourly.update_layout(
    xaxis=dict(tickmode="linear", dtick=1),
    hovermode="x unified",
)
fig_hourly.write_html("timeseries_hourly.html")
print("   ✅ timeseries_hourly.html saved.")

# ── 5-3  YEARLY TRIPS BAR CHART ──────────────────────────────────────────────
fig_yearly = px.bar(
    yearly_agg,
    x="year",
    y="trip_count",
    title="📅 NYC Taxi Trips by Year",
    labels={"year": "Year", "trip_count": "Number of Trips"},
    template="plotly_dark",
    color="trip_count",
    color_continuous_scale="Blues",
    height=450,
    text_auto=True,
)
fig_yearly.update_layout(
    xaxis=dict(tickmode="linear", dtick=1),
    coloraxis_showscale=False,
)
fig_yearly.write_html("timeseries_yearly.html")
print("   ✅ timeseries_yearly.html saved.")

# ── 5-4  ANIMATED HEATMAP (hour as animation frame) ─────────────────────────
# Each frame shows the pickup density for one hour of the day,
# letting you watch how hotspots shift from midnight through to 11 PM.
fig_animated = px.density_mapbox(
    hourly_geo_agg,
    lat="lat_bin",
    lon="lon_bin",
    z="trip_count",
    animation_frame="hour",
    radius=20,
    center=NYC_CENTER,
    zoom=10,
    mapbox_style="open-street-map",
    color_continuous_scale="Plasma",
    title="🎬 Animated NYC Taxi Pickup Heatmap by Hour",
    labels={"trip_count": "Trip Count", "hour": "Hour"},
    height=680,
    range_color=[0, hourly_geo_agg["trip_count"].quantile(0.97)],
)
fig_animated.update_layout(
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title="Trips"),
    sliders=[{
        "currentvalue": {"prefix": "Hour: "},
        "pad": {"t": 50},
    }],
)
fig_animated.write_html("heatmap_animated.html")
print("   ✅ heatmap_animated.html saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Combined Dashboard (bonus)
# ─────────────────────────────────────────────────────────────────────────────
from plotly.subplots import make_subplots

fig_dashboard = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Trips by Hour", "Trips by Year"),
)

fig_dashboard.add_trace(
    go.Scatter(
        x=hourly_agg["hour"],
        y=hourly_agg["trip_count"],
        mode="lines+markers",
        name="Hourly",
        line=dict(color="#FFA500", width=2),
    ),
    row=1, col=1,
)

fig_dashboard.add_trace(
    go.Bar(
        x=yearly_agg["year"],
        y=yearly_agg["trip_count"],
        name="Yearly",
        marker_color="#4C9BE8",
    ),
    row=1, col=2,
)

fig_dashboard.update_layout(
    title_text="📊 NYC Taxi – Time Series Dashboard",
    template="plotly_dark",
    height=450,
    showlegend=False,
)
fig_dashboard.write_html("dashboard_timeseries.html")
print("   ✅ dashboard_timeseries.html saved.")

print("\n🎉 All done! Open any .html file in your browser to explore.")
