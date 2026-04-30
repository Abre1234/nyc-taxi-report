# =============================================================================
# 🚕 Big Data Analytics: Urban Mobility Trends
# ALL-IN-ONE Interactive Dashboard
# =============================================================================

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("NYC_Taxi_Fare.csv")

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"].str.replace(" UTC", "", regex=False), errors="coerce"
)
df["hour"] = df["pickup_datetime"].dt.hour
df["day"]  = df["pickup_datetime"].dt.day_name()
df["year"] = df["pickup_datetime"].dt.year

df.dropna(subset=["pickup_latitude","pickup_longitude",
                  "dropoff_latitude","dropoff_longitude",
                  "fare_amount","pickup_datetime"], inplace=True)

df = df[
    df["pickup_latitude"].between(40.4, 41.0) &
    df["pickup_longitude"].between(-74.3, -73.6) &
    (df["fare_amount"] > 0)
]

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE = min(100_000, len(df))
df_s = df.sample(n=SAMPLE, random_state=42).copy()
df_s["lat_bin"] = df_s["pickup_latitude"].round(2)
df_s["lon_bin"] = df_s["pickup_longitude"].round(2)

geo_agg     = df_s.groupby(["lat_bin","lon_bin"]).size().reset_index(name="trip_count")
hourly_agg  = df_s.groupby("hour").size().reset_index(name="trip_count").sort_values("hour")
yearly_agg  = df_s.groupby("year").size().reset_index(name="trip_count").sort_values("year")
day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
daily_agg   = df_s.groupby("day").size().reset_index(name="trip_count")
daily_agg["day"] = pd.Categorical(daily_agg["day"], categories=day_order, ordered=True)
daily_agg   = daily_agg.sort_values("day")
fare_bins   = df_s.copy()
fare_bins["fare_range"] = pd.cut(fare_bins["fare_amount"],
                                  bins=[0,5,10,15,20,30,50,200],
                                  labels=["$0-5","$5-10","$10-15","$15-20","$20-30","$30-50","$50+"])
fare_agg    = fare_bins.groupby("fare_range", observed=True).size().reset_index(name="count")
hourly_geo  = df_s.groupby(["hour","lat_bin","lon_bin"]).size().reset_index(name="trip_count")

print(f"   ✅ Data ready: {SAMPLE:,} rows, {len(geo_agg):,} geo-bins")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD INDIVIDUAL FIGURES
# ─────────────────────────────────────────────────────────────────────────────
NYC = dict(lat=40.73, lon=-73.99)
DARK = "plotly_dark"

# 1. Static heatmap
fig_map = px.density_mapbox(
    geo_agg, lat="lat_bin", lon="lon_bin", z="trip_count",
    radius=20, center=NYC, zoom=10, mapbox_style="open-street-map",
    color_continuous_scale="Inferno", height=500,
    labels={"trip_count":"Trips"},
)
fig_map.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    margin=dict(l=0,r=0,t=0,b=0),
    coloraxis_colorbar=dict(title="Trips", tickfont=dict(color="#ccc")),
)

# 2. Hourly line
fig_hour = px.line(
    hourly_agg, x="hour", y="trip_count", markers=True,
    labels={"hour":"Hour of Day","trip_count":"Trips"},
    template=DARK, height=320,
    color_discrete_sequence=["#FFA500"],
)
fig_hour.update_traces(line_width=2.5, marker_size=7)
fig_hour.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
    margin=dict(l=40,r=20,t=10,b=40),
    xaxis=dict(tickmode="linear", dtick=2, gridcolor="#2a2f3e"),
    yaxis=dict(gridcolor="#2a2f3e"),
    hovermode="x unified",
)

# 3. Yearly bar
fig_year = px.bar(
    yearly_agg, x="year", y="trip_count",
    labels={"year":"Year","trip_count":"Trips"},
    template=DARK, height=320,
    color="trip_count", color_continuous_scale="Blues", text_auto=True,
)
fig_year.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
    margin=dict(l=40,r=20,t=10,b=40),
    xaxis=dict(tickmode="linear", dtick=1, gridcolor="#2a2f3e"),
    yaxis=dict(gridcolor="#2a2f3e"),
    coloraxis_showscale=False,
)
fig_year.update_traces(textfont_size=10, textangle=0)

# 4. Daily bar
fig_day = px.bar(
    daily_agg, x="day", y="trip_count",
    labels={"day":"Day","trip_count":"Trips"},
    template=DARK, height=320,
    color="trip_count", color_continuous_scale="Teal",
)
fig_day.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
    margin=dict(l=40,r=20,t=10,b=60),
    xaxis=dict(gridcolor="#2a2f3e"),
    yaxis=dict(gridcolor="#2a2f3e"),
    coloraxis_showscale=False,
)

# 5. Fare distribution donut
fig_fare = px.pie(
    fare_agg, names="fare_range", values="count",
    hole=0.55, template=DARK, height=320,
    color_discrete_sequence=px.colors.sequential.Plasma_r,
)
fig_fare.update_traces(textposition="outside", textinfo="percent+label",
                        textfont_size=11)
fig_fare.update_layout(
    paper_bgcolor="#0f1117",
    margin=dict(l=20,r=20,t=10,b=10),
    showlegend=False,
)

# 6. Animated heatmap
fig_anim = px.density_mapbox(
    hourly_geo, lat="lat_bin", lon="lon_bin", z="trip_count",
    animation_frame="hour", radius=22, center=NYC, zoom=10,
    mapbox_style="open-street-map", color_continuous_scale="Plasma",
    height=500, range_color=[0, hourly_geo["trip_count"].quantile(0.97)],
    labels={"trip_count":"Trips","hour":"Hour"},
)
fig_anim.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    margin=dict(l=0,r=0,t=0,b=0),
    coloraxis_colorbar=dict(title="Trips", tickfont=dict(color="#ccc")),
    sliders=[{"currentvalue":{"prefix":"Hour: "},"pad":{"t":50}}],
)

# ─────────────────────────────────────────────────────────────────────────────
# CONVERT FIGURES TO JSON FOR EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def fig_json(fig):
    return fig.to_json()

j_map   = fig_json(fig_map)
j_hour  = fig_json(fig_hour)
j_year  = fig_json(fig_year)
j_day   = fig_json(fig_day)
j_fare  = fig_json(fig_fare)
j_anim  = fig_json(fig_anim)

# KPI values
total_trips   = f"{len(df_s):,}"
avg_fare      = f"${df_s['fare_amount'].mean():.2f}"
peak_hour     = f"{int(hourly_agg.loc[hourly_agg.trip_count.idxmax(),'hour']):02d}:00"
busiest_day   = daily_agg.loc[daily_agg.trip_count.idxmax(),'day']
top_year      = str(int(yearly_agg.loc[yearly_agg.trip_count.idxmax(),'year']))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD HTML
# ─────────────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🚕 NYC Taxi – Urban Mobility Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg:      #0f1117;
    --card:    #161b27;
    --border:  #2a2f3e;
    --accent:  #FFA500;
    --accent2: #4C9BE8;
    --text:    #e0e0e0;
    --muted:   #8892a4;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; }}

  /* ── HEADER ── */
  header {{
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 22px 32px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: .5px; }}
  header h1 span {{ color: var(--accent); }}
  header p {{ color: var(--muted); font-size: .85rem; margin-top: 4px; }}
  .badge {{
    background: var(--accent); color: #000; font-size: .75rem;
    font-weight: 700; padding: 4px 12px; border-radius: 20px;
  }}

  /* ── NAV TABS ── */
  nav {{
    display: flex; gap: 6px; padding: 16px 32px;
    background: var(--card); border-bottom: 1px solid var(--border);
    overflow-x: auto;
  }}
  nav button {{
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); padding: 8px 20px; border-radius: 8px;
    cursor: pointer; font-size: .85rem; white-space: nowrap;
    transition: all .2s;
  }}
  nav button:hover {{ border-color: var(--accent); color: var(--accent); }}
  nav button.active {{
    background: var(--accent); border-color: var(--accent);
    color: #000; font-weight: 700;
  }}

  /* ── MAIN ── */
  main {{ padding: 24px 32px; max-width: 1600px; margin: 0 auto; }}

  /* ── KPI STRIP ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px; margin-bottom: 24px;
  }}
  .kpi {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 20px;
    display: flex; flex-direction: column; gap: 6px;
  }}
  .kpi .label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; }}
  .kpi .value {{ font-size: 1.7rem; font-weight: 700; color: var(--accent); }}
  .kpi .sub   {{ font-size: .75rem; color: var(--muted); }}

  /* ── SECTION PANELS ── */
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}

  /* ── CARDS ── */
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; margin-bottom: 20px;
  }}
  .card h2 {{
    font-size: 1rem; font-weight: 600; margin-bottom: 14px;
    color: var(--text); display: flex; align-items: center; gap: 8px;
  }}
  .card h2 .icon {{ font-size: 1.2rem; }}

  /* ── GRID LAYOUTS ── */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}

  /* ── FOOTER ── */
  footer {{
    text-align: center; padding: 20px;
    color: var(--muted); font-size: .78rem;
    border-top: 1px solid var(--border); margin-top: 10px;
  }}

  /* ── RESPONSIVE ── */
  @media (max-width: 900px) {{
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
    main {{ padding: 16px; }}
  }}
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div>
    <h1>🚕 NYC Taxi &nbsp;<span>Urban Mobility</span>&nbsp; Dashboard</h1>
    <p>Big Data Analytics &nbsp;·&nbsp; Plotly Interactive Visualizations &nbsp;·&nbsp; Dataset: NYC Taxi Fare (Kaggle)</p>
  </div>
  <span class="badge">LIVE INTERACTIVE</span>
</header>

<!-- NAV TABS -->
<nav>
  <button class="active" onclick="showTab('overview',this)">📊 Overview</button>
  <button onclick="showTab('map',this)">🗺️ Pickup Heatmap</button>
  <button onclick="showTab('timeseries',this)">📈 Time Series</button>
  <button onclick="showTab('animated',this)">🎬 Animated Map</button>
  <button onclick="showTab('fares',this)">💰 Fare Analysis</button>
</nav>

<main>

  <!-- KPI STRIP (always visible) -->
  <div class="kpi-row">
    <div class="kpi">
      <span class="label">Total Trips</span>
      <span class="value">{total_trips}</span>
      <span class="sub">sampled records</span>
    </div>
    <div class="kpi">
      <span class="label">Avg Fare</span>
      <span class="value">{avg_fare}</span>
      <span class="sub">per trip</span>
    </div>
    <div class="kpi">
      <span class="label">Peak Hour</span>
      <span class="value">{peak_hour}</span>
      <span class="sub">busiest time</span>
    </div>
    <div class="kpi">
      <span class="label">Busiest Day</span>
      <span class="value" style="font-size:1.2rem">{busiest_day}</span>
      <span class="sub">day of week</span>
    </div>
    <div class="kpi">
      <span class="label">Top Year</span>
      <span class="value">{top_year}</span>
      <span class="sub">most trips</span>
    </div>
  </div>

  <!-- ── TAB: OVERVIEW ── -->
  <div id="tab-overview" class="panel active">
    <div class="grid-2">
      <div class="card">
        <h2><span class="icon">🕐</span> Trips by Hour of Day</h2>
        <div id="p-hour"></div>
      </div>
      <div class="card">
        <h2><span class="icon">📅</span> Trips by Year</h2>
        <div id="p-year"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h2><span class="icon">📆</span> Trips by Day of Week</h2>
        <div id="p-day"></div>
      </div>
      <div class="card">
        <h2><span class="icon">💰</span> Fare Distribution</h2>
        <div id="p-fare-ov"></div>
      </div>
    </div>
  </div>

  <!-- ── TAB: MAP ── -->
  <div id="tab-map" class="panel">
    <div class="card">
      <h2><span class="icon">🗺️</span> NYC Taxi Pickup Density Heatmap &nbsp;<small style="color:var(--muted);font-weight:400">— zoom · pan · hover</small></h2>
      <div id="p-map"></div>
    </div>
  </div>

  <!-- ── TAB: TIME SERIES ── -->
  <div id="tab-timeseries" class="panel">
    <div class="card">
      <h2><span class="icon">🕐</span> Hourly Trip Volume</h2>
      <div id="p-hour2"></div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h2><span class="icon">📅</span> Yearly Trip Volume</h2>
        <div id="p-year2"></div>
      </div>
      <div class="card">
        <h2><span class="icon">📆</span> Day-of-Week Breakdown</h2>
        <div id="p-day2"></div>
      </div>
    </div>
  </div>

  <!-- ── TAB: ANIMATED ── -->
  <div id="tab-animated" class="panel">
    <div class="card">
      <h2><span class="icon">🎬</span> Animated Pickup Heatmap by Hour &nbsp;<small style="color:var(--muted);font-weight:400">— press ▶ to play</small></h2>
      <div id="p-anim"></div>
    </div>
  </div>

  <!-- ── TAB: FARES ── -->
  <div id="tab-fares" class="panel">
    <div class="grid-2">
      <div class="card">
        <h2><span class="icon">💰</span> Fare Range Distribution</h2>
        <div id="p-fare"></div>
      </div>
      <div class="card">
        <h2><span class="icon">📊</span> Fare Amount Histogram</h2>
        <div id="p-hist"></div>
      </div>
    </div>
  </div>

</main>

<footer>
  Built with Python · Pandas · Plotly &nbsp;|&nbsp; NYC Taxi Fare Dataset (Kaggle) &nbsp;|&nbsp; Big Data Analytics Project
</footer>

<script>
// ── Plotly data ──
const DATA = {{
  map:  {j_map},
  hour: {j_hour},
  year: {j_year},
  day:  {j_day},
  fare: {j_fare},
  anim: {j_anim},
}};

// ── Histogram (built client-side from fare data) ──
const fareRaw = {json.dumps(df_s['fare_amount'].clip(upper=80).round(1).tolist())};

// ── Render helpers ──
function render(id, figJson) {{
  const f = (typeof figJson === 'string') ? JSON.parse(figJson) : figJson;
  Plotly.newPlot(id, f.data, f.layout, {{responsive:true, displayModeBar:true}});
}}

function renderHist(id) {{
  const trace = {{
    x: fareRaw, type: 'histogram', nbinsx: 40,
    marker: {{ color: '#4C9BE8', line: {{ color: '#0f1117', width: 0.5 }} }},
    name: 'Fare ($)',
  }};
  const layout = {{
    paper_bgcolor:'#0f1117', plot_bgcolor:'#161b27',
    font: {{ color:'#e0e0e0' }},
    xaxis: {{ title:'Fare Amount ($)', gridcolor:'#2a2f3e' }},
    yaxis: {{ title:'Number of Trips', gridcolor:'#2a2f3e' }},
    margin: {{ l:50,r:20,t:10,b:50 }},
    height: 320,
    bargap: 0.05,
  }};
  Plotly.newPlot(id, [trace], layout, {{responsive:true}});
}}

// ── Tab switching ──
let rendered = {{}};

function showTab(name, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');

  if (!rendered[name]) {{
    rendered[name] = true;
    if (name === 'overview') {{
      render('p-hour',    DATA.hour);
      render('p-year',    DATA.year);
      render('p-day',     DATA.day);
      render('p-fare-ov', DATA.fare);
    }} else if (name === 'map') {{
      render('p-map', DATA.map);
    }} else if (name === 'timeseries') {{
      render('p-hour2', DATA.hour);
      render('p-year2', DATA.year);
      render('p-day2',  DATA.day);
    }} else if (name === 'animated') {{
      render('p-anim', DATA.anim);
    }} else if (name === 'fares') {{
      render('p-fare', DATA.fare);
      renderHist('p-hist');
    }}
  }}
}}

// ── Initial render (overview tab) ──
window.addEventListener('load', () => {{
  rendered['overview'] = true;
  render('p-hour',    DATA.hour);
  render('p-year',    DATA.year);
  render('p-day',     DATA.day);
  render('p-fare-ov', DATA.fare);
}});
</script>
</body>
</html>"""

with open("nyc_taxi_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ nyc_taxi_dashboard.html saved — open it in your browser!")
