from pathlib import Path
import io, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AeroSense | Air Intelligence", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "delhi_daily.csv"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
COLORS = ["#45D6B5", "#62A8FF", "#FFB45B", "#C59BFF", "#FF718B"]

def load_dataset(upload=None):
    if upload is not None:
        df = pd.read_csv(upload)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        candidates = [BASE / "clean_data.csv", BASE / "delhi.csv"]
        source = next((p for p in candidates if p.exists()), None)
        if source is None: raise FileNotFoundError("No bundled dataset found.")
        df = pd.read_csv(source)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime", "timestamp"]), None)
    if date_col is None: raise ValueError("CSV needs a Date column.")
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["Date"]).copy()
    for c in POLLUTANTS:
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["Date"] + POLLUTANTS].sort_values("Date")
    df = df.groupby("Date", as_index=False)[POLLUTANTS].mean()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def features_from_series(df):
    d = df.copy().sort_values("Date").set_index("Date")
    # Use past observations only; target is the next calendar day.
    x = pd.DataFrame(index=d.index)
    for c in POLLUTANTS:
        if c in d:
            for lag in [0, 1, 2, 3, 7, 14]:
                x[f"{c}_lag{lag}"] = d[c].shift(lag)
            x[f"{c}_mean7"] = d[c].rolling(7, min_periods=3).mean()
            x[f"{c}_mean14"] = d[c].rolling(14, min_periods=5).mean()
            x[f"{c}_std7"] = d[c].rolling(7, min_periods=3).std()
    x["month"] = x.index.month
    x["dayofyear"] = x.index.dayofyear
    x["dayofweek"] = x.index.dayofweek
    x["month_sin"] = np.sin(2*np.pi*x.index.dayofyear/365.25)
    x["month_cos"] = np.cos(2*np.pi*x.index.dayofyear/365.25)
    y = d["PM2.5"].shift(-1).rename("target")
    out = x.join(y).replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
    return out.dropna(axis=1, how="all")

def model_zoo(seed=42):
    return {
        "Ridge Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=8.0))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=260, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=-1))]),
        "Extra Trees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=260, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed))])
    }

def score_models(df, test_fraction=0.2):
    feat = features_from_series(df)
    if len(feat) < 80: raise ValueError("At least 80 usable daily observations are required for reliable model comparison.")
    split = max(50, int(len(feat)*(1-test_fraction)))
    split = min(split, len(feat)-20)
    X, y = feat.drop(columns="target"), feat["target"]
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    preds = {"Persistence": df.set_index("Date")["PM2.5"].reindex(y_test.index).values}
    fitted = {}
    for name, model in model_zoo().items():
        m = clone(model)
        m.fit(X_train, y_train)
        preds[name] = m.predict(X_test)
        fitted[name] = m
    rows = []
    for name, p in preds.items():
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, p), "RMSE": mean_squared_error(y_test, p)**0.5, "R²": r2_score(y_test, p)})
    metrics = pd.DataFrame(rows).sort_values("MAE")
    pred_frame = pd.DataFrame({"Date": y_test.index, "Actual": y_test.values, **preds})
    return metrics, pred_frame, fitted, (X_train, X_test, y_train, y_test), feat

def aqi_label(v):
    if pd.isna(v): return ("Unknown", "—")
    if v <= 30: return ("Good", "#16A34A")
    if v <= 60: return ("Satisfactory", "#65A30D")
    if v <= 90: return ("Moderate", "#CA8A04")
    if v <= 120: return ("Poor", "#EA580C")
    if v <= 250: return ("Very Poor", "#DC2626")
    return ("Severe", "#7F1D1D")

def make_forecast(df, fitted, feat):
    latest = df.sort_values("Date").copy()
    hist = latest.set_index("Date")
    row = pd.DataFrame(index=[hist.index.max()])
    for c in POLLUTANTS:
        s = hist[c]
        for lag in [0,1,2,3,7,14]:
            row[f"{c}_lag{lag}"] = s.iloc[-1-lag] if len(s)>lag else np.nan
        row[f"{c}_mean7"] = s.tail(7).mean()
        row[f"{c}_mean14"] = s.tail(14).mean()
        row[f"{c}_std7"] = s.tail(7).std()
    dt = hist.index.max()
    row["month"], row["dayofyear"], row["dayofweek"] = dt.month, dt.dayofyear, dt.dayofweek
    row["month_sin"], row["month_cos"] = np.sin(2*np.pi*dt.dayofyear/365.25), np.cos(2*np.pi*dt.dayofyear/365.25)
    X = feat.drop(columns="target")
    row = row.reindex(columns=X.columns)
    # Refit selected estimator on all available historical rows.
    output = {}
    for name, model in fitted.items():
        model.fit(X, feat["target"])
        output[name] = float(model.predict(row)[0])
    return output

def kpi(label, value, delta=None):
    st.metric(label, value, delta)

# Sidebar
with st.sidebar:
    st.markdown("## 🌍 AeroSense")
    st.caption("AIR QUALITY INTELLIGENCE PLATFORM")
    page = st.radio("NAVIGATION", ["Command Center", "Forecast Lab", "Model Arena", "Air Quality Explorer", "Anomaly Radar", "Data Studio", "About & Methodology"])
    st.divider()
    st.markdown("**DATA SOURCE**")
    uploaded = st.file_uploader("Upload your own CSV", type=["csv"], help="Expected columns: Date and PM2.5. Other pollutant columns are optional.")
    st.caption("Default dataset: historical Delhi daily observations supplied with this project.")
    st.divider()
    st.caption("Academic prototype • Forecasts are estimates, not official CPCB readings.")

try:
    df = load_dataset(uploaded)
    if df["PM2.5"].notna().sum() < 30: raise ValueError("Dataset must contain at least 30 valid PM2.5 observations.")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

df = df.dropna(subset=["PM2.5"]).sort_values("Date").reset_index(drop=True)
metrics, pred_frame, fitted, split_data, feature_table = score_models(df)
best_name = metrics.iloc[0]["Model"]
best_mae = float(metrics.iloc[0]["MAE"])
forecast_values = make_forecast(df, fitted, feature_table)
best_forecast = forecast_values.get(best_name, np.nan)
latest_value = float(df["PM2.5"].iloc[-1])
aqi_text, aqi_color = aqi_label(latest_value)
next_day = df["Date"].max() + pd.Timedelta(days=1)

# Header
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 15% 0%, rgba(25,93,105,.18), transparent 35%), #08111D; color:#EAF2F7;}
section[data-testid="stSidebar"] {background:#0D1927;}
.block-container {padding-top:1.6rem; max-width:1500px;}
.hero {padding:24px 28px; border:1px solid #23384A; border-radius:18px; background:linear-gradient(120deg,#102A39,#122033 65%,#18243A); margin-bottom:18px;}
.hero h1 {font-size:2.25rem; margin:0; color:#F4FAFC;}
.hero p {color:#9FB7C7; margin:.4rem 0 0 0;}
.small-muted {color:#8DA6B7; font-size:.85rem;}
div[data-testid="stMetric"] {background:#101E2D; border:1px solid #22384A; padding:14px 16px; border-radius:14px;}
div[data-testid="stMetricLabel"] {color:#9FB7C7;}
div[data-testid="stMetricValue"] {color:#F2F8FB;}
</style>
<div class="hero"><h1>🌍 AeroSense Intelligence</h1><p>Predictive air-quality analytics • Explainable machine learning • Environmental decision support</p></div>
""", unsafe_allow_html=True)

if page == "Command Center":
    st.subheader("Command Center")
    st.caption(f"Historical monitoring window: {df.Date.min():%d %b %Y} — {df.Date.max():%d %b %Y}  |  {len(df):,} daily observations")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Latest PM2.5", f"{latest_value:.1f} µg/m³", f"{aqi_text} category")
    c2.metric("Next-day estimate", f"{best_forecast:.1f} µg/m³", best_name)
    c3.metric("Best backtest MAE", f"{best_mae:.2f} µg/m³", "chronological holdout")
    c4.metric("Historical records", f"{len(df):,}", f"{df.Date.nunique():,} dates")
    left,right = st.columns([1.55,1])
    with left:
        st.markdown("#### Pollution timeline")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df.Date,y=df["PM2.5"],mode="lines",name="PM2.5",line=dict(color="#45D6B5",width=1.5)))
        fig.add_trace(go.Scatter(x=df.Date,y=df["PM2.5"].rolling(30).mean(),mode="lines",name="30-day trend",line=dict(color="#FFB45B",width=2.5)))
        fig.update_layout(template="plotly_dark",height=360,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="µg/m³",legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("#### Pollutant snapshot")
        latest=df.iloc[-1]
        vals=pd.DataFrame({"Pollutant":POLLUTANTS,"Latest value":[latest[c] for c in POLLUTANTS]})
        vals=vals.dropna()
        fig=px.bar(vals,x="Latest value",y="Pollutant",orientation="h",color="Pollutant",color_discrete_sequence=COLORS,template="plotly_dark")
        fig.update_layout(height=360,showlegend=False,margin=dict(l=5,r=5,t=15,b=5),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Seasonal intelligence")
    monthly=df.assign(Month=df.Date.dt.month).groupby("Month")["PM2.5"].mean().reindex(range(1,13))
    fig=px.bar(x=monthly.index,y=monthly.values,labels={"x":"Month","y":"Average PM2.5 (µg/m³)"},template="plotly_dark",color=monthly.values,color_continuous_scale="Tealgrn")
    fig.update_layout(height=270,margin=dict(l=10,r=10,t=10,b=10),coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    st.info("Interpretation: the forecast is generated from historical pollutant patterns. It is not a live sensor feed or an official AQI bulletin.")

elif page == "Forecast Lab":
    st.subheader("🔮 Forecast Lab")
    st.write("Estimate the next day's PM2.5 using multiple machine-learning models trained on a chronological historical split.")
    fc=pd.DataFrame([{"Model":k,"Forecast PM2.5 (µg/m³)":v,"Indicative category":aqi_label(v)[0]} for k,v in forecast_values.items()])
    fc.loc[len(fc)]={"Model":"Model ensemble (median)","Forecast PM2.5 (µg/m³)":float(np.median(list(forecast_values.values()))),"Indicative category":aqi_label(float(np.median(list(forecast_values.values()))))[0]}
    st.metric(f"Forecast for {next_day:%d %B %Y}",f"{best_forecast:.1f} µg/m³",f"Best backtest model: {best_name}")
    st.dataframe(fc.style.format({"Forecast PM2.5 (µg/m³)":"{:.2f}"}),use_container_width=True,hide_index=True)
    st.markdown("#### Forecast uncertainty band")
    vals=list(forecast_values.values())
    lo,hi=min(vals),max(vals)
    fig=go.Figure()
    fig.add_trace(go.Indicator(mode="gauge+number",value=best_forecast,title={"text":"PM2.5 estimate (µg/m³)"},gauge={"axis":{"range":[0,max(300,hi*1.2)]},"bar":{"color":"#45D6B5"},"steps":[{"range":[0,30],"color":"#173C34"},{"range":[30,90],"color":"#594B24"},{"range":[90,250],"color":"#5A2B2D"},{"range":[250,max(300,hi*1.2)],"color":"#382039"}]}))
    fig.update_layout(template="plotly_dark",height=330,margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Model spread: {lo:.1f}–{hi:.1f} µg/m³. This is the range across model point estimates, not a statistically calibrated confidence interval.")
    st.markdown("#### Recent context")
    recent=df.tail(30)[["Date","PM2.5"]].copy()
    recent["7-day average"]=recent["PM2.5"].rolling(7,min_periods=1).mean()
    fig=px.line(recent,x="Date",y=["PM2.5","7-day average"],template="plotly_dark",height=300)
    st.plotly_chart(fig,use_container_width=True)

elif page == "Model Arena":
    st.subheader("🏁 Model Arena")
    st.write("Models are compared on the same untouched final chronological segment. Random shuffling is avoided to reduce time-series leakage.")
    st.dataframe(metrics.style.format({"MAE":"{:.2f}","RMSE":"{:.2f}","R²":"{:.3f}"}),use_container_width=True,hide_index=True)
    fig=px.bar(metrics,x="Model",y=["MAE","RMSE"],barmode="group",template="plotly_dark",color_discrete_sequence=["#45D6B5","#FFB45B"])
    fig.update_layout(height=350)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Actual vs predicted — holdout")
    chosen=st.selectbox("Prediction model",list(pred_frame.columns[2:]),index=list(pred_frame.columns[2:]).index(best_name) if best_name in pred_frame.columns else 0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pred_frame.Date,y=pred_frame.Actual,name="Actual",line=dict(color="white",width=2)))
    fig.add_trace(go.Scatter(x=pred_frame.Date,y=pred_frame[chosen],name=chosen,line=dict(color="#45D6B5",width=1.7)))
    fig.update_layout(template="plotly_dark",height=400,yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Residual diagnostics")
    residual=pred_frame.Actual-pred_frame[chosen]
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(x=residual,nbins=35,template="plotly_dark",labels={"x":"Prediction error"})
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.scatter(x=pred_frame.Actual,y=residual,template="plotly_dark",labels={"x":"Actual PM2.5","y":"Residual"})
        fig.add_hline(y=0,line_dash="dash",line_color="#FFB45B")
        st.plotly_chart(fig,use_container_width=True)
    st.download_button("Download holdout predictions CSV",pred_frame.to_csv(index=False).encode(), "holdout_predictions.csv","text/csv")

elif page == "Air Quality Explorer":
    st.subheader("📊 Air Quality Explorer")
    c1,c2=st.columns(2)
    with c1: start=st.date_input("Start date",df.Date.min().date(),min_value=df.Date.min().date(),max_value=df.Date.max().date())
    with c2: end=st.date_input("End date",df.Date.max().date(),min_value=df.Date.min().date(),max_value=df.Date.max().date())
    view=df[(df.Date.dt.date>=start)&(df.Date.dt.date<=end)]
    if view.empty: st.warning("No observations in selected range.")
    else:
        pollutant=st.selectbox("Pollutant",POLLUTANTS,index=0)
        fig=px.line(view,x="Date",y=pollutant,template="plotly_dark",title=f"{pollutant} over selected period")
        st.plotly_chart(fig,use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Mean",f"{view[pollutant].mean():.2f}")
        c2.metric("Median",f"{view[pollutant].median():.2f}")
        c3.metric("Maximum",f"{view[pollutant].max():.2f}")
        corr=view[POLLUTANTS].corr()
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,template="plotly_dark",title="Pollutant correlation matrix")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(view,use_container_width=True,hide_index=True)
        st.download_button("Download filtered data",view.to_csv(index=False).encode(),"filtered_air_quality.csv","text/csv")

elif page == "Anomaly Radar":
    st.subheader("🚨 Anomaly Radar")
    st.write("Isolation Forest flags unusual PM2.5 observations relative to the historical distribution. A flag is an investigation cue, not proof of an event.")
    contamination=st.slider("Expected anomaly proportion",0.01,0.15,0.04,0.01)
    work=df[["Date"]+POLLUTANTS].copy()
    matrix=work[POLLUTANTS].interpolate(limit_direction="both")
    matrix=matrix.fillna(matrix.median())
    detector=IsolationForest(n_estimators=300,contamination=contamination,random_state=42)
    work["Anomaly"]=detector.fit_predict(matrix)==-1
    work["Anomaly score"]=detector.decision_function(matrix)
    fig=go.Figure()
    normal=work[~work.Anomaly]; abnormal=work[work.Anomaly]
    fig.add_trace(go.Scatter(x=normal.Date,y=normal["PM2.5"],mode="markers",name="Typical",marker=dict(color="#45D6B5",size=5)))
    fig.add_trace(go.Scatter(x=abnormal.Date,y=abnormal["PM2.5"],mode="markers",name="Flagged anomaly",marker=dict(color="#FF718B",size=8,symbol="x")))
    fig.update_layout(template="plotly_dark",height=390,yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig,use_container_width=True)
    st.metric("Flagged observations",int(work.Anomaly.sum()),f"{work.Anomaly.mean()*100:.1f}% of records")
    st.dataframe(work[work.Anomaly].sort_values("Anomaly score").head(100),use_container_width=True,hide_index=True)
    st.download_button("Download anomaly report",work[work.Anomaly].to_csv(index=False).encode(),"anomaly_report.csv","text/csv")

elif page == "Data Studio":
    st.subheader("🧪 Data Studio")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{len(df):,}")
    c2.metric("Date coverage",f"{(df.Date.max()-df.Date.min()).days:,} days")
    c3.metric("Missing PM2.5",f"{df['PM2.5'].isna().sum():,}")
    st.markdown("#### Data quality profile")
    quality=pd.DataFrame({"Column":["Date"]+POLLUTANTS,"Missing values":[df.Date.isna().sum()]+[df[c].isna().sum() for c in POLLUTANTS],"Missing %":[df.Date.isna().mean()*100]+[df[c].isna().mean()*100 for c in POLLUTANTS],"Unique values":[df.Date.nunique()]+[df[c].nunique() for c in POLLUTANTS]})
    st.dataframe(quality.style.format({"Missing %":"{:.2f}%"}),use_container_width=True,hide_index=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df[POLLUTANTS].describe().T,use_container_width=True)
    st.markdown("#### Raw records")
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Download dataset currently loaded",df.to_csv(index=False).encode(),"air_quality_dataset.csv","text/csv")

else:
    st.subheader("ℹ️ About & Methodology")
    st.markdown("""
### Project objective
AeroSense is an academic decision-support prototype that explores historical air pollution patterns and estimates next-day PM2.5 concentration from recent pollutant observations.

### Technical pipeline
1. **Data ingestion:** bundled Delhi daily CSV or user-uploaded CSV.
2. **Preprocessing:** date parsing, numeric conversion, duplicate-date aggregation and missing-value handling inside model pipelines.
3. **Feature engineering:** pollutant lags (1, 2, 3, 7 and 14 days), rolling means and volatility, plus calendar seasonality.
4. **Chronological evaluation:** earlier observations train the models; the latest segment is held out for testing.
5. **Model comparison:** persistence baseline, Ridge Regression, Random Forest, Extra Trees and Histogram Gradient Boosting.
6. **Interpretation:** MAE, RMSE, R², prediction traces and residual diagnostics.
7. **Anomaly screening:** Isolation Forest identifies multivariate observations that differ from the historical pattern.

### Important scientific limitations
- This supplied dataset contains historical observations; the app does not connect to live monitoring stations.
- A next-day prediction is an estimate, not an official forecast or health advisory.
- The supplied project data has pollutant measurements but no meteorological variables. The app therefore does not claim to use weather inputs.
- The anomaly detector finds statistical outliers; it cannot identify their cause.
- AQI category labels in this prototype are indicative bands for display and must not be confused with an official CPCB AQI calculation. PM2.5 concentration alone is not sufficient to calculate full AQI.

### Reproducibility
The split is chronological, not randomly shuffled. All models use the same holdout period. The app recalculates model scores when the dataset changes.
""")
    st.markdown("**Suggested academic extension:** add station-level data, meteorological features, calibrated prediction intervals, and external validation on a later time period.")
st.divider()
st.caption("AeroSense Intelligence • Built for academic demonstration • Historical-data analysis only")
