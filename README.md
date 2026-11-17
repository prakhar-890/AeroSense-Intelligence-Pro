# AeroSense Intelligence — Air Quality Analytics & PM2.5 Forecasting

A full Streamlit application for historical air-quality exploration, next-day PM2.5 modelling, model evaluation, anomaly screening and CSV analysis.

## Quick start (Windows)

1. Extract the ZIP folder.
2. Open the extracted `AirQuality_Intelligence_Pro` folder in VS Code.
3. Make sure Python 3.10–3.12 is installed.
4. Double-click **START_HERE.bat** OR open the VS Code terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

`run_app.py` is also provided as a Python launcher. The app requires internet only for the first dependency installation; the bundled dataset and app run locally.

## Features

- Executive command centre with KPIs, pollutant trend, latest pollutant snapshot and seasonality.
- Next-day PM2.5 forecast from multiple ML models.
- Model arena with chronological holdout comparison, actual-vs-predicted plots, residual diagnostics and downloadable predictions.
- Date-filtered pollutant explorer, correlation matrix and data export.
- Isolation Forest anomaly radar and anomaly report export.
- Data quality profile, descriptive statistics, raw data viewer and dataset download.
- CSV upload to analyse a compatible dataset.
- Methodology and limitations page.

## Dataset

`data/delhi_daily.csv` is derived from the real Delhi observations provided with the original project (`clean_data.csv`). It contains daily PM2.5, PM10, NO2, SO2, CO and O3 values. It is historical data, not live sensor data.

Expected uploaded CSV format:
- Required: `Date`, `PM2.5`
- Optional: `PM10`, `NO2`, `SO2`, `CO`, `O3`
- One or more rows per date are accepted and aggregated to daily averages.

## Model methodology

- Target: next day's PM2.5 concentration.
- Predictors: lagged pollutant values, rolling means/standard deviation, calendar features.
- Models: persistence baseline, Ridge Regression, Random Forest, Extra Trees, Histogram Gradient Boosting.
- Evaluation: chronological holdout; no random shuffle.
- Metrics: MAE, RMSE and R².

The app fits models to the earlier segment for evaluation and refits available estimators on all usable historical samples for the next-day estimate.

## Important limitations

- No live CPCB/API feed is connected.
- No weather variables are present in the supplied dataset; the model does not claim to use weather.
- The forecast is an academic estimate, not an official air-quality bulletin or health advisory.
- The anomaly model identifies statistical outliers, not their causes.
- PM2.5 alone does not determine official AQI. The category badge is illustrative only.

## Project structure

```text
AirQuality_Intelligence_Pro/
├── app.py
├── run_app.py
├── START_HERE.bat
├── requirements.txt
├── README.md
└── data/
    └── delhi_daily.csv
```

## Troubleshooting

- If `python` is not recognized, install Python and tick **Add Python to PATH**.
- If the browser does not open automatically, copy the `Local URL` printed in the terminal.
- If port 8501 is busy, run `python -m streamlit run app.py --server.port 8502`.
- Keep the `data` folder beside `app.py`.
