# AeroSense Intelligence — Project Report

## Abstract
AeroSense is a historical air-quality analytics and next-day PM2.5 forecasting prototype. It combines exploratory time-series analysis, feature engineering, machine-learning model comparison, anomaly screening and an interactive dashboard.

## Dataset and scope
- Source: the Delhi daily observations supplied with the original project.
- Coverage: 2015-01-01 to 2020-07-01.
- Records: 1,925 daily observations.
- Variables: PM2.5, PM10, NO2, SO2, CO and O3.
- The supplied data contains no weather variables and the application does not claim to use meteorological inputs.
- The application is not connected to a live monitoring API.

## Prediction task
The target is PM2.5 on the next day. Inputs include recent pollutant lags (current day, 1, 2, 3, 7 and 14 days), rolling means and standard deviation, and calendar seasonality features.

## Evaluation protocol
The data is ordered chronologically. The earliest 80% of usable samples are used for training and the latest 20% are held out for evaluation. The test set is not randomly shuffled. All candidate models use the same holdout period.

## Verified holdout results
The metrics below were recomputed from the bundled dataset with the project's current modelling code.

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
|---|---:|---:|---:|
| Ridge Regression | 22.95 | 38.38 | 0.779 |
| Random Forest | 23.21 | 37.87 | 0.785 |
| Persistence | 23.61 | 40.96 | 0.748 |
| Gradient Boosting | 23.72 | 38.70 | 0.775 |
| Extra Trees | 24.18 | 38.85 | 0.774 |

The lowest holdout MAE in this run was achieved by **Ridge Regression** (22.95 µg/m³). Model rankings can change when a different dataset or time window is uploaded.

## Modules
1. Command Center — headline indicators, pollutant history and monthly pattern.
2. Forecast Lab — next-day model estimates and model spread.
3. Model Arena — holdout comparison, actual-vs-predicted series and residual plots.
4. Air Quality Explorer — date filtering, pollutant correlation and exports.
5. Anomaly Radar — multivariate Isolation Forest outlier screening.
6. Data Studio — data completeness, descriptive statistics and raw records.
7. About & Methodology — reproducibility and scientific limitations.

## Limitations and responsible interpretation
- Forecasts are statistical estimates, not official CPCB forecasts or medical advice.
- The model uses historical pollutant data only; it does not include weather, traffic, fire events or station metadata.
- The model spread is the range of point predictions across candidate models, not a calibrated prediction interval.
- Isolation Forest flags unusual observations; it does not establish the cause of an anomaly.
- PM2.5 concentration categories displayed in the interface are indicative pollutant bands, not a complete official AQI calculation. Official AQI uses multiple pollutant sub-indices and applicable rules.
- Historical correlations do not prove causation.

## Future research directions
- Add temperature, humidity, wind speed/direction and rainfall from a documented source.
- Validate on additional cities and a strictly later out-of-time period.
- Add calibrated quantile/interval forecasts and prediction coverage analysis.
- Incorporate station-level data and source metadata.
- Add drift monitoring and automated retraining after new verified data arrives.
