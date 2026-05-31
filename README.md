# F1 Race Prediction App

A machine learning application that predicts F1 race finishing positions using historical race data.

The model uses qualifying times, tyre strategy, weather conditions, and pit stop data to predict finishing order via XGBoost, with results visualized in a Streamlit dashboard.

---

## Features (Phase 1)

- Historical F1 race data acquisition (2018–2024) via FastF1 with local caching
- Feature engineering from qualifying, tyre, pit stop, and weather data
- XGBoost regression model for finishing position prediction
- Time-series cross-validation (TimeSeriesSplit) to prevent data leakage
- Baseline comparison against "qualifying position = finishing position"
- Streamlit dashboard showing predicted standings and feature importance

---

## Requirements

- Python 3.10+
- Internet connection (required only for initial data collection)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/F1-prediction_app.git
cd F1-prediction_app
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac / Linux)
source venv/bin/activate

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Collection

Training data is fetched from the FastF1 API.  
**This step is only required once.** Data is cached locally after the first download.

### Quick start (2 seasons only, recommended for testing)

```bash
python build_dataset.py --start-year 2023 --end-year 2024
```

### Full dataset (2018–2024)

Due to the FastF1 API rate limit (500 requests/hour), fetching one year at a time is recommended.

```bash
python build_dataset.py --start-year 2018 --end-year 2018
python build_dataset.py --start-year 2019 --end-year 2019
python build_dataset.py --start-year 2020 --end-year 2020
python build_dataset.py --start-year 2021 --end-year 2021
python build_dataset.py --start-year 2022 --end-year 2022
python build_dataset.py --start-year 2023 --end-year 2023
python build_dataset.py --start-year 2024 --end-year 2024
```

Once all years are cached, build the combined CSV:

```bash
python build_dataset.py --start-year 2018 --end-year 2024 --force
```

### Notes

- Estimated time per year: 40–60 minutes (first run, no cache)
- Safe to interrupt with `Ctrl+C` — cached rounds are skipped on restart
- Cache location: `data/cache/`
- Output CSV: `data/processed/training_features.csv`

---

## Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

### Sidebar

| Field | Description |
|---|---|
| Season | Select the race year |
| Round Number | Enter the round (1–24) |
| Circuit / Country | Auto-displays the venue for the selected round |

### Feature Importance

A bar chart showing which features the model relies on most.  
`quali_pos` (qualifying position) is typically the most important feature.

### Run Prediction

Fetches data for the selected race and predicts finishing order.  
For past races, actual results are shown alongside predictions with MAE and Top-5 hit rate.

### Cross-Validation

Runs time-series cross-validation to evaluate model generalization.  
Results are compared against the baseline to confirm the model adds value beyond qualifying order.

---

## Evaluation Metrics

| Metric | Description | Phase 1 Target |
|---|---|---|
| MAE | Mean absolute error between predicted and actual position | Within 4–6 places |
| Top-5 Hit Rate | Fraction of top-5 finishers correctly predicted | 40–50% |
| vs Baseline | Does the model outperform "qualifying = finishing"? | Must exceed baseline |

---

## Project Structure

```
F1-prediction_app/
├── app.py                  # Streamlit dashboard
├── build_dataset.py        # Data collection script (run once)
├── explore_data.py         # FastF1 data structure explorer
├── requirements.txt
├── data/
│   ├── cache/              # FastF1 cache (auto-generated, not tracked by git)
│   └── processed/          # Training CSV and model file (auto-generated, not tracked by git)
└── src/
    ├── constants.py        # All project-wide constants
    ├── fetch.py            # FastF1 data fetching with rate-limit handling
    ├── features.py         # Feature engineering pipeline
    ├── model.py            # XGBoost training, prediction, and persistence
    ├── evaluate.py         # MAE, Top-N metrics, baseline comparison
    └── rag/                # Phase 2+ (LightRAG integration placeholder)
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | FastF1 data pipeline + XGBoost MVP | ✅ Complete |
| Phase 2 | LightRAG + LLM team DNA scoring | 🔜 Next |
| Phase 3 | Bayesian updating + Monte Carlo simulation | 🔜 Planned |

---

## Tech Stack

| Category | Library |
|---|---|
| Data acquisition | FastF1, OpenF1 API |
| Data processing | pandas, scikit-learn |
| ML model | XGBoost |
| Dashboard | Streamlit |
| Visualization | Plotly |
