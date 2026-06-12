# 🚦 Traffic Demand Prediction — Flipkart Grid 6.0 / Gridlock Hackathon 2.0

> **Leaderboard R² Score: 95.69 | Top 300 Rank**

An end-to-end machine learning pipeline for predicting urban traffic demand using spatial, temporal, road-network, and weather-related features on 77K+ training records.

---

## 📌 Problem Statement

Given historical traffic records tagged with geohash, timestamp, road type, and auxiliary attributes, predict a normalized **demand** value (0–1) for each test record.

**Evaluation Metric:** R² Score

---

## 🗂️ Repository Structure

```
├── gridlock_optimized_v2.ipynb   # Main notebook (full pipeline)
├── train.csv                     # Training data (not included — see below)
├── test.csv                      # Test data (not included — see below)
├── gridlock_optimized_submission.csv  # Generated submission file
└── README.md
```

> **Note:** `train.csv` and `test.csv` are competition datasets and are not included in this repository. Download them from the official Flipkart Grid / Gridlock competition portal.

---

## ⚙️ Pipeline Overview

### 1. Feature Engineering

- **Temporal features** — Hour, minute, minute-of-day, 15-min and 30-min buckets
- **Cyclic encoding** — `sin`/`cos` transforms for hour and time-of-day to capture periodicity
- **Rush-hour flags** — AM peak (7–10h), PM peak (17–20h), night, morning, afternoon, evening buckets
- **Geospatial hierarchy** — Geohash prefix features at 4, 5, and 6 character precision (`geo4`, `geo5`, `geo6`)
- **Road interaction features** — Lanes × landmark indicator, highway/residential flags, large-vehicle highway flag

### 2. Out-of-Fold (OOF) Target Encoding

The most impactful upgrade. The `geohash × timestamp` mean demand correlates **0.987** with the target. OOF encoding with 5-fold KFold prevents data leakage while capturing this signal.

Encoded interaction groups:
| Group | Suffix |
|---|---|
| `geohash` × `timestamp` | `geo_ts` |
| `geohash` × `hour` | `geo_hour` |
| `geohash` | `geo` |
| `geo5` × `hour` | `geo5_hour` |
| `geo4` × `hour` | `geo4_hour` |
| `geohash` × `minute_15` | `geo_min15` |
| `timestamp` | `ts` |
| `day` × `hour` | `day_hour` |
| `geohash` × `day` | `geo_day` |
| `RoadType` × `hour` | `road_hour` |
| `geohash` × `day` × `hour` | `geo_day_hour` |

### 3. Statistical Aggregation Features

Per-geohash demand statistics computed strictly from training data (no leakage on test): std, min, max, p25, p75, median, skew. Also includes geohash frequency as a proxy for urban density, and hour-level demand mean/std.

### 4. Preprocessing

- Ordinal encoding for categorical columns with unknown-value handling
- Numeric NaN imputation with column median
- All features cast to `float32` for memory efficiency

### 5. Log-Transform Target

Demand is right-skewed. Training on `log1p(demand)` and back-transforming with `expm1` improved R² by ~0.5–1 point.

### 6. Model Training — OOF Stacking Ensemble

Three gradient-boosted regressors trained with 5-fold KFold:

| Model | Key Hyperparameters |
|---|---|
| **LightGBM** | `n_estimators=3000`, `lr=0.02`, `num_leaves=127`, `max_depth=10` |
| **XGBoost** | `n_estimators=2000`, `lr=0.02`, `max_depth=9`, `tree_method=hist` |
| **CatBoost** | `iterations=2000`, `lr=0.025`, `depth=9` |

OOF predictions from all three models are stacked using a **Ridge meta-learner**, whose coefficients determine the final blend weights.

---

## 📊 Results

| Model | OOF R² |
|---|---|
| LightGBM | ~0.955+ |
| XGBoost | ~0.953+ |
| CatBoost | ~0.954+ |
| **Stacked Ensemble** | **~0.957** |
| **Leaderboard (public)** | **95.69** |

---

## 🛠️ Setup & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```

### Run

1. Place `train.csv` and `test.csv` in the project root.
2. Open `gridlock_optimized_v2.ipynb` in Jupyter or any compatible environment.
3. Run all cells top to bottom.
4. The submission file `gridlock_optimized_submission.csv` will be generated automatically.

---

## 🔑 Key Takeaways

- **OOF target encoding** on geohash × timestamp was the single most impactful feature (0.987 correlation with target).
- **Log-transforming a skewed target** before training and back-transforming predictions consistently improves gradient boosting performance.
- **Stacking with a Ridge meta-learner** outperforms simple averaging by learning optimal blend weights from OOF predictions.
- Cyclic time encoding (`sin`/`cos`) provides smoother periodicity representation than raw hour values.

---

## 🏆 Competition

**Flipkart Grid 6.0 — Gridlock Hackathon 2.0**  
Track: Urban Traffic Demand Forecasting  
Final Rank: **Top 300**  
Public Leaderboard R²: **95.69**

> 📓 **[Click here to view the full notebook](https://nbviewer.org/github/Hiten0305l/gridlock-traffic-demand-prediction/blob/main/gridlock_optimized_v2.ipynb)**
