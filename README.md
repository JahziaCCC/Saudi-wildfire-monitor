# 🇸🇦 Saudi Arabia Wildfire & Industrial Risk Monitor

An automated spatial analysis system written in R to detect, classify, and evaluate wildfire and industrial flare risks across Saudi Arabia using real-time NASA FIRMS satellite data.

---

## 📌 Project Overview
This project processes satellite thermal detections from MODIS and VIIRS sensors to build a comprehensive risk dashboard for Saudi Arabia. It automatically filters false positives, differentiates between industrial gas flares and natural vegetation fires, and calculates localized Key Risk Indices (KRI) and Fire Weather Indices (FWI).

---

## 🚀 Key Features
* **Automated Data Retrieval:** Real-time integration with NASA FIRMS API (VIIRS SNPP/NOAA20/NOAA21 & MODIS).
* **Smart Classification Engine:** Separates industrial flares from vegetation wildfires based on NDVI thresholds and spatial recurrence.
* **Risk Quantification:** Integrates meteorology and vegetation density to compute custom Fire Weather Index (FWI) and Key Risk Index (KRI).
* **Multi-Format Export:** Generates standardized outputs for GIS environments (`.csv` and `.geojson`).

---

## 📊 Summary Statistics (Latest Run)
| Metric | Value |
| :--- | :--- |
| **Total Active Detections** | `1,491` |
| **Classified Wildfires** | `1,491` |
| **Industrial Flares** | `0` |
| **High Risk Areas (KRI ≥ 60)** | `0` |

---

## 🛠️ Tech Stack & Dependencies
* **Language:** R (v4.x)
* **Spatial & Data Manipulation:** `sf`, `rnaturalearth`, `dplyr`, `readr`, `httr2`, `lubridate`
* **Analytics & Visualization:** `ggplot2`, `terra`, `dbscan`, `randomForest`, `leaflet`

---

## 📁 Repository Structure
```text
├── final code.R            # Main script for data fetching, processing & risk modeling
├── ml_feedback_labels.csv  # Feedback dataset for ML refinement
└── README.md               # Documentation
