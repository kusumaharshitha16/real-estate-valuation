# PropAI: Enterprise Real Estate Valuation Engine 🔮

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://real-estate-valuation-9ambiz5uqevgddfjz7zsme.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

PropAI is a full-stack Machine Learning application that bridges the gap between raw, unstructured web data and automated real estate valuation. Instead of relying on clean, pre-packaged academic datasets, this project builds a complete software life cycle: raw data generation, advanced Regex feature purification, robust feature engineering, hyperparameter-optimized gradient boosting with **XGBoost**, and an active cloud dashboard deployment.

---

## 🚀 Live Demonstration & Codebase
* **Live Web Interface:** [Explore the Active Web Dashboard](YOUR_STREAMLIT_CLOUD_LINK_HERE)
* **Production Codebase:** `app.py`
 
---

## 🏗️ Core Architecture & Pipeline

The system processes data sequentially through four distinct architectural layers:

```text
 [ Data Ingestion Simulation ] ──> 2,000 Complex Property Profiles Generated
               │
               ▼
 [ Regex Purification Engine ] ──> Strips currency symbols & extracts structural attributes
               │
               ▼
 [ Feature Engineering Layer ] ──> Calculates spatial layout ratios (Space_Per_BHK)
               │
               ▼
 [ XGBoost Predictor Matrix ]  ──> Hyperparameter-tuned decision trees evaluate price
               │
               ▼
 [ Cloud UI Workspace (SaaS) ] ──> Renders interactive maps and Plotly market charts
