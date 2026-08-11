# 🏛️ Machine Learning-Based Property Valuation for Philadelphia Buildings

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

**An end-to-end Machine Learning pipeline for predicting residential and commercial property market values across Philadelphia's 521,491-property, $139.49B portfolio — built to support the Office of Property Assessment (OPA) in accelerating and standardising the city-wide assessment cycle.**

[Overview](#-overview) • [Dataset](#-dataset) • [Pipeline](#-pipeline) • [Results](#-model-results) • [Key Insights](#-key-insights) • [Setup](#-setup) • [Project Structure](#-project-structure) • [Recommendations](#-strategic-recommendations)

</div>

---

## 📋 Overview

The **Office of Property Assessment (OPA)** of the City of Philadelphia is responsible for assessing the market value of every property in the city. These assessments determine property taxes — a primary revenue source for Philadelphia's public schools and city services. With **521,491 properties** to assess and a process that currently takes **3 months to 1 year per full cycle**, maintaining accurate, equitable, and up-to-date valuations is an enormous operational challenge.

This project delivers a **data-driven Decision Support Tool** that predicts `market_value` from physical property attributes, location data, and transaction history — reducing the assessment cycle from months to **minutes for the first-pass valuation**, while providing transparent, auditable SHAP-based explanations for every prediction.

### Business Questions Answered

| # | Question |
|---|----------|
| 1 | Which ML approach best supports OPA in predicting residential property market values? |
| 2 | What are the key drivers and property characteristics that significantly impact property values? |
| 3 | What strategic recommendations can improve OPA's property assessment process? |

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [Open Data Philly — OPA Property Assessments](https://opendataphilly.org/) |
| **Publisher** | City of Philadelphia |
| **Raw Records** | 581,456 properties |
| **Usable Records (post-cleaning)** | 521,491 properties |
| **Geographic Coverage** | 66 geographic wards across Philadelphia |
| **Property Categories** | 6 (Residential, Hotels & Apts, Store w/ Dwelling, Commercial, Industrial, Vacant Land) |
| **Total Portfolio Value** | **$139.49 billion** |
| **Raw Features** | 75 columns |
| **Features Used for ML** | 33 (after selection & engineering) |
| **Target Variable** | `market_value` (log-transformed for modelling: `log1p(market_value)`) |

### Key Dataset Characteristics

- **Median market value:** $134,900 &nbsp;|&nbsp; **Mean:** $267,475 (right-skewed, $79.49 skewness before log transform)
- **Dominant property type:** Residential Single-Family — **86.46%** of all records
- **75% of properties** valued below ~$218K; 95% below ~$525K
- Building stock is **mature** — majority built before 1950 (Historic + Early 20th Century eras)
- Market value is **highly geographic** — the spread between the highest- and lowest-value wards spans several multiples

---

## 🔄 Pipeline

The notebook follows a five-phase end-to-end pipeline:

```
Raw CSV (581,456 rows)
       │
       ▼
┌─────────────────────────────────────┐
│  Phase 1 · Setup & Data Overview    │
│  • Library imports & config         │
│  • Initial data inspection          │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Phase 2 · Data Cleaning            │
│  2.1  Column reduction (44 dropped) │
│  2.2  Duplicate removal (6,949)     │
│  2.3  Structural outlier treatment  │
│  2.4  Missing value handling        │
│  2.5  Feature engineering (+13)     │
│  2.6  Create df_viz & df_ml         │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Phase 3 · Exploratory Data         │
│  Analysis (EDA)                     │
│  3.1  Property inventory KPIs       │
│  3.2  Value tier distribution       │
│  3.3  Property type composition     │
│  3.4  Building characteristics      │
│  3.5  Geographic / ward analysis    │
│  3.6  Market activity over time     │
│  3.7  Feature association analysis  │
│       (Spearman ρ + Eta η)          │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Phase 4 · ML Modelling             │
│  4.1  Dataset inventory + 3 new     │
│       log/ratio features            │
│  4.2  Feature selection (33) +      │
│       Train/Test split 80/20        │
│  4.3  Baseline models (7 algos)     │
│  4.4  Cross-validation top-3        │
│  4.5  Hyperparameter tuning         │
│       (RandomizedSearchCV 50 itr)   │
│  4.6  Final evaluation (test set)   │
│  4.7  Permutation importance + SHAP │
│  4.8  Cost-Benefit analysis         │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Phase 5 · Conclusions &            │
│  Strategic Recommendations          │
└─────────────────────────────────────┘
```

---

## 🧹 Data Cleaning Summary

| Step | Action | Records / Columns Affected |
|------|--------|---------------------------|
| Column Reduction | Dropped identifiers, PII, >70% missing, leakage fields | **44 columns removed** → 31 remaining |
| Zero Value Removal | Removed `market_value = 0` and `sale_price = 0` | **161 rows removed** |
| Duplicate Removal | Removed fully identical rows (post-admin-col drop) | **6,949 rows removed** |
| Structural Outliers | Domain-threshold filters (year_built, bedrooms, bathrooms, stories, garage spaces) | **1,052 rows removed** |
| Missing Value Handling | Rule-based → "Unknown" label → category-mode imputation → row drops | Handled in sequence |
| Post-FE Cleaning | Dropped rows with invalid `price_per_livable_area` | Minimal additional rows |
| **Final Dataset** | | **521,491 rows × 33 ML features** |

**Anti-leakage note:** `taxable_building`, `taxable_land`, `exempt_building`, `exempt_land`, and `homestead_exemption` were all **explicitly removed** — they are derived from `market_value` by OPA and would cause the model to "cheat" if included as features.

---

## ⚙️ Feature Engineering

13 new features were created from raw attributes:

| Category | Feature | Description |
|----------|---------|-------------|
| **Time / Age** | `building_age` | `2026 − year_built` — depreciation signal |
| | `building_era` | Era bucket: Historic / Early 20th / Mid-century / Modern / Contemporary |
| | `sale_year` | Year of last recorded transaction |
| | `sale_month` | Month of last recorded transaction |
| **Area / Density** | `price_per_livable_area` | `market_value / total_livable_area` ($/sqft) |
| | `livable_area_ratio` | `total_livable_area / total_area` (development intensity) |
| | `lot_size_calc` | `frontage × depth` (alternative area measure) |
| **Binary Amenities** | `has_central_air` | 1 if property has central air conditioning |
| | `has_garage` | 1 if `garage_spaces > 0` |
| | `has_fireplace` | 1 if `fireplaces > 0` |
| | `has_basement` | 1 if basement is present and recorded |
| **Summary Score** | `value_tier` | Market segment: Low / Medium / High / Luxury |

**Additional ML-specific features (Section 4.1):**

| Feature | Rationale |
|---------|-----------|
| `log_total_livable_area` | Log-transforms right-skewed size; improves split quality in tree models |
| `log_total_area` | Log-transforms right-skewed lot size |
| `bath_bed_ratio` | `bathrooms / bedrooms` — luxury proxy; high-end homes show ratio > 1.0 |

---

## 🤖 Model Results

### Baseline Comparison (7 Algorithms)

| Model | Train MAPE | Test MAPE | Train R² | Test R² |
|-------|-----------|-----------|---------|---------|
| Linear Regression | ~45% | ~45% | 0.65 | 0.64 |
| Ridge Regression | ~45% | ~45% | 0.65 | 0.64 |
| Lasso Regression | ~45% | ~45% | 0.65 | 0.65 |
| Extra Trees | ~35% | ~37% | 0.73 | 0.72 |
| Random Forest | ~27% | ~25% | 0.86 | 0.85 |
| Gradient Boosting | ~20% | ~20% | 0.91 | 0.90 |
| **XGBoost** | **~11%** | **~18%** | **0.97** | **0.91** |

> **Key finding:** Non-linear ensemble models dramatically outperform linear approaches, confirming that property value relationships in Philadelphia are inherently non-linear and interaction-driven.

### Cross-Validation (Top-3, 5-Fold)

XGBoost selected as best model:

| Model | CV MAPE Mean | CV MAPE Std | CV R² Mean | CV R² Std |
|-------|-------------|-------------|-----------|-----------|
| **XGBoost** | **21.48%** | **±4.38%** | **0.9114** | **±0.0014** |
| Gradient Boosting | 23.15% | ±3.91% | 0.8978 | ±0.0018 |
| Random Forest | 27.43% | ±2.87% | 0.8671 | ±0.0021 |

Low fold-to-fold standard deviation confirms performance is **stable and not driven by a fortunate random split**.

### Hyperparameter Tuning (RandomizedSearchCV)

- **50 iterations × 3-fold CV** on 200,000 training samples
- **Best parameters found:** `n_estimators=500, max_depth=8, learning_rate=0.1, reg_alpha=0.5`
- **Best tuning CV R²:** 0.9333

### ✅ Final Model Performance (Held-Out Test Set — 104,299 Properties)

| Metric | Train | Test | Gap | Assessment |
|--------|-------|------|-----|------------|
| **R²** | 0.9587 | **0.9397** | −0.019 | 🟢 Excellent — explains 94% of value variance |
| **MAPE** | 11.51% | **13.49%** | +1.98% | 🟡 Good — avg error < 20% of true value |
| **MAE ($)** | $39,531 | **$72,810** | +$33K | 🟡 Driven by high-value commercial outliers |
| **Median AE ($)** | $7,755 | **$8,168** | +$413 | 🟢 Typical property: very tight prediction |
| **Overfitting ΔR²** | — | **0.019** | — | ✅ Minimal — model generalises well |

> **IAAO Compliance:** Both R² (0.94 > 0.70 threshold) and MAPE (13.49% < 15% threshold) meet the **International Association of Assessing Officers (IAAO)** mass appraisal accuracy standards for residential assessment.

> **Note on RMSE:** Train RMSE ($816K) vs Test RMSE ($1,865K) shows a large gap — this is not a model failure. RMSE is highly sensitive to a small number of ultra-high-value commercial properties with large absolute errors. MAPE and Median AE — both outlier-resistant metrics — confirm the model performs reliably for the vast majority of properties.

---

## 🔍 Key Insights

### Feature Importance (Permutation + SHAP)

| Rank | Feature | Permutation Importance | Insight |
|------|---------|----------------------|---------|
| 🥇 1 | `geographic_ward` | **0.51** | Location is the single most powerful value driver — shuffling ward assignment alone drops R² by 0.51 points. Identical physical properties can differ by several multiples in value depending on ward. |
| 🥈 2 | `log_total_livable_area` | **0.26** | Livable area is the dominant *physical* predictor. Log-transformed form outperforms raw area. |
| 🥉 3 | `log_total_area` | **0.05** | Lot size matters, but less than livable space. |
| 4 | `interior_condition` | **0.05** | Property condition directly impacts value — yet it is one of the most frequently stale data fields in OPA's records. |
| 5 | `building_age` | **0.04** | Older properties (Historic era) and newer builds (Contemporary 2000+) both command premiums vs the mid-century cohort. |

### SHAP Waterfall Highlights

- **Median-value property** (actual: $135,700 → predicted: $139,138): **2.5% error** — the model excels at the core residential market that constitutes 86% of OPA's portfolio.
- **Ultra-high-value commercial** (actual: $405.6M → predicted: $13.1M): **96.8% error** — the model's clear boundary. Properties above ~$10M should be routed to specialist commercial appraisers, not assessed by this model.

### EDA Highlights

- **Property inventory:** 521,491 properties across 66 wards, $139.49B total portfolio, median building age 84 years
- **Value distribution:** 75% of properties below $218K; top 5% account for a disproportionate share of total portfolio value
- **Geographic disparity:** Ward-level median values can differ by 10× — geographic ward is the #1 value multiplier
- **Strongest numerical predictor:** `total_livable_area` (Spearman ρ = 0.46)
- **Strongest categorical predictor:** `building_code_description` (Eta η = 0.60)

---

## 💰 Cost-Benefit Analysis

### Case A — Tax Impact Asymmetry

At Philadelphia's **1.3998% FY2024 property tax rate**, prediction errors have directional financial consequences:

| Error Direction | Who Is Affected | Financial Consequence |
|----------------|----------------|-----------------------|
| **Under-assessment** (predicted < actual) | City of Philadelphia | Lost tax revenue — properties taxed below true market value |
| **Over-assessment** (predicted > actual) | Property owners | Unfair tax burden — leading driver of formal assessment appeals |

The model's directional error distribution is directly computable from the test-set predictions (see Section 4.8.1 in the notebook), giving OPA an unprecedented ability to **quantify the direction and dollar size of assessment error** before deployment — rather than discovering it anecdotally through appeals.

### Case B — Operational Time Efficiency

| Dimension | Manual Process | ML-Assisted Hybrid | Improvement |
|-----------|---------------|-------------------|-------------|
| **Cycle time** | 90–365 days | ~2–3 weeks | **12–50× faster** |
| **First-pass throughput** | ~16 properties/day per assessor | 521,491 properties in < 1 hour | **Orders of magnitude faster** |
| **Staff required** | 5 assessors (full cycle) | 1–2 reviewers (flagged subset only) | **60–80% reduction** |
| **Annual salary savings** | — | ~$130,000–$195,000/year | Payback < 2 years on model investment |

**Recommended hybrid workflow:**
1. Run ML model on all 521,491 properties (< 1 hour)
2. Flag any property where `|model_prediction / current_assessment − 1| > 30%` (~30% of portfolio)
3. Route flagged properties to assessor review queue
4. Accept model prediction for non-flagged properties (~70%), with final assessor sign-off
5. Auto-exclude properties above $10M → specialist commercial appraiser

---

## 🏗️ Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| **Language** | Python | 3.10+ |
| **Data** | Pandas, NumPy | Latest |
| **Visualisation** | Matplotlib, Seaborn | Latest |
| **Statistics** | SciPy | Latest |
| **ML — Preprocessing** | Scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`) | 1.4+ |
| **ML — Models** | Scikit-learn (`LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`) | 1.4+ |
| **ML — Best Model** | **XGBoost** (`XGBRegressor`) | 2.0+ |
| **ML — Tuning** | Scikit-learn (`RandomizedSearchCV`, `KFold`, `cross_val_score`) | 1.4+ |
| **Explainability** | SHAP (`TreeExplainer`) | 0.45+ |
| **Notebook** | Jupyter Notebook / Google Colab | — |

---

## 🚀 Setup

### Prerequisites

- Python 3.10 or higher
- 8 GB RAM minimum (16 GB recommended for full-dataset training)
- Dataset file: `PHL_OPA_PROPERTIES.csv` from [Open Data Philly](https://opendataphilly.org/)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/philadelphia-property-valuation.git
cd philadelphia-property-valuation

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Google Colab (Recommended for Full Training)

```python
# Mount Google Drive and set data path
from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/My Drive/<your-folder>/PHL_OPA_PROPERTIES.csv'
```

### Environment Configuration

Update the `DATA_PATH` variable in the notebook's **Setup cell (Section 1)** to point to your local copy of `PHL_OPA_PROPERTIES.csv`.

---

## 📁 Project Structure

```
philadelphia-property-valuation/
│
├── 📓 Machine_Learning-Based_Property_Valuation_for_Philadelphia_Building.ipynb
│                              ← Main analysis notebook (all 5 phases)
│
├── 📄 README.md               ← This file
├── 📄 requirements.txt        ← Python dependencies
│
└── 📂 assets/                 ← (Optional) Charts exported from notebook
    ├── kpi_dashboard.png
    ├── baseline_comparison.png
    ├── shap_beeswarm.png
    └── ...
```

### Notebook Sections at a Glance

| Section | Title | Subsections |
|---------|-------|-------------|
| **1** | Setup & Data Overview | 1.1 Data Overview |
| **2** | Data Cleaning | 2.1 Column Reduction → 2.2 Duplicates → 2.3 Outliers → 2.4 Missing Values → 2.5 Feature Engineering → 2.6 `df_viz` & `df_ml` |
| **3** | Exploratory Data Analysis | 3.1 KPIs → 3.2 Value Distribution → 3.3 Property Types → 3.4 Building Characteristics → 3.5 Geographic Analysis → 3.6 Market Activity → 3.7 Feature Associations |
| **4** | ML Modelling | 4.1 Data Prep → 4.2 Feature Selection & Split → 4.3 Baselines → 4.4 Cross-Validation → 4.5 Hyperparameter Tuning → 4.6 Final Evaluation → 4.7 SHAP Explainability → 4.8 Cost-Benefit Analysis |
| **5** | Conclusions & Recommendations | 5.1 Findings Summary → 5.2 Strategic Recommendations |

---

## 💡 Strategic Recommendations

### For the City of Philadelphia

1. **Data-driven tax revenue forecasting:** Apply the tuned XGBoost pipeline to project ward-level market value shifts for forward-looking school budget planning — reducing mid-year revenue shortfalls.

2. **Geographic tax equity audit:** Use ward-level model predictions to review whether current tax rates and exemption policies produce equitable outcomes across Philadelphia's 66 wards.

### For the Office of Property Assessment (OPA)

1. **Deploy as a decision-support tool** alongside human assessors — not as a standalone replacement. Use the ±30% deviation threshold to flag properties for priority manual review.

2. **Implement ward-level assessment ratio monitoring** — since `geographic_ward` accounts for 2× more predictive importance than any other single feature, systematic ratio disparities within wards are the highest-priority fairness concern.

3. **Prioritise livable area data quality** — `log_total_livable_area` is the most important physical feature. Properties where livable area has not been field-verified in 5+ years should be top of the inspection queue, as unreported renovations are the most common source of systematic error in mid-range residential assessments.

4. **Exclude ultra-high-value properties (> $10M)** from model-based assessment and route them to specialist commercial appraisers — SHAP waterfall analysis confirms the model significantly underestimates extreme-value assets due to income-approach factors not present in the OPA dataset.

5. **Retrain annually** using a rolling 3-year transaction window. Market dynamics — including post-pandemic price shifts, neighbourhood gentrification, and new development — can cause model drift within 12–18 months. Monthly monitoring of prediction-to-actual ratios on newly recorded sales provides an early-warning retraining signal.

---

## 📈 Evaluation Metrics

| Metric | Formula | Why Used |
|--------|---------|---------|
| **MAPE** (Primary) | `mean(|actual − predicted| / actual) × 100%` | Scale-independent; interpretable as "% off" for any property value; robust for communicating accuracy to non-technical OPA stakeholders |
| **R²** (Secondary) | `1 − SS_res / SS_tot` | Measures proportion of market value variance explained; comparable to published mass-appraisal literature |
| **MAE ($)** | `mean(|actual − predicted|)` | Absolute dollar error; operational meaning for assessors |
| **Median AE ($)** | `median(|actual − predicted|)` | Outlier-resistant error; reflects typical property accuracy |

**MAPE interpretation scale:**

| MAPE | Rating |
|------|--------|
| 0% – 10% | 🟢 Excellent |
| 10% – 20% | 🟡 Good *(project target met: 13.49%)* |
| 20% – 50% | 🟠 Reasonable |
| > 50% | 🔴 Inaccurate |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

The OPA dataset is publicly available from [Open Data Philly](https://opendataphilly.org/) and is published by the City of Philadelphia.

---

## 🙏 Acknowledgements

- **City of Philadelphia — Office of Property Assessment (OPA)** for publishing the property assessment dataset via Open Data Philly
- **Open Data Philly** ([opendataphilly.org](https://opendataphilly.org)) for open data infrastructure
- **International Association of Assessing Officers (IAAO)** for mass appraisal accuracy standards referenced in model evaluation
- **Scikit-learn**, **XGBoost**, and **SHAP** open-source communities

---

<div align="center">

**Built with ❤️ for fair, transparent, and efficient property assessment in Philadelphia**

*For questions, issues, or contributions — please open a GitHub Issue or Pull Request.*

</div>
