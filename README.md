# PCOS Risk Prediction

A machine learning project for exploring clinical data and producing a PCOS risk estimate from 41 clinical inputs. The repository includes a Streamlit interface, a FastAPI prediction API, reproducible analysis scripts, and serialized model artifacts.

> **Research and educational use only.** This tool is not a diagnostic device and must not be used to make medical decisions. A qualified clinician should interpret symptoms and test results.

## What is included

- `app.py` — Streamlit prediction application.
- `api.py` — FastAPI service with `/api/health` and `/api/predict` endpoints.
- `frontend/` — static HTML/CSS/JavaScript interface that calls the FastAPI service.
- `pcos_model.pkl`, `scaler.pkl`, `feature_names.pkl` — trained model and preprocessing artifacts used by the apps.
- `pcos_cleaned.csv`, `PCOS_data_without_infertility.xlsx` — dataset files included in this workspace.

The prediction interfaces derive features such as BMI, FSH/LH, and waist/hip ratio and send the resulting 41 columns to the saved model. The form includes clinical, laboratory, symptom, and ultrasound fields.

## Run locally

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

To run the API in a separate terminal:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API docs are available at `http://127.0.0.1:8000/docs`; health check: `http://127.0.0.1:8000/api/health`.

The Streamlit application is available directly. To use the static frontend, start the API as above, then serve the frontend in another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`. The frontend's API URL field defaults to `http://127.0.0.1:8000`.

## Exploratory data analysis

Generate target balance, missingness, correlation, and numeric summary outputs with:

```bash
python eda.py
```

The generated charts and `summary.json` are written to `eda_outputs/`.

## Reproduce training and evaluation

The reproducible training workflow is in `train_model.py`. It reads `pcos_cleaned.csv`, uses `PCOS (Y/N)` as the target, drops the patient file identifier and spreadsheet artifact columns, imputes missing feature values with training medians, and uses a stratified 80/20 train/test split with random seed 42. It benchmarks Extra Trees, Random Forest, SVM, KNN, Logistic Regression, Decision Tree, and Gaussian Naive Bayes. Extra Trees is tuned with 5-fold stratified GridSearchCV using accuracy as the selection score. The script reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and a per-class report for the tuned estimator.

Run after installing `requirements.txt`:

```bash
python train_model.py --report-only
```

The report-only run compares the models and writes `model_results.json` in the project root and `frontend/model_results.json` for the browser report at `http://127.0.0.1:5500/training-report.html`. It does not replace the model artifacts. The report includes test accuracy, precision, recall, F1, ROC-AUC, the split and class counts, preprocessing, model settings, cross-validation setup, and software versions.

To retrain and replace the app's saved artifacts as well, run `python train_model.py` without `--report-only`. The saved scaler artifact is a fitted imputation-and-scaling pipeline; inference code calls its `transform` method. Use the same scikit-learn version used to train the saved model when loading its pickle artifacts.

The accuracy percentages previously quoted for this project have not been confirmed against this workflow. Run the script and use the resulting `model_results.json` values in a résumé or project description. These holdout results are exploratory, not clinical validation; a separate external dataset and careful leakage review are needed before drawing clinical conclusions.

## Data and privacy

The repository includes clinical dataset files. Verify that you have permission to redistribute them before publishing this repository. Do not enter identifiable patient information into a public deployment. The app's default form values are examples, not clinical recommendations.

## Technologies

Python, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Streamlit, FastAPI, HTML, CSS, and JavaScript.
