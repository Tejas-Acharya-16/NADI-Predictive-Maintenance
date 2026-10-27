# NASA C-MAPSS FD001 RUL project

This project provides a CPU-bounded, leakage-controlled FD001 predictive-maintenance demonstration. It uses simulated turbofan telemetry and must not be presented as results from the Siemens pump-vibration prototype.

## Setup

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-models.txt
```

PyTorch is needed for the sequence and autoencoder studies. Streamlit is optional for the dashboard.

## Reproduce the completed study

```powershell
& '.\.venv\Scripts\python.exe' run_fd001_study.py
& '.\.venv\Scripts\python.exe' run_fd001_sequence_study.py
& '.\.venv\Scripts\python.exe' evaluate_selected_fd001.py
& '.\.venv\Scripts\python.exe' run_anomaly_triage_demo.py
& '.\.venv\Scripts\python.exe' build_report_artifacts.py
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

The first two commands use only `train_FD001.txt`. The third command is the explicit official-test stage and loads `test_FD001.txt` and `RUL_FD001.txt` once for the validation-selected model.

## Outputs

- `reports/data_audit.md`, `metric_audit.md`, `experiment_plan.md`, `literature_review.md`, and `final_summary.md`
- `reports/metrics_summary_all_models.csv` and selected-model prediction/error files
- `reports/figures/` for actual-vs-predicted, residuals, per-engine error, and PHM08 contributions
- `dashboard.py` for an optional Streamlit view based only on saved files

## Dashboard

```powershell
& '.\.venv\Scripts\python.exe' -m pip install streamlit
& '.\.venv\Scripts\python.exe' -m streamlit run dashboard.py
```
