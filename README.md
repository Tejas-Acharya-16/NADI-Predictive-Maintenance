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

## Audit-improve grouped validation study

The current audit uses five engine-level folds and predeclared 10/25/50/75/100-cycle pseudo-censoring endpoints. It does not read the official test files during selection.

```powershell
& '.\.venv\Scripts\python.exe' run_fd001_grouped_audit.py
& '.\.venv\Scripts\python.exe' run_cycle_free_ablation.py
& '.\.venv\Scripts\python.exe' finalize_grouped_audit.py
& '.\.venv\Scripts\python.exe' run_mlops_demo.py
& '.\.venv\Scripts\python.exe' write_audit_reports.py
```

Run the end-to-end local demo with:

```powershell
.\run_demo.ps1
```

`evaluate_grouped_audit_candidate.py` is deliberately separate. It is a historical official-test benchmark comparison, because the official FD001 results have already been observed. It must not be used to select a new feature, cap, or model.

## FD004 source-to-target transfer study

This train-on-FD004 study selects the source model only by FD004 grouped validation, using FD004-fitted six-condition normalization. It then evaluates the locked source model on FD001, FD002, and FD003 endpoints.

```powershell
& '.\.venv\Scripts\python.exe' run_fd004_transfer_study.py
& '.\.venv\Scripts\python.exe' write_fd004_transfer_docs.py
```

Results are saved under `reports/fd004_transfer/`. The target-subset labels are evaluation-only and never used for model or normalization fitting.

## Global train-all, test-all study

This is the requested single-model experiment: it combines every FD001--FD004
training trajectory, selects a configuration using only grouped validation on
that combined training population, then evaluates the locked model separately
on every official FD001--FD004 test file.

```powershell
& '.\.venv\Scripts\python.exe' run_all_subsets_global_study.py
& '.\.venv\Scripts\python.exe' build_all_subsets_global_pdf.py
```

Results are saved under `reports/all_subsets_global/`. The runner remaps engine
IDs before combining source files, which prevents rolling telemetry features
from crossing the original file boundaries.

The first two commands use only `train_FD001.txt`. The third command is the explicit official-test stage and loads `test_FD001.txt` and `RUL_FD001.txt` once for the validation-selected model.

## Outputs

- `reports/data_audit.md`, `metric_audit.md`, `experiment_plan.md`, `literature_review.md`, and `final_summary.md`
- `reports/pipeline_map.md`, `validation_protocol.md`, `baseline_reproduction.md`, `feature_target_ablation.csv`, and `predictions_oof.csv`
- `reports/metrics_summary_all_models.csv` and selected-model prediction/error files
- `reports/figures/` for actual-vs-predicted, residuals, per-engine error, and PHM08 contributions
- `dashboard.py` for an optional Streamlit view based only on saved files

## Dashboard

```powershell
& '.\.venv\Scripts\python.exe' -m pip install streamlit
& '.\.venv\Scripts\python.exe' -m streamlit run dashboard.py
```
