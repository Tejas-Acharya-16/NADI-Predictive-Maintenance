param([switch]$IncludeHistoricalBenchmark)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python run_fd001_grouped_audit.py
& $python run_cycle_free_ablation.py
& $python finalize_grouped_audit.py
& $python run_mlops_demo.py
& $python run_anomaly_triage_demo.py
& $python write_audit_reports.py
& $python -m unittest discover -s tests -v
if ($IncludeHistoricalBenchmark) {
    & $python evaluate_grouped_audit_candidate.py
}
