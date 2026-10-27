"""Optional Streamlit view of reproducible FD001 report artifacts."""

from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as error:
        raise SystemExit("Install optional dashboard dependencies: pip install streamlit") from error
    st.set_page_config(page_title="C-MAPSS predictive maintenance", layout="wide")
    st.title("NASA C-MAPSS predictive-maintenance demo")
    st.caption("Uses saved experiment artifacts; this is a simulated benchmark, not an aircraft-maintenance system.")
    st.warning("FD001 is simulated turbofan telemetry. Triage thresholds are demo placeholders, not certified maintenance limits.")
    mission = REPORT_DIR / "domain_aware_mission"
    if mission.exists():
        st.header("Domain-aware global fleet study")
        st.caption("Official test results are shown as historical audit results because they were inspected before this study. Model selection uses grouped validation only.")
        for title, name in [
            ("Raw-target and capped-target audit", "audit_raw_and_capped_metrics.csv"),
            ("Specialist HGB validation", "specialist_validation_comparison.csv"),
            ("Specialist historical official metrics", "specialist_official_metrics.csv"),
            ("Transfer validation", "transfer_validation_metrics.csv"),
            ("Compact TCN validation", "compact_tcn_validation_metrics.csv"),
            ("Competition work orders", "competition_work_orders.csv"),
        ]:
            path = mission / name
            if path.exists():
                st.subheader(title)
                st.dataframe(pd.read_csv(path), use_container_width=True)
        for figure in sorted((mission / "figures").glob("*.png")) if (mission / "figures").exists() else []:
            st.image(str(figure), caption=figure.stem)
    for title, file_name in [("Grouped validation tree ablations", "metrics_summary.csv"), ("Sequence experiments", "sequence_metrics_summary.csv")]:
        path = REPORT_DIR / file_name
        if path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(path), use_container_width=True)
    prediction_path = REPORT_DIR / "predictions_selected_hgb_full_training.csv"
    if prediction_path.exists():
        prediction = pd.read_csv(prediction_path)
        st.subheader("Selected model: official FD001 endpoint predictions")
        st.scatter_chart(prediction, x="actual_rul", y="predicted_rul")
        st.dataframe(prediction.sort_values("absolute_error", ascending=False), use_container_width=True)
    order_path = REPORT_DIR / "triage_demo_work_orders.json"
    if order_path.exists():
        st.subheader("Rule-based mock work orders")
        st.dataframe(pd.DataFrame(json.loads(order_path.read_text(encoding="utf-8"))), use_container_width=True)
    mlops_path = REPORT_DIR / "mlops_demo.json"
    if mlops_path.exists():
        mlops = json.loads(mlops_path.read_text(encoding="utf-8"))
        st.subheader("Data validation, drift, and retraining demonstration")
        st.caption(mlops["scope"])
        left, right = st.columns(2)
        left.dataframe(pd.DataFrame(mlops["top_drift_features"]), use_container_width=True)
        right.dataframe(pd.DataFrame(mlops["retraining_history"]), use_container_width=True)
    audit_test = REPORT_DIR / "grouped_audit_historical_official_test_metrics.json"
    if audit_test.exists():
        st.subheader("Locked candidate historical benchmark comparison")
        st.json(json.loads(audit_test.read_text(encoding="utf-8")))
    transfer_path = REPORT_DIR / "fd004_transfer" / "fd004_to_other_subsets_transfer_metrics.csv"
    if transfer_path.exists():
        st.subheader("FD004 source model transfer to other C-MAPSS subsets")
        st.caption("The FD004 model is trained on FD004 only. FD001, FD002, and FD003 are endpoint transfer evaluations, not target-tuned models.")
        st.dataframe(pd.read_csv(transfer_path), use_container_width=True)


if __name__ == "__main__":
    main()
