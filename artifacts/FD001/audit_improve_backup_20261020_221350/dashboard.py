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
    st.set_page_config(page_title="C-MAPSS FD001", layout="wide")
    st.title("NASA C-MAPSS FD001 predictive-maintenance demo")
    st.caption("Uses saved experiment artifacts; this is a simulated benchmark, not an aircraft-maintenance system.")
    for title, file_name in [("Tree ablations", "metrics_summary.csv"), ("Sequence experiments", "sequence_metrics_summary.csv")]:
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


if __name__ == "__main__":
    main()
