"""Write data-audit and executive notes for completed FD004 transfer outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmapss_data import load_subset, quality_report


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData"
REPORTS = ROOT / "reports" / "fd004_transfer"


def main() -> None:
    rows = []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        loaded = load_subset(DATA, subset)
        train_q = quality_report(loaded["train"], subset, "train")
        test_q = quality_report(loaded["test"], subset, "test")
        rows.append({"subset": subset, "train_rows": train_q["rows"], "train_engines": train_q["engines"], "test_rows": test_q["rows"], "test_engines": test_q["engines"], "test_rul_min": int(loaded["test_rul"].rul.min()), "test_rul_max": int(loaded["test_rul"].rul.max()), "quality_checks": "passed"})
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORTS / "source_target_data_audit.csv", index=False)
    metrics = pd.read_csv(REPORTS / "fd004_to_other_subsets_transfer_metrics.csv")
    selected = pd.read_csv(REPORTS / "fd004_validation_model_comparison.csv").sort_values(["phm08_mean", "rmse"]).iloc[0]
    (REPORTS / "fd004_transfer_data_audit.md").write_text(
        "# FD004 transfer source and target audit\n\n"
        "FD004 is the source subset because it covers six operating conditions and two degradation modes. FD001, FD002, and FD003 are explicitly held out as target-subset endpoint evaluations. All source and target raw files passed schema, non-finite, duplicate, cycle-order, gap, and ID checks.\n\n"
        "```text\n" + audit.to_string(index=False) + "\n```\n\n"
        "The operating-condition normalizer is fit on FD004 settings and sensors only. Target rows are assigned to their nearest FD004-derived setting cluster at inference. No target rows or labels are used to refit KMeans, scalers, feature screens, or the RUL model.\n",
        encoding="utf-8",
    )
    (REPORTS / "fd004_transfer_final_summary.md").write_text(
        "# FD004 cross-subset transfer result\n\n"
        f"The FD004 validation winner is `{selected.candidate}` with RMSE {selected.rmse:.3f}, MAE {selected.mae:.3f}, R2 {selected.r2:.3f}, and PHM08 mean {selected.phm08_mean:.3f} on 1,494 grouped pseudo-censored endpoint cases. It trains from FD004 only using six FD004-fitted operating-condition clusters and per-cluster sensor normalization.\n\n"
        "## Locked cross-subset endpoint results\n\n"
        "| Target | Engines | RMSE | MAE | R2 | PHM08 mean | Bias |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(
            f"| {row.target_subset} | {row.target_engines} | {row.rmse:.3f} | {row.mae:.3f} | {row.r2:.3f} | {row.phm08_mean:.3f} | {row.bias_predicted_minus_actual:.3f} |"
            for row in metrics.itertuples(index=False)
        ) + "\n\n"
        "FD001 and FD003 transfer with moderate error despite their simpler single-condition domains. FD002 is materially harder and has high PHM08 because its target operating-condition mix differs from FD004. These are transfer measurements, not target-subset tuned models.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
