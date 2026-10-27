"""Reproduce the previously selected FD001 HGB with grouped validation.

This CPU-bounded runner intentionally evaluates only the FD001 validation
champion selected in the earlier full audit.  The full audit remains preserved
under ``reports/``; this script provides the requested FD001-first rerun before
the global FD001--FD004 study.
"""

from pathlib import Path

import run_fd001_grouped_audit as audit


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "domain_aware_next" / "fd001_first_champion"
ARTIFACTS = ROOT / "artifacts" / "domain_aware_next" / "fd001_first_champion"


def main() -> None:
    # Keep the selected champion and the audit's cap-125 control.  The control
    # is required by the shared report writer and makes the rerun comparison
    # interpretable without re-running the slow forest candidates.
    audit.CANDIDATES = tuple(
        candidate
        for candidate in audit.CANDIDATES
        if candidate.name in {"hgb_cap100_w10_30", "hgb_control_cap125_w10_30"}
    )
    result = audit.run(report_dir=REPORTS, artifact_dir=ARTIFACTS)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
