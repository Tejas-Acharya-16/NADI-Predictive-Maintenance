import unittest

import pandas as pd

from approaches.next_experiment_utils import full_range_endpoint_metadata


class NextExperimentUtilityTests(unittest.TestCase):
    def test_full_range_endpoints_are_causal_and_keep_engines_together(self):
        rows = []
        for unit, length in ((101, 201), (202, 151)):
            for cycle in range(1, length + 1):
                rows.append({"unit": unit, "cycle": cycle, "subset": "FD004", "source_unit": unit - 100})
        metadata = full_range_endpoint_metadata(pd.DataFrame(rows))
        self.assertEqual(set(metadata["unit"]), {101, 202})
        final_cycle = pd.DataFrame(rows).groupby("unit")["cycle"].max()
        self.assertTrue(
            (metadata["cutoff_cycle"] + metadata["actual_rul"] == metadata["unit"].map(final_cycle)).all()
        )
        self.assertTrue((metadata["actual_rul"] > 0).all())
        self.assertTrue((metadata["actual_rul"] > 125).any())


if __name__ == "__main__":
    unittest.main()
