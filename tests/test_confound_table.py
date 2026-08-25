import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/make_confound_table.py"
CSV_PATH = ROOT / "data/confounds.csv"
TABLE_PATH = ROOT / "tables/confounds.tex"


class ConfoundTableTests(unittest.TestCase):
    def test_generator_is_current_and_checks_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "confounds.tex"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(output)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(output.read_text(), TABLE_PATH.read_text())
            self.assertIn("10 rows match exactly", result.stdout)
            self.assertIn("90 cells covered", result.stdout)
            self.assertIn("calibration_burden: 7/10 (70.0%)", result.stdout)
            self.assertIn("split_discipline: 5/10 (50.0%)", result.stdout)
            self.assertIn("hours_per_participant: 1/10 (10.0%)", result.stdout)
            self.assertIn("language_model: 1/10 (10.0%)", result.stdout)

    def test_reference_table_has_no_result_encoding(self):
        table = TABLE_PATH.read_text()
        self.assertIn(r"\label{tab:confounds}", table)
        self.assertEqual(table.count(r"\begin{tabularx}"), 2)
        self.assertNotIn("OVMI", table)
        self.assertNotIn(r"\cellcolor", table)
        self.assertNotIn("|", table)

    def test_csv_uses_only_explicit_em_dash_for_missing_values(self):
        with CSV_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        self.assertFalse(any(value == "" for row in rows for value in row.values()))
        self.assertTrue(any(value == "—" for row in rows for value in row.values()))


if __name__ == "__main__":
    unittest.main()
