import sys
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_saturation_figure as saturation


class SaturationFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = Namespace(
            systems=saturation.table.DEFAULT_SYSTEMS,
            references_dir=saturation.table.DEFAULT_REFERENCES,
            predictions_dir=saturation.table.DEFAULT_PREDICTIONS,
            cmudict=saturation.table.DEFAULT_CMUDICT,
            armeni_text=saturation.table.DEFAULT_ARMENI_TEXT,
            meg_masc_vocabulary=saturation.table.DEFAULT_MEG_MASC_VOCABULARY,
            context=saturation.DEFAULT_CONTEXT,
            main_table=saturation.table.DEFAULT_OUTPUT,
            appendix_table=saturation.table.DEFAULT_APPENDIX_OUTPUT,
            output_base=saturation.DEFAULT_OUTPUT_BASE,
            caption_output=saturation.DEFAULT_CAPTION_OUTPUT,
        )
        (
            cls.systems,
            cls.references,
            cls.entropies,
            cls.vocabularies,
            cls.points,
            cls.scores,
        ) = saturation.load_pipeline(cls.args)
        cls.curves = saturation.build_curves(cls.references["subtlex"])
        cls.context = saturation.load_context(cls.args.context)
        cls.plotted = saturation.build_plotted_points(
            cls.systems,
            cls.references["subtlex"],
            cls.vocabularies,
            cls.points,
            cls.context,
            cls.curves,
        )
        cls.intervals = saturation.interval_report(
            cls.systems,
            cls.references["subtlex"],
            cls.vocabularies,
            cls.context,
        )

    def test_frequency_curve_p95_values_are_stable(self):
        expected = {
            50: 0.977724736506,
            250: 0.972665814941,
            1_000: 0.969272434563,
            15_000: 0.963894265831,
            125_000: 0.957828832693,
        }
        for vocabulary_size, probability in expected.items():
            curve = self.curves[vocabulary_size]
            self.assertAlmostEqual(curve.p95, probability, places=10)
            self.assertAlmostEqual(
                saturation.ovmi(
                    self.references["subtlex"],
                    curve.vocabulary,
                    accuracy=curve.p95,
                ),
                saturation.SATURATION_FRACTION * curve.asymptote,
                places=10,
            )

    def test_system_points_match_own_vocabularies_not_representative_curves(self):
        self.assertEqual(len(self.plotted), 9)
        for item in self.plotted:
            own_score = saturation.ovmi(
                self.references["subtlex"],
                self.vocabularies[item.point.system_id],
                accuracy=item.point.probability,
            )
            self.assertAlmostEqual(item.score, own_score, places=10)
            self.assertAlmostEqual(item.score, item.point.csv_ovmi, places=10)
        self.assertTrue(
            all(
                abs(item.score - item.frequency_curve_score)
                > saturation.CHECK_TOLERANCE
                for item in self.plotted
            )
        )

    def test_card_willett_gap_exceeds_published_intervals(self):
        card = self.intervals["card_2024_v125k"]
        willett = self.intervals["willett_2023_v125k"]
        self.assertAlmostEqual(card[0] - willett[0], 2.117722346591, places=10)
        self.assertGreater(card[1], willett[2])

    def test_exact_main_table_agreement(self):
        saturation.assert_main_table_agreement(
            self.args.main_table,
            self.args.appendix_table,
            self.points,
            self.scores,
            self.entropies,
            self.vocabularies,
            self.references,
        )

    def test_outputs_and_caption_are_current(self):
        for extension in ("pdf", "png"):
            path = self.args.output_base.with_suffix(f".{extension}")
            self.assertTrue(path.exists(), msg=str(path))
            self.assertGreater(path.stat().st_size, 0, msg=str(path))
        caption = self.args.caption_output.read_text(encoding="utf-8")
        self.assertIn("not near-identical", caption)
        self.assertIn("95% of its own asymptote", caption)
        self.assertIn("Figure points omit uncertainty", caption)
        self.assertNotIn("[9.09", caption)


if __name__ == "__main__":
    unittest.main()
