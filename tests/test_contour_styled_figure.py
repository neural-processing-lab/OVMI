import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_contour_figure as contour


class StyledContourFigureTests(unittest.TestCase):
    def test_every_styled_point_has_a_unique_shape_and_legend_label(self):
        styles = list(contour.STYLED_POINT_STYLES.values())
        self.assertEqual(len(styles), 10)
        self.assertEqual(len({marker for marker, _label in styles}), 10)
        self.assertEqual(len({label for _marker, label in styles}), 10)

    def test_styled_output_names_do_not_replace_primary_figure(self):
        self.assertEqual(
            contour.output_stem(["subtlex-uk"], False, False),
            "contour_subtlex",
        )
        self.assertEqual(
            contour.output_stem(["subtlex-uk"], False, False, styled=True),
            "contour_subtlex_styled",
        )

    def test_styled_outputs_exist(self):
        for suffix in ("", "_linear"):
            for extension in ("pdf", "png"):
                path = PROJECT_ROOT / f"figures/contour_subtlex_styled{suffix}.{extension}"
                self.assertTrue(path.exists(), msg=str(path))
                self.assertGreater(path.stat().st_size, 0, msg=str(path))


if __name__ == "__main__":
    unittest.main()
