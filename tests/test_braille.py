import unittest

from samvaad_braille import BrailleRecognizer, BrailleCell, auto_space_text, decode_cells


class AutoSpaceTests(unittest.TestCase):
    def test_auto_space_splits_long_text(self):
        self.assertEqual(auto_space_text("hellothisis\ntestforbraille"), "hellothisis test for braille")

    def test_auto_space_preserves_short_words(self):
        self.assertEqual(auto_space_text("gaurav"), "gaurav")

    def test_auto_space_merges_split_line_boundary_words(self):
        self.assertEqual(auto_space_text("tempor incid\nidunt ut"), "tempor incididunt ut")


class BrailleRecognizerTests(unittest.TestCase):
    def test_recognize_empty_image_returns_empty_text(self):
        recognizer = BrailleRecognizer()
        result = recognizer.analyze(__import__("numpy").zeros((64, 64, 3), dtype="uint8"))

        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.dots_count, 0)


class DecodeSpacingTests(unittest.TestCase):
    def test_decode_cells_infers_space_from_large_gap(self):
        cells = [
            BrailleCell(col=0, row=0, pattern=(1, 1, 1, 0, 0, 0), bbox=(0, 0, 10, 20)),   # l
            BrailleCell(col=1, row=0, pattern=(1, 0, 1, 0, 1, 0), bbox=(28, 0, 10, 20)),  # o
            BrailleCell(col=2, row=0, pattern=(1, 1, 1, 0, 1, 0), bbox=(56, 0, 10, 20)),  # r
            BrailleCell(col=3, row=0, pattern=(1, 0, 0, 0, 1, 0), bbox=(106, 0, 10, 20)), # e
        ]

        self.assertEqual(decode_cells(cells), "lor e")


if __name__ == "__main__":
    unittest.main()
