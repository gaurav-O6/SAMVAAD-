import unittest

from samvaad_braille import BrailleRecognizer, auto_space_text


class AutoSpaceTests(unittest.TestCase):
    def test_auto_space_splits_long_text(self):
        self.assertEqual(auto_space_text("hellothisis\ntestforbraille"), "hellothisis test for braille")

    def test_auto_space_preserves_short_words(self):
        self.assertEqual(auto_space_text("gaurav"), "gaurav")


class BrailleRecognizerTests(unittest.TestCase):
    def test_recognize_empty_image_returns_empty_text(self):
        recognizer = BrailleRecognizer()
        result = recognizer.analyze(__import__("numpy").zeros((64, 64, 3), dtype="uint8"))

        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.dots_count, 0)


if __name__ == "__main__":
    unittest.main()
