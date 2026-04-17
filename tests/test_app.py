import io
import importlib.util
import unittest

import cv2
import numpy as np
import werkzeug

class AppRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("flask") is None:
            raise unittest.SkipTest("Flask is not installed in this interpreter")

        if not hasattr(werkzeug, "__version__"):
            werkzeug.__version__ = "patched-for-tests"

        from templates.app import app

        cls.client = app.test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"SAMVAAD", response.data)
        finally:
            response.close()

    def test_process_landmarks_accepts_empty_list(self):
        response = self.client.post("/process_landmarks", json=[])
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"text": ""})
        finally:
            response.close()

    def test_braille_endpoint_requires_image(self):
        response = self.client.post("/api/recognize-braille", data={})
        try:
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["error"], "No image uploaded")
        finally:
            response.close()

    def test_braille_endpoint_handles_blank_image(self):
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        response = self.client.post(
            "/api/recognize-braille",
            data={"image": (io.BytesIO(encoded.tobytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["error"], "No Braille detected in image")
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
