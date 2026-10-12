
import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85,
                "assignment_marks": 88,
                "previous_score": 80
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"attendance": 90}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
