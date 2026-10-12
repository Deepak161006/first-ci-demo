
import json
import joblib
import unittest
import pandas as pd
from pathlib import Path


class TestMLPipeline(unittest.TestCase):

    def test_output_files_exist(self):
        for filename in [
            "student_results.csv",
            "student_result_model.pkl",
            "metrics.json"
        ]:
            self.assertTrue(Path(filename).exists())

    def test_metrics_are_valid(self):
        with open("metrics.json") as f:
            metrics = json.load(f)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80
        }])

        prediction = model.predict(sample)
        self.assertIn(int(prediction[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
