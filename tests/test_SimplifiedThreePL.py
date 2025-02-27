import unittest
import numpy as np
from SimplifiedThreePL import SimplifiedThreePL
from Experiment import Experiment

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        self.difficulties = np.array([2, 1, 0, -1, -2])
        self.trials = np.array([100, 100, 100, 100, 100])
        self.correct_responses = np.array([55, 60, 75, 90, 95])
        self.experiment = Experiment(self.difficulties, self.trials, self.correct_responses)
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        self.assertFalse(self.model._is_fitted)

    def test_summary(self):
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 500)
        self.assertEqual(summary["n_correct"], 375)
        self.assertEqual(summary["n_incorrect"], 125)
        self.assertEqual(summary["n_conditions"], 5)

    def test_predict(self):
        params = [1, 0]
        predictions = self.model.predict(params)
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

    def test_negative_log_likelihood(self):
        initial_nll = self.model.negative_log_likelihood([1, 0])
        self.model.fit()
        fitted_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model.get_base_rate()])
        self.assertLess(fitted_nll, initial_nll)

    def test_get_parameters_before_fit(self):
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()

    def test_fit(self):
        self.model.fit()
        self.assertTrue(self.model._is_fitted)
        self.assertIsInstance(self.model.get_discrimination(), float)
        self.assertIsInstance(self.model.get_base_rate(), float)

if __name__ == "__main__":
    unittest.main()
