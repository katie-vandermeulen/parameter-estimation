import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
    #    self.difficulties = np.array([2, 1, 0, -1, -2])
    #    self.trials = np.array([100, 100, 100, 100, 100])
    #    self.correct_responses = np.array([55, 60, 75, 90, 95])
       self.experiment = Experiment()
       self.experiment.add_condition(SignalDetection(15, 5, 27, 8))
       self.experiment.add_condition(SignalDetection(13, 6, 23, 9))
       self.experiment.add_condition(SignalDetection(17, 7, 10, 4))
       self.experiment.add_condition(SignalDetection(1, 8, 34, 6))
       self.experiment.add_condition(SignalDetection(10, 9, 30, 3)) 
       self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        self.assertFalse(self.model._is_fitted)

    def test_summary(self):
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 245)
        self.assertEqual(summary["n_correct"], 86)
        self.assertEqual(summary["n_incorrect"], 159)
        self.assertEqual(summary["n_conditions"], 5)

    def test_predict(self):
        params = [1, 0]
        predictions = self.model.predict(params)
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

    def test_predict_c(self):
        params1 = [1.2, 0]
        prediction1 = self.model.predict(params1)
        params2 = [1.2, 1]
        prediction2 = self.model.predict(params2)
        for p1, p2 in zip(prediction1, prediction2):
            self.assertLess(p1, p2)

    def test_predict_b(self):
        params1 = [-1.2, 0]
        prediction1 = self.model.predict(params1)
        params2 = [-1.1, 1]
        prediction2 = self.model.predict(params2)
        for p1, p2 in zip(prediction1, prediction2):
            self.assertLess(p1, p2)

    def test_predict_theta(self):
        params1 = [1.2, 0]
        prediction1 = self.model.predict(params1)
        params2 = [2.2, 1]
        prediction2 = self.model.predict(params2)
        for p1, p2 in zip(prediction1, prediction2):
            self.assertLess(p1, p2)

    def test_predict_expectedoutput(self):
        params1 = [3, 0]
        prediction1 = self.model.predict(params1)
        expectoutput = [0.50123631, 0.52371294, 0.75, 0.97628706, 0.99876369]
        for p1, e in zip(prediction1, expectoutput):
            self.assertAlmostEqual(p1, e, places = 2)

    def test_negative_log_likelihood(self):
        initial_nll = self.model.negative_log_likelihood([1, 0])
        self.model.fit()
        fitted_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model.get_base_rate()])
        self.assertLess(fitted_nll, initial_nll)

# test_larger_a code pulled from ChatGPT
    def test_larger_a(self):
        self.steepExperiment = Experiment()
        self.steepExperiment.add_condition(SignalDetection(10, 90, 30, 870))
        self.steepExperiment.add_condition(SignalDetection(100, 20, 40, 800))
        self.steepExperiment.add_condition(SignalDetection(300, 10, 10, 680))
        self.steepExperiment.add_condition(SignalDetection(600, 5, 5, 390))
        self.steepExperiment.add_condition(SignalDetection(900, 1, 1, 100))

        self.steepModel = SimplifiedThreePL(self.steepExperiment)
        self.steepModel.fit() 
        
        self.experiment = Experiment() 
        self.experiment.add_condition(SignalDetection(15, 5, 27, 8)) 
        self.experiment.add_condition(SignalDetection(13, 6, 23, 9)) 
        self.experiment.add_condition(SignalDetection(17, 7, 10, 4)) 
        self.experiment.add_condition(SignalDetection(1, 8, 34, 6)) 
        self.experiment.add_condition(SignalDetection(10, 9, 30, 3)) 

        self.model = SimplifiedThreePL(self.experiment) 
        self.model.fit()
        
        self.assertGreater(self.steepModel.get_discrimination(), self.model.get_discrimination()) 


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
    
    def test_integration(self):
        pass

    def test_model_does_not_corrupt_experiment(self):
        original_experiment_data = self.experiment.conditions.copy()
        self.model.fit()
        self.assertEqual(self.experiment.conditions, original_experiment_data, "Expiriment data was modified!")

    def test_predict_does_not_modify_parameters(self):
        self.model.fit()
        before_a = self.model.get_discrimination()
        before_c = self.model.get_base_rate()

        self.model.predict([1, 0]) 

        after_a = self.model.get_discrimination()
        after_c = self.model.get_base_rate()

        self.assertEqual(before_a, after_a, "Discrimination parameter modified!")
        self.assertEqual(before_c, after_c, "Base rate parameter modified!")

    def test_nll_does_not_modify_experiment(self):
        before_conditions = self.experiment.conditions.copy()
        self.model.negative_log_likelihood([1, 0])
        after_conditions = self.experiment.conditions.copy()

        self.assertEqual(before_conditions, after_conditions, "Experiment conditions modified by NLL calculation!")

if __name__ == "__main__":
    unittest.main()

