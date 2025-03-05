import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        self.experiment = experiment
        self._discrimination = None
        self._logit_base_rate = None
        self._is_fitted = False
        self._base_rate = None
        self.abilities = 0
        self.difficulties = np.array([2, 1, 0, -1, -2])

    def summary(self):
        n_correct = sum(conditions.hits + conditions.correct_rejections for conditions in self.experiment.conditions)
        n_total = sum(conditions.misses + conditions.false_alarms + conditions.hits + conditions.correct_rejections for conditions in self.experiment.conditions)
        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_total - n_correct,
            "n_conditions": len(self.experiment.conditions),
        }

    def predict(self, parameters):
        results = []
        for b in self.difficulties:
            a, logit_c = parameters
            c = 1 / (1 + np.exp(-logit_c))  # Inverse logit transformation
            results.append(c + ((1 - c) / (1 + np.exp(-a * (self.abilities - b)))))
        return np.array(results)

    def negative_log_likelihood(self, parameters):
        probabilities = self.predict(parameters)
        summary = self.summary()
        likelihoods = summary["n_correct"] * np.log(probabilities) + \
                      summary["n_incorrect"] * np.log(1 - probabilities)
        return -np.sum(likelihoods)

    def fit(self):
        result = minimize(self.negative_log_likelihood, [1, 0], method="L-BFGS-B")
        self._discrimination, self._logit_base_rate = result.x
        self._is_fitted = True

    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return 1 / (1 + np.exp(-self._logit_base_rate))
