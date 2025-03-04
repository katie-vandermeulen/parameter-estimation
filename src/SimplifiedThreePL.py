import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        self.experiment = experiment
        self._discrimination = None
        self._logit_base_rate = None
        self._is_fitted = False

    def summary(self):
        n_correct = np.sum(self.experiment.correct_responses)
        n_total = np.sum(self.experiment.trials)
        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_total - n_correct,
            "n_conditions": len(self.experiment.difficulties),
        }

    def predict(self, parameters):
        a, logit_c = parameters
        c = 1 / (1 + np.exp(-logit_c))  # Inverse logit transformation
        difficulties = self.experiment.difficulties
        return c + (1 - c) / (1 + np.exp(-a * (self.experiment.abilities - difficulties)))

    def negative_log_likelihood(self, parameters):
        probabilities = self.predict(parameters)
        likelihoods = self.experiment.correct_responses * np.log(probabilities) + \
                      (self.experiment.trials - self.experiment.correct_responses) * np.log(1 - probabilities)
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
