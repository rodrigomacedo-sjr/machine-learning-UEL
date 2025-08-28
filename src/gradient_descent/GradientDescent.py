from typing import List, Tuple
import math


class GradientDescentCalculator:

    def __init__(
        self,
        training_set: List[Tuple[float, ...]],
        learning_rate: float = 0.5,
        max_iterations: int = 20000,
        tolerance_criteria: int = 8,
    ):
        """
        Args:
            training_set (List[Tuple[float, ...]]): List of n-dimensional tuples representing coordinates on a n-dimensional space
            learning_rate (float): the size of the steps taken in the training process
            max_iterations (int): ceiling number of iterations in which the training stops
            tolerance_criteria (int): when the difference between steps is smaller than 10**(-tolerance_criteria), the trainig stops
        """
        self.trainig_set = training_set
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance_criteria = tolerance_criteria

    def calculate_gradient_descent(self):
        initial_beta = 0.1
        predicted_values = _calculate_logistic_function(self.training_set@self.initial_beta)
        pass

    def _calculate_initial_beta(self):
        pass

    def _calculate_logistic_function(self, z):
        return 1 / (1 + math.e ** (-z))
