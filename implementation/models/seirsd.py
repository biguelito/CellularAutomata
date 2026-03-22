from compartmentals.CompartmentalModelSolver import CompartmentalModelSolver
from scipy.stats import truncnorm
import numpy as np

class SEIRSD:
    COMPARTMENTS = ["Susceptíveis", "Expostos", "Infectados", "Recuperados", "Mortos"]
    INITIAL_METRICS = {
        "S": 9985,
        "E": 10,
        "I": 5,
        "R": 0,
        "D": 0,
        "beta": 0.4332,
        "sigma": 0.192,
        "gamma": 0.141,
        "alfa": 0.03,
        "mu": 0.0014,
        "r0": 3,
        "ticks": 365,
        "size": 100
    }

    def __init__(self):
        pass

    def get_initial_metrics(self, key):
        return self.INITIAL_METRICS[key]

    def set_initial_metrics(self, key, value):
        self.INITIAL_METRICS[key] = value

    def update_metrics(self, new_metrics : dict):
        for key, value in new_metrics.items():
            if key not in new_metrics.keys():
                continue
            self.set_initial_metrics(key, value)