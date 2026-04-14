import numpy as np
from math import ceil

class SEIRSD:
    def __init__(self):
        self.metrics = {
            "E": 10,
            "I": 5,
            "R": 0,
            "D": 0,
            "beta": 0.4332,
            "sigma": 0.192,
            "gamma": 0.141,
            "alfa": 0.0056,
            "mu": 0.0014,
            "r0": 3,
            "ticks": 365,
            "size": 100,
            "quant_condition": 5
        }
        self.calculate_indices()
        self.conditions_effect = [
            self.susceptible_cell,
            self.exposed_cell,
            self.infected_cell,
            self.recovered_cell
        ]

    def calculate_indices(self):
        self.days_to_infection = ceil(1/self.metrics['sigma'])
        self.days_to_lose_immunity = ceil(1/self.metrics['alfa'])
        self.days_to_recover = ceil(1/self.metrics['gamma'])
        self.probability_die = self.metrics['mu']
        self.beta = self.metrics['beta']

    def get_metrics(self, key):
        return self.metrics[key]

    def get_ticks(self):
        return self.metrics['ticks']

    def set_metrics(self, key, value):
        self.metrics[key] = value

    def update_metrics(self, new_metrics : dict):
        for key, value in new_metrics.items():
            if key not in new_metrics.keys():
                continue
            self.set_metrics(key, value)
        self.calculate_indices()
    
    def progress(self, cell, size, i, j, matrix):
        condition_old = cell.get_condition()
        if (condition_old == 4):
            return
        self.conditions_effect[cell.get_condition()](cell, size, i, j, matrix)

    def susceptible_cell(self, cell, size, i, j, matrix):
        infected_neighbors = 0
        for x in range(max(0, i - 1), min(size, i + 2)):
            for y in range(max(0, j - 1), min(size, j + 2)):
                if matrix[x, y].get_condition() == 2:
                    infected_neighbors += 1

        if infected_neighbors > 0:
            prob_infection = 1 - ((1 - self.beta)**infected_neighbors)
            if (np.random.rand() < prob_infection):
                cell.set_condition(1)
        return

    def exposed_cell(self, cell, size=0, i=0, j=0, matrix=None):
        cell.increase_ticks_in_condition(1)
        if (cell.days_in_condition[1] == self.days_to_infection):
            cell.set_condition(2)
        return

    def infected_cell(self, cell, siz=0, i=0, j=0, matrix=None):
        cell.increase_ticks_in_condition(2)
        if (np.random.rand() < self.probability_die):
            cell.set_condition(4)
            return
        
        if (cell.days_in_condition[2] == self.days_to_recover):
            cell.set_condition(3)
        return

    def recovered_cell(self, cell, size=0, i=0, j=0, matrix=None):
        cell.increase_ticks_in_condition(3)
        if (cell.days_in_condition[3] == self.days_to_lose_immunity):
            cell.set_condition(0)
        return