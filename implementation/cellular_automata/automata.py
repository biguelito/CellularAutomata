import numpy as np

from cellular_automata.states import States
from cellular_automata.cell import Cell
from cellular_automata.grid import Grid

class Automata:
    def __init__(self, model : object):
        self.model = model
        self.size = self.model.get_metrics("size")
        self.quant_condition = self.model.get_metrics("quant_condition")
        
        self.matrix = [[Cell(self.quant_condition) for j in range(self.size)] for i in range(self.size)]
        self.matrix = np.array(self.matrix, dtype=object)

        self.states = States(self.size, self.quant_condition)        
        self.states.increase_count(0, self.size * self.size)
        return

    def create_random_population(self, initial_conditions: dict):      
        for condition, value in initial_conditions.items():
            self.initialize_condition_random(value, condition)
        return

    def initialize_condition_random(self, quantity, condition):
        total_population = self.matrix.size
        flat_indices = np.random.choice(total_population, quantity, replace=False)
        indices = np.unravel_index(flat_indices, (len(self.matrix), len(self.matrix)))
        for row, column in zip(indices[0], indices[1]):
            self.matrix[row, column].set_condition(condition)

        self.states.update_count(0, condition, -quantity, quantity)
        self.states.decrease_max_initial_condition(quantity)
        return

    def check_condition(self, cell, i, j):
        condition_old = cell.get_condition()
        self.model.check(cell, self.size, i, j, self.matrix)

        condition_new = cell.get_condition()
        if (condition_old != condition_new):
            self.states.update_count(condition_old, condition_new)
        return

    def tick(self):
        for i in range(self.size):
            for j in range(self.size):
                self.check_condition(self.matrix[i, j], i, j)
        
        for i in range(self.size):
            for j in range(self.size):
                self.model.progress_marked(self.matrix[i, j])
        return

    def create_visualization(self, ticks, sleep_between_tick):
        self.states.record(self.matrix)
        for _ in range(ticks):
            self.tick()
            self.states.record(self.matrix)

        return Grid(self.quant_condition, self.states, sleep_between_tick)

    def run(self, ticks=0, sleep_between_tick=1, filename=None):
        ticks = self.model.get_metrics("ticks")
        grid = self.create_visualization(ticks, sleep_between_tick)
        grid.show(filename)
        return