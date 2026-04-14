import numpy as np
import copy

from cellular_automata.cell import Cell
from cellular_automata.grid import Grid

class Automata:
    def __init__(self, model : object, initial_conditions: dict):
        self.model = model
        self.size = self.model.get_metrics("size")
        self.quant_condition = self.model.get_metrics("quant_condition")
        self.initial_conditions = initial_conditions

        self.tick_count = 0
        self.cell = None

        self.create_population()
        self.initial_state = copy.deepcopy(self.matrix)
        return
    
    def revert(self):
        self.matrix = copy.deepcopy(self.initial_state)

    def create_population(self):
        self.matrix = [[Cell(self.quant_condition) for j in range(self.size)] for i in range(self.size)]
        self.matrix = np.array(self.matrix, dtype=object)
        
        for condition, value in self.initial_conditions.items():
            self.initialize_random_condition(value, condition)
        return

    def initialize_random_condition(self, quantity, condition):
        total_susceptibles = self.matrix.size
        flat_indices = np.random.choice(total_susceptibles, quantity, replace=False)
        indices = np.unravel_index(flat_indices, (len(self.matrix), len(self.matrix)))
        for row, column in zip(indices[0], indices[1]):
            self.matrix[row, column].set_condition(condition)
        return

    def tick(self):
        self.tick_count += 1
        for i in range(self.size):
            for j in range(self.size):
                self.model.progress(self.matrix[i, j], self.size, i, j, self.matrix)
        return

    def matrix_cell_to_condition(self):
        matrix = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                matrix[i, j] = self.matrix[i, j].get_condition()
        return matrix

    def run(self):
        for _ in range(self.model.get_ticks()):
            self.tick()
        return

    def run_save(self, filename=None):
        self.run()
        grid = Grid(self.matrix_cell_to_condition(), self.quant_condition)
        grid.show(filename)
        self.revert()
