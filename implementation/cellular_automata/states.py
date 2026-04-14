import numpy as np

class States:
    def __init__(self, size, quant_condition : int):
        self.size = size
        self.quant_condition = quant_condition
        self.conditions_in_tick = [0 for i in range(quant_condition)]
        self.conditions_max = [0 for i in range(quant_condition)]

        self.statistics_per_tick = []
        self.statistics_max = []
        self.matrices_per_tick = []
        return

    def increase_count(self, condition : int, value=1):
        self.conditions_in_tick[condition] += value
        self.update_max(condition)

    def update_count(self, condition_old : int, condition_new : int, increase_old=-1, increase_new=1):
        self.increase_count(condition_old, increase_old)
        self.increase_count(condition_new, increase_new)

    def update_max(self, condition : int):
        if self.conditions_in_tick[condition] > self.conditions_max[condition]:
            self.conditions_max[condition] = self.conditions_in_tick[condition]

    def decrease_max_initial_condition(self, value : int):
        self.conditions_max[0] -= value

    def matrix_cell_to_condition(self, matrix_cells):
        matrix = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                matrix[i, j] = matrix_cells[i, j].get_condition()
        return matrix

    def record(self, matrix_cell):
        self.statistics_per_tick.append(list(self.conditions_in_tick))
        self.statistics_max.append(list(self.conditions_max))
        self.matrices_per_tick.append(self.matrix_cell_to_condition(matrix_cell))