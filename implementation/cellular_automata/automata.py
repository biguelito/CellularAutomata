import numpy as np
from math import ceil
from time import sleep

# from implementation.models.condition import Condition
from cellular_automata.statistic import Statistic
from cellular_automata.cell import Cell
from cellular_automata.grid import Grid
from models.seirsd import SEIRSD

class Automata:
    def __init__(self, size : int, seirsd : SEIRSD, initial_conditions: dict, quant_condition):
        self.size = size
        self.seirsd = seirsd
        self.initial_conditions = initial_conditions
        self.quant_condition = quant_condition

        self.beta = self.seirsd.get_initial_metrics("beta")
        self.sigma  = self.seirsd.get_initial_metrics("sigma")
        self.gamma = self.seirsd.get_initial_metrics("gamma")
        self.alfa = self.seirsd.get_initial_metrics("alfa")
        self.mu = self.seirsd.get_initial_metrics("mu")
        self.days_to_infection = ceil(1/self.sigma)
        self.days_to_lose_immunity = ceil(1/self.alfa)
        self.days_to_recover = ceil(1/self.gamma)

        self.conditions_effect = [
            self.susceptible_cell,
            self.exposed_cell,
            self.infected_cell,
            self.recovered_cell
        ]

        self.statistic = Statistic(quant_condition)        
        self.tick_count = 0
        self.cell = None

        self.create_population()
        return

    def create_population(self):
        self.matrix = [[Cell(self.quant_condition) for j in range(self.size)] for i in range(self.size)]
        self.matrix = np.array(self.matrix, dtype=object)
        self.statistic.increase_count(0, self.size * self.size)
        
        for condition, value in self.initial_conditions.items():
            self.initialize_random_condition(value, condition)
        return

    def initialize_random_condition(self, quantity, condition):
        total_susceptibles = self.matrix.size
        flat_indices = np.random.choice(total_susceptibles, quantity, replace=False)
        indices = np.unravel_index(flat_indices, (len(self.matrix), len(self.matrix)))
        for row, column in zip(indices[0], indices[1]):
            self.matrix[row, column].set_condition(condition)
        self.statistic.update_count(0, condition, -quantity, quantity)
        self.statistic.decrease_max_initial_condition(quantity)
        return

    def susceptible_cell(self, i, j):
        infected_neighbors = 0
        for x in range(max(0, i - 1), min(self.size, i + 2)):
            for y in range(max(0, j - 1), min(self.size, j + 2)):
                if self.matrix[x, y].get_condition() == 2:
                    infected_neighbors += 1

        if infected_neighbors > 0:
            prob_infection = 1 - ((1 - self.beta)**infected_neighbors)
            if (np.random.rand() < prob_infection):
                self.cell.set_condition(1)
        return

    def exposed_cell(self, i=0, j=0):
        self.cell.increase_ticks_in_condition(1)
        if (self.cell.days_in_condition[1] == self.days_to_infection):
            self.cell.set_condition(2)
        return

    def infected_cell(self, i=0, j=0):
        self.cell.increase_ticks_in_condition(2)
        prob_die = self.mu
        if (np.random.rand() < prob_die):
            self.cell.set_condition(4)
            return
        
        if (self.cell.days_in_condition[2] == self.days_to_recover):
            self.cell.set_condition(3)
        return

    def recovered_cell(self, i=0, j=0):
        self.cell.increase_ticks_in_condition(3)
        if (self.cell.days_in_condition[3] == self.days_to_lose_immunity):
            self.cell.set_condition(0)
        return

    def progress_condition(self, i, j):
        condition_old = self.cell.get_condition()
        if (condition_old == 4):
            return

        self.conditions_effect[condition_old](i, j)
        condition_new = self.cell.get_condition()
        if (condition_old != condition_new):
            self.statistic.update_count(condition_old, condition_new)
        return

    def tick(self):
        self.tick_count += 1
        for i in range(self.size):
            for j in range(self.size):
                self.cell = self.matrix[i, j]
                self.progress_condition(i, j)
        return

    def matrix_cell_to_condition(self):
        matrix = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                matrix[i, j] = self.matrix[i, j].get_condition()
        return matrix

    def create_interface(self, ticks, sleep_between_tick):
        all_matrices = []
        stats_per_tick = []
        stats_max = []
        all_matrices.append(self.matrix_cell_to_condition())
        stats_per_tick.append(list(self.statistic.conditions_in_tick))
        stats_max.append(list(self.statistic.conditions_max))

        for _ in range(ticks):
            self.tick()
            all_matrices.append(self.matrix_cell_to_condition())
            stats_per_tick.append(list(self.statistic.conditions_in_tick))
            stats_max.append(list(self.statistic.conditions_max))

        return Grid(self.quant_condition, all_matrices, stats_per_tick, stats_max, sleep_between_tick)

    def run(self, interface='iframe', ticks=0, sleep_between_tick=1):
        if (ticks == 0):
            ticks = self.seirsd.get_initial_metrics('ticks')
        
        if (interface == 'terminal'):
            self.terminal_interation(ticks, sleep_between_tick)
            return None
        
        grid = self.create_interface(ticks, sleep_between_tick)
        grid.show(interface)
        return

    def print_matrix(self):
        for i in range(self.size):
            for j in range(self.size):
                cell = self.matrix[i][j]
                print(cell, end=' ')
            print()
        print()
        return

    def interation(self):
        self.tick()
        print(f'tick: {self.tick_count}')
        self.print_matrix()
        self.statistic.print_statistics()
        return

    def terminal_interation(self, ticks, sleep_between_tick=1):
        for i in range(ticks):
            print("\033c", end="")
            self.interation()
            sleep(sleep_between_tick)
        return