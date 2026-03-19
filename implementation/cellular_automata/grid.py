import numpy as np
from math import ceil
from time import sleep

from cellular_automata.condition import Condition
from cellular_automata.statistic import Statistic
from cellular_automata.cell import Cell
from models.seirsd import SEIRSD

class Grid:
    def __init__(self, size : int, seirsd : SEIRSD):
        self.size = size
        self.seirsd = seirsd

        self.beta = self.seirsd.get_initial_metrics("beta")
        self.sigma  = self.seirsd.get_initial_metrics("sigma")
        self.gamma = self.seirsd.get_initial_metrics("gamma")
        self.alfa = self.seirsd.get_initial_metrics("alfa")
        self.mu = self.seirsd.get_initial_metrics("mu")
        self.days_to_infection = ceil(1/self.sigma)
        self.days_to_lose_immunity = ceil(1/self.alfa)
        self.days_to_recover = ceil(1/self.gamma)

        self.conditions_effect = {
            Condition.SUSCEPTIBLE: self.susceptible_cell,
            Condition.EXPOSED: self.exposed_cell,
            Condition.INFECTED: self.infected_cell,
            Condition.RECOVERED: self.recovered_cell
        }

        self.statistic = Statistic()        
        self.tick_count = 0
        self.cell = None

        self.create_population()
        return

    def create_population(self):
        self.grid = [[Cell() for j in range(self.size)] for i in range(self.size)]
        self.grid = np.array(self.grid, dtype=object)
        self.statistic.increase_count(Condition.SUSCEPTIBLE, self.size * self.size)
        return

    def print_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i][j]
                print(cell, end=' ')
            print()
        print()
        return

    def initialize_random_condition(self, quantity, condition):
        total_susceptibles = self.grid.size
        flat_indices = np.random.choice(total_susceptibles, quantity, replace=False)
        indices = np.unravel_index(flat_indices, (len(self.grid), len(self.grid)))
        for row, column in zip(indices[0], indices[1]):
            self.grid[row, column].set_condition(condition)
        self.statistic.update_count(Condition.SUSCEPTIBLE, condition, -quantity, quantity)
        self.statistic.decrease_max_susceptible(quantity)
        return

    def susceptible_cell(self, i, j):
        infected_neighbors = 0
        for x in range(max(0, i - 1), min(self.size, i + 2)):
            for y in range(max(0, j - 1), min(self.size, j + 2)):
                if self.grid[x, y].get_condition() == Condition.INFECTED:
                    infected_neighbors += 1

        if infected_neighbors > 0:
            prob_infection = 1 - ((1 - self.beta)**infected_neighbors)
            if (np.random.rand() < prob_infection):
                self.cell.set_condition(Condition.EXPOSED)
        return

    def exposed_cell(self, i=0, j=0):
        self.cell.increase_days_exposed()
        if (self.cell.days_exposed == self.days_to_infection):
            self.cell.set_condition(Condition.INFECTED)
        return

    def infected_cell(self, i=0, j=0):
        self.cell.increase_days_infected()
        prob_die = self.mu
        if (np.random.rand() < prob_die):
            self.cell.set_condition(Condition.DEAD)
            return
        
        if (self.cell.days_infected == self.days_to_recover):
            self.cell.set_condition(Condition.RECOVERED)
        return

    def recovered_cell(self, i=0, j=0):
        self.cell.increase_days_recovered()
        if (self.cell.days_recovered == self.days_to_lose_immunity):
            self.cell.set_condition(Condition.SUSCEPTIBLE)
        return

    def progress_condition(self, i, j):
        condition_old = self.cell.get_condition()
        if (condition_old == Condition.DEAD):
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
                self.cell = self.grid[i, j]
                self.progress_condition(i, j)
        return
        
    def interation(self):
        self.tick()
        print(f'tick: {self.tick_count}')
        self.print_grid()
        self.statistic.print_statistics()
        return

    def loop_interation(self, ticks=0, sleep_between_tick=3):
        if (ticks == 0):
            ticks = self.seirsd.get_initial_metrics('ticks')

        for i in range(ticks):
            print("\033c", end="")
            self.interation()
            sleep(sleep_between_tick)
        return
