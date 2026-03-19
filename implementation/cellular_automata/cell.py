from cellular_automata.condition import Condition

class Cell:
    def __init__(self):
        self.condition = Condition.SUSCEPTIBLE
        self.reset_days_count()

    def reset_days_count(self):
        self.days_exposed = 0
        self.days_infected = 0
        self.days_recovered = 0

    def get_condition(self):
        return self.condition

    def set_condition(self, condition):
        self.reset_days_count()
        self.condition = condition

    def increase_days_exposed(self):
        self.days_exposed += 1

    def increase_days_infected(self):
        self.days_infected += 1

    def increase_days_recovered(self):
        self.days_recovered += 1

    def __str__(self):
        return str(self.condition)