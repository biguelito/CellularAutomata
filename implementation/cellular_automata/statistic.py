from cellular_automata.condition import Condition

class Statistic:
    def __init__(self):
        self.conditions_in_tick = {
            Condition.SUSCEPTIBLE: 0,
            Condition.EXPOSED: 0,
            Condition.INFECTED: 0,
            Condition.RECOVERED: 0, 
            Condition.DEAD: 0
        }
        self.conditions_max = {
            Condition.SUSCEPTIBLE: 0,
            Condition.EXPOSED: 0,
            Condition.INFECTED: 0,
            Condition.RECOVERED: 0, 
            Condition.DEAD: 0
        }

    def increase_count(self, condition : Condition, value=1):
        self.conditions_in_tick[condition] += value
        self.increase_max(condition)

    def update_count(self, condition_old : Condition, condition_new : Condition, increase_old=-1, increase_new=1):
        self.increase_count(condition_old, increase_old)
        self.increase_count(condition_new, increase_new)

    def increase_max(self, condition : Condition):
        if self.conditions_in_tick[condition] > self.conditions_max[condition]:
            self.conditions_max[condition] = self.conditions_in_tick[condition]

    def decrease_max_susceptible(self, value : int):
        self.conditions_max[Condition.SUSCEPTIBLE] -= value

    def print_statistics(self):
        print(f'Quantidades [ atual | maximo ]')
        print(f'Suscetiveis [{Condition.SUSCEPTIBLE}]: {self.conditions_in_tick[Condition.SUSCEPTIBLE]} | {self.conditions_max[Condition.SUSCEPTIBLE]}')
        print(f'Expostos [{Condition.EXPOSED}]: {self.conditions_in_tick[Condition.EXPOSED]} | {self.conditions_max[Condition.EXPOSED]}')
        print(f'Infectados [{Condition.INFECTED}]: {self.conditions_in_tick[Condition.INFECTED]} | {self.conditions_max[Condition.INFECTED]}')
        print(f'Recuperados [{Condition.RECOVERED}]: {self.conditions_in_tick[Condition.RECOVERED]} | {self.conditions_max[Condition.RECOVERED]}')
        print(f'Mortos [{Condition.DEAD}]: {self.conditions_in_tick[Condition.DEAD]} | {self.conditions_max[Condition.DEAD]}')