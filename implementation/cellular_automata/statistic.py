class Statistic:
    def __init__(self, quant_condition : int):
        self.quant_condition = quant_condition
        self.conditions_in_tick = [0 for i in range(quant_condition)]
        self.conditions_max = [0 for i in range(quant_condition)]

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

    def print_statistics(self):
        print(f'Quantidades [ atual | maximo ]')
        for i in range(self.quant_condition):
            print(f'[{i}]: {self.conditions_in_tick[i]} | {self.conditions_max[i]}')
