class Cell:
    def __init__(self, quant_condition):
        self.condition = 0
        self.quant_condition = quant_condition
        self.reset_days_count()

    def reset_days_count(self):
        self.days_in_condition = [0 for i in range(self.quant_condition)]

    def get_condition(self):
        return self.condition

    def set_condition(self, condition):
        self.reset_days_count()
        self.condition = condition

    def increase_ticks_in_condition(self, condition):
        self.days_in_condition[condition] += 1

    def __str__(self):
        return str(self.condition)