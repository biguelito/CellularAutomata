class Cell:
    def __init__(self, quant_condition):
        self.condition = 0
        self.marked_condition = 0
        self.quant_condition = quant_condition
        self.reset_days_count()

    def reset_days_count(self):
        self.days_in_condition = [0 for i in range(self.quant_condition)]

    def get_condition(self):
        return self.condition

    def get_marked_condition(self):
        return self.marked_condition

    def set_condition(self, condition):
        self.reset_days_count()
        self.condition = condition
        self.marked_condition = condition

    def set_condition_marked(self):
        self.condition = self.marked_condition

    def mark_condition(self, condition):
        self.reset_days_count()
        self.marked_condition = condition

    def increase_ticks_in_condition(self, condition):
        self.days_in_condition[condition] += 1

    def __str__(self):
        return str(self.condition)
