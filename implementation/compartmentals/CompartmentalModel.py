class CompartmentalModel:
    def __init__(self, initial_values, transfer_rates, days, compartments=None):
        self.initialValues = initial_values
        self.transfer_rates = transfer_rates
        self.days = days
        self.compartments = compartments if compartments else [f'C{i}' for i in range(len(initial_values))]
