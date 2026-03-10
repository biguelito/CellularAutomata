from scipy.integrate import odeint
import numpy as np
from compartmentals.CompartmentalGraph import CompartmentalGraph
from compartmentals.CompartmentalModel import CompartmentalModel

class CompartmentalModelSolver:
    def __init__(self, ode_function, initial_conditions, transfer_rates, days, compartments, model_name):
        self.model = CompartmentalModel(initial_conditions, transfer_rates, days, compartments)
        self.ode_function = ode_function
        self.solved_odes = None
        self.tspan = None
        self.model_name = model_name

    def solve(self):
        self.tspan = np.arange(0, self.model.days, 1)
        solved_odes = odeint(
            self.ode_function,
            self.model.initialValues,
            self.tspan,
            args=(self.model.transfer_rates,)
        )
        self.solved_odes = solved_odes

    def get_figure(self):
        curves = {
            name: self.solved_odes[:, idx]
            for idx, name in enumerate(self.model.compartments)
        }
        fig = CompartmentalGraph(self.tspan, curves, self.model_name)
        return fig.get_fig()

    def save(self):
        fig = self.get_figure()
        fig.save()
