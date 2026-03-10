from compartmentals.CompartmentalModelSolver import CompartmentalModelSolver


class SEIRSD:
    DEFAULTS = {
        "S": 999985,
        "E": 10,
        "I": 5,
        "R": 0,
        "D": 0,
        "beta": 0.4332,
        "sigma": 0.192,
        "gamma": 0.141,
        "alfa": 0.0056,
        "mu": 0.0014,
        "r0": 3,
        "days": 365
    }
    COMPARTMENTS = ["Susceptíveis", "Expostos", "Infectados", "Recuperados", "Mortos"]

    def __init__(self):
        pass

    def get_default(self, key):
        return self.DEFAULTS[key]

    def odes(self, initialValues, time, transfer_rates):
        S, E, I, R, D = initialValues
        beta, sigma, gamma, alfa, mu = transfer_rates
        N = S + E + I + R + D

        dSdt = -beta * I * (S / N) + (alfa * R)
        dEdt = beta * I * (S / N) - (sigma * E)
        dIdt = sigma * E - (gamma * I) - (mu * I)
        dRdt = gamma * I - (alfa * R)
        dDdt = mu * I

        return [dSdt, dEdt, dIdt, dRdt, dDdt]
    
    def solve(self, days, initial_conditions, transfer_rates):
        solver = CompartmentalModelSolver(
            ode_function=self.odes,
            initial_conditions=initial_conditions,
            transfer_rates=transfer_rates,
            days=days,
            compartments=self.COMPARTMENTS,
            model_name="SEIRSD"
        )
        solver.solve()
        return solver.get_figure()