from compartmentals.CompartmentalModelSolver import CompartmentalModelSolver
from scipy.stats import truncnorm
import numpy as np

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

    def sample_truncnorm(self, mean, sd, lower, upper, size):
        a, b = (lower - mean) / sd, (upper - mean) / sd
        return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=42)

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
        return solver

    def run_simulation(self, days, initial_conditions, transfer_rates):
        solver = self.solve(days, initial_conditions, transfer_rates)
        return solver.get_figure()
    
    def run_alfa_metric_basic_scenario(self, alfa_values, beta, sigma, gamma, mu, days, initial_conditions, timespan_days):
        results = {}

        for alfa in alfa_values:
            transfer_rates = (beta, sigma, gamma, alfa, mu)
            solver = self.solve(days, initial_conditions, transfer_rates)
            S, E, I, R, D = solver.solved_odes.T
            results[alfa] = {
                "t": timespan_days,
                "S": S,
                "E": E,
                "I": I,
                "R": R,
                "D": D,
                "total_deaths": D[-1]
            }

        return results
    
    def run_alfa_metric_monte_carlo(self, alfa_values, beta, sigma, gamma, mu, days, initial_conditions, N_sim, cv):
        results = {}

        for alfa in alfa_values:
            D_curves = []
            D_final = []
            
            sigma_samples = self.sample_truncnorm(sigma, sigma*cv, sigma*0.5, sigma*1.5, N_sim)
            gamma_samples = self.sample_truncnorm(gamma, gamma*cv, gamma*0.5, gamma*1.5, N_sim)
            mu_samples = self.sample_truncnorm(mu, mu*cv, mu*0.5, mu*1.5, N_sim)
            beta_samples = self.sample_truncnorm(beta, beta*cv, beta*0.5, beta*1.5, N_sim)
            
            for i in range(N_sim):
                sigma = sigma_samples[i]
                gamma = gamma_samples[i]
                mu = mu_samples[i]
                beta = beta_samples[i]
                transfer_rates = (beta, sigma, gamma, alfa, mu)

                solver = self.solve(days, initial_conditions, transfer_rates)
                D_t = solver.solved_odes[:, 4]
                
                D_curves.append(D_t)
                D_final.append(D_t[-1])
            
            D_curves = np.array(D_curves)
            D_final = np.array(D_final)
            
            mean_curve = np.mean(D_curves, axis=0)
            low_curve = np.percentile(D_curves, 2.5, axis=0)
            high_curve = np.percentile(D_curves, 97.5, axis=0)
            
            results[alfa] = {
                "mean": mean_curve,
                "low": low_curve,
                "high": high_curve,
                "final_mean": np.mean(D_final),
                "final_low": np.percentile(D_final, 2.5),
                "final_high": np.percentile(D_final, 97.5)
            }

        return results