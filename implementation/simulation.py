import json
from pathlib import Path
import numpy as np
np.random.seed(42)

from cellular_automata.automata import Automata
from models.seirsd import SEIRSD


class Simulation:
    def __init__(self, size=100, initial_conditions=None):
        self.model = SEIRSD(size)
        self.initial_conditions = initial_conditions or {2: 20}
        self.figures_dir = Path(__file__).resolve().parent / "figures"

    def run_once(self, ticks=0, sleep_between_tick=1, filename=None):
        automata = Automata(self.model)
        automata.create_random_population(self.initial_conditions)
        automata.run(ticks=ticks, sleep_between_tick=sleep_between_tick, filename=filename)

    def run(self, runs=1, ticks=0, sleep_between_tick=1):
        for _ in range(runs):
            self.run_once(ticks=ticks, sleep_between_tick=sleep_between_tick)

    def get_metrics_files(self):
        if not self.figures_dir.exists():
            return []
        return sorted(self.figures_dir.glob("*.json"))

    def summarize_conditions_max(self, output_filename="conditions_max_statistics.json"):
        metrics_files = self.get_metrics_files()
        if not metrics_files:
            return {
                "total_files": 0,
                # "files": [],
                "conditions": {},
            }

        conditions_max_per_file = []
        files = []

        for metrics_file in metrics_files:
            with open(metrics_file, "r", encoding="utf-8") as current_file:
                metrics = json.load(current_file)

            conditions_max = metrics.get("conditions_max")
            if not conditions_max:
                continue

            conditions_max_per_file.append(conditions_max)
            files.append(metrics_file.name)

        if not conditions_max_per_file:
            return {
                "total_files": 0,
                # "files": [],
                "conditions": {},
            }

        conditions_array = np.array(conditions_max_per_file, dtype=float)
        summary = {
            "total_files": len(files),
            # "files": files,
            "conditions": {},
        }

        for index in range(conditions_array.shape[1]):
            values = conditions_array[:, index]
            summary["conditions"][index] = {
                # "values": values.tolist(),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": round(float(np.std(values)), 4),
                "cv": round((float(np.std(values))/float(np.mean(values))) * 100, 4)
            }

        output_path = self.figures_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=4)

        return summary

def run_simulation(runs):
    simulation = Simulation()
    simulation.run(runs=runs)
    if (runs > 1):
        simulation.summarize_conditions_max()

if __name__ == "__main__":    
    run_simulation(1)