from cellular_automata.cell import Cell
from cellular_automata.condition import Condition
from cellular_automata.grid import Grid
from cellular_automata.statistic import Statistic
from models.seirsd import SEIRSD

size = 10
intial_exposeds = 5
initial_infecteds = 1
tick_count = 0
seirsd = SEIRSD()
seirsd.update_metrics(
    {
        'alfa': 0.3,
        'ticks': 100
    }
)


grid = Grid(size, seirsd)
grid.initialize_random_condition(intial_exposeds, Condition.EXPOSED)
grid.initialize_random_condition(initial_infecteds, Condition.INFECTED)
grid.loop_interation(sleep_between_tick=1)