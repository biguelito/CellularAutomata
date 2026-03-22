from cellular_automata.automata import Automata
from models.seirsd import SEIRSD

seirsd = SEIRSD()
seirsd.update_metrics(
    {
        'alfa': 0.06,
        'ticks': 120
    }
)

size = 10
initial_conditions = {
    1: 5
}

automata = Automata(size, seirsd, initial_conditions, 5)
automata.run()