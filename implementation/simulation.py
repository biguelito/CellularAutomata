from cellular_automata.automata import Automata
from models.seirsd import SEIRSD

seirsd = SEIRSD()
seirsd.update_metrics(
    {
        'alfa': 0.06,
        'ticks': 365
    }
)

size = 100
initial_conditions = {
    2: 20
}

automata = Automata(size, seirsd, initial_conditions, 5)
automata.run()