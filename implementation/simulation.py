from cellular_automata.condition import Condition
from cellular_automata.automata import Automata
from models.seirsd import SEIRSD

seirsd = SEIRSD()
seirsd.update_metrics(
    {
        'alfa': 0.1,
        'ticks': 365
    }
)

size = 100
initial_conditions = {
    Condition.EXPOSED: 10,
    Condition.INFECTED: 5
}

automata = Automata(size, seirsd, initial_conditions)
automata.run()