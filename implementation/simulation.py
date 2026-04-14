from cellular_automata.automata import Automata
from models.seirsd import SEIRSD

seirsd = SEIRSD()
seirsd.update_metrics(
    {
        'ticks': 365
    }
)

initial_conditions = {
    2: 20
}

automata = Automata(seirsd, initial_conditions)
automata.run_save()