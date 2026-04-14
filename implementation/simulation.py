from cellular_automata.automata import Automata
from models.seirsd import SEIRSD

seirsd = SEIRSD()
automata = Automata(seirsd)

initial_conditions = {
    2: 20
}
automata.create_random_population(initial_conditions)

automata.run()