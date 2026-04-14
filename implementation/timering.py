from cellular_automata.automata import Automata
from models.seirsd import SEIRSD
import timeit
from statistics import mean

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
# for i in range(20):
#     automata.run_save()

generations_time = timeit.repeat(lambda: automata.run(), number=20, repeat=10)
print(f"minimo {min(generations_time)}")
print(f"maximo {max(generations_time)}")
print(f"media {mean(generations_time)}")
print(f"total {sum(generations_time)}")