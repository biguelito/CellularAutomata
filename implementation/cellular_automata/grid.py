import json
import plotly.graph_objects as go
import time
from pathlib import Path

class Grid:
    def __init__(self, quant_condition, states, sleep_between_tick=1):
        self.matrices_per_tick = states.matrices_per_tick
        self.statistics_per_tick = states.statistics_per_tick
        self.statistics_max = states.statistics_max
        self.conditions_max = states.conditions_max
        self.size = len(self.matrices_per_tick[0])
        self.sleep_between_tick = sleep_between_tick
        self.fig = None
        self.quant_condition = quant_condition

        self.colorscale = [
            [0.00, '#2ecc71'], [0.20, '#2ecc71'],
            [0.20, '#f1c40f'], [0.40, '#f1c40f'],
            [0.40, '#e74c3c'], [0.60, '#e74c3c'],
            [0.60, '#ffffff'], [0.80, '#ffffff'],
            [0.80, '#111111'], [1.00, '#111111'],
        ]
    
        fig = go.Figure(data=[go.Heatmap(
            z=self.matrices_per_tick[0].tolist(),
            colorscale=self.colorscale,
            zmin=0,
            zmax=4,
            showscale=False,
        )])

        fig.frames = self.create_frames()

        fig.update_layout(
            title=dict(text=self.make_title(0), x=0.5, xanchor='center'),
            updatemenus=self.create_menu(),
            sliders=self.create_slider(),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
            margin=dict(l=20, r=20, t=100, b=80),
        )

        self.fig = fig
        return
    
    def make_title(self, tick_idx):
        stats_tick = self.statistics_per_tick[tick_idx]
        stats_max = self.statistics_max[tick_idx]
        stats_line = ""
        for i in range(self.quant_condition):
            stats_line += f"{i}: [{stats_tick[i]}/{stats_max[i]}]  "
        return f"Simulação SEIRSD - {self.size}*{self.size} indivíduos<br><sub>{stats_line}</sub>"

    def create_menu(self):
        return [
            dict(
                type='buttons',
                showactive=False,
                y=1.05, x=0.0,
                xanchor='left', yanchor='top',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=self.sleep_between_tick*100, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        )],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0),
                        )],
                    ),
                ],
            )
        ]
    
    def create_slider(self):
        return [
            dict(
                active=0,
                steps=[dict(
                    method='animate',
                    args=[[str(idx)], dict(
                        mode='immediate',
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    )],
                    label=str(idx),
                ) for idx in range(len(self.matrices_per_tick))],
                currentvalue=dict(prefix='Tick: ', visible=True, xanchor='center'),
                transition=dict(duration=0),
            )
        ]
    
    def create_frames(self):
        frames = []
        for idx in range(len(self.matrices_per_tick)):
            frames.append(go.Frame(
                data=[go.Heatmap(
                    z=self.matrices_per_tick[idx].tolist(),
                    colorscale=self.colorscale,
                    zmin=0,
                    zmax=4,
                    showscale=False,
                )],
                layout=go.Layout(title=dict(text=self.make_title(idx))),
                name=str(idx),
            ))

        return frames

    def make_metrics_payload(self):
        return {
            "size": self.size,
            "quant_condition": self.quant_condition,
            "sleep_between_tick": self.sleep_between_tick,
            "ticks_recorded": len(self.statistics_per_tick),
            "conditions_max": self.conditions_max,
            "statistics_per_tick": self.statistics_per_tick,
            # "matrices_per_tick": [matrix.tolist() for matrix in self.matrices_per_tick],
        }

    def save(self, filename=None):
        if self.fig == None:
            return

        if filename == None:
            filename = time.time()

        output_dir = Path(__file__).resolve().parent.parent / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.fig.write_html(output_dir / f"{filename}.html")
        with open(output_dir / f"{filename}.json", "w", encoding="utf-8") as metrics_file:
            json.dump(self.make_metrics_payload(), metrics_file, indent=4)
        return
