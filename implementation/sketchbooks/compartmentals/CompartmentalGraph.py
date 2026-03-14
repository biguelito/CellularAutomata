import plotly.graph_objs as go
from time import time

class CompartmentalGraph:
    def __init__(self, tspan, curves_dict, model_name):
        self.tspan = tspan
        self.curves = curves_dict
        self.model_name = model_name

    def get_fig(self):
        fig = go.Figure()

        for name, values in self.curves.items():
            fig.add_trace(go.Scatter(
                x=self.tspan,
                y=values,
                mode="lines",
                name=name
            ))

        fig.update_layout(
            title=f"Simulação de Modelo Compartimental - {self.model_name}",
            xaxis_title="Dias",
            yaxis_title="População",
            legend_title="Compartimentos",
        )

        return fig

    def save(self):
        fig = self.get_fig()
        fig.write_image(f"{self.model_name}_{time()}.png")
