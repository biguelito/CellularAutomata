import plotly.graph_objects as go
import time

class Grid:
    def __init__(self, matrix, quant_condition):
        self.size = len(matrix)
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
            z=matrix.tolist(),
            colorscale=self.colorscale,
            zmin=0,
            zmax=4,
            showscale=False,
        )])

        fig.update_layout(
            title=dict(text=self.make_title(), x=0.5, xanchor='center'),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
            margin=dict(l=20, r=20, t=100, b=80),
        )

        self.fig = fig
        return
    
    def make_title(self):
        return f"Simulação SEIRSD - {self.size}*{self.size} indivíduos<br>"

    def show(self, filename=None):
        if self.fig == None:
            return

        if filename == None:
            filename = time.time()

        self.fig.write_html(f"figures/{filename}.html")
        return