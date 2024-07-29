# Códigos de visualização gráfica
import plotly.graph_objects as go


def scatter(nomes, titulo, *args):
    if len(args) == 2:
        x, y = args
        fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers', text=nomes)])
    else:
        x, y, z = args
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', text=nomes)])
    fig.update_layout(showlegend=False, title=titulo)
    fig.show()
    return fig
