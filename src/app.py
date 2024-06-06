import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import os
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors


# Set Plotly to render charts using the web browser
# pio.renderers.default = "browser"
STEPS = 1000

# Function to read CSV files and prepare the data
def read_csv_files(filepaths, n=STEPS):
    data = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, names=['x', 'y', 'xdot', 'ydot'], skiprows=lambda x: x % 5)
        data.append(df[0:n])
        #sort the data by the starting x value
        data = sorted(data, key=lambda x: x['x'].iloc[0])
    return data

# Function to prepare data for animation with fading trails
def prepare_data_for_animation(data, dt=0.01, trail_length=50):
    frames = []
    max_length = max(len(df) for df in data)
    time_steps = np.arange(0, max_length * dt, dt)
    
    # Generate a rainbow colormap
    cmap = plt.get_cmap('jet', len(data))
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    for t_index, t in enumerate(time_steps):
        frame_data = []
        for i, df in enumerate(data):
            start_index = max(0, t_index - trail_length)
            end_index = t_index + 1

            if end_index <= len(df):
                x_vals = df['x'].iloc[start_index:end_index].tolist()
                y_vals = df['y'].iloc[start_index:end_index].tolist()
                
                frame_data.append(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name=f'Trace {i+1}',
                    marker=dict(color=colors[i], opacity=1.0),
                    line=dict(color=colors[i], width=2)
                ))

        frames.append(go.Frame(
            data=frame_data,
            name=f'time={t:.2f}',
            layout=go.Layout(
                # xaxis=dict(autorange=False),
                # yaxis=dict(autorange=False)
            )
        ))
    return frames, time_steps

def create_animation(frames, time_steps, charges):
    colors = ['red', 'green', 'blue']
    charge_scatter = [
        go.Scatter(
            x=[charge[0]],
            y=[charge[1]],
            mode='markers',
            marker=dict(color=color, size=12),
            name=f'Charge {i+1}'
        )
        for i, (charge, color) in enumerate(zip(charges, colors))
    ]

    fig = go.Figure(
        data=list(frames[0].data) + charge_scatter,
        layout=go.Layout(
            title=dict(text='Pendulum Trace near Charges'),
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            updatemenus=[dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')]),
                    dict(label='Reset',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0)), [0]])  # Reset to initial frame
                ]
            )]
        ),
        frames=frames
    )
    return fig

# List of CSV file paths
current_directory = os.path.dirname(__file__)
globpath = glob.glob('data/*.csv')
datapath = os.path.join(current_directory, 'data', '*.csv')
datapathglob = glob.glob(datapath)
print("Example file path:", datapathglob[0])

# Read the CSV files
data = read_csv_files(datapathglob)

# Prepare the data for animation
frames, time_steps = prepare_data_for_animation(data, trail_length=50)  # Reduced trail length

# Define the charges
d = 0.01  # Example value for d
CHARGES = [
    np.array([1, 1 * np.sqrt(3), -d]),
    np.array([2, 0, -0.01]),
    np.array([-1, 1 * np.sqrt(3), -d])
]

fig = create_animation(frames, time_steps, CHARGES)

from dash import Dash, dcc, html

app = Dash(__name__)
server = app.server
app.layout = html.Div([
    dcc.Graph(figure=fig, style={'height': '100vh', 'width': '100vw'})
], style={'height': '100vh', 'width': '100vw'})

if __name__ == '__main__':
    app.run(use_reloader=False, debug=False)
