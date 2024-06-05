import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import os

# Set Plotly to render charts using the web browser
# pio.renderers.default = "browser"
STEPS = 10
# Function to read CSV files and prepare the data
def read_csv_files(filepaths, n = STEPS):
    data = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, names= ['x', 'y', 'xdot', 'ydot'], skiprows=lambda x: x % 4 )


        data.append(df[0:n]) # TODO - Remove the : to read the entire dataset
    return data

# Function to prepare data for animation with fading trails
def prepare_data_for_animation(data, dt=0.01, trail_length=50):
    frames = []
    max_length = max(len(df) for df in data)
    time_steps = np.arange(0, max_length * dt, dt)
    
    for t_index, t in enumerate(time_steps):
        frame_data = []
        for i, df in enumerate(data):
            x_vals = []
            y_vals = []
            opacities = []
            
            for trail_index in range(max(0, t_index - trail_length), t_index + 1):
                if trail_index < len(df):
                    x, y = df.loc[trail_index, ['x', 'y']]
                    x_vals.append(x)
                    y_vals.append(y)
                    # Calculate opacity based on how old the trail is
                    opacity = (trail_index - max(0, t_index - trail_length)) / trail_length
                    opacities.append(opacity)
            
            frame_data.append(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+lines',
                name=f'Trace {i+1}',
                marker=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], opacity=1.0),
                line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], width=2)
            ))
                
        frames.append(go.Frame(data=frame_data, name=f'time={t:.2f}'))
    return frames, time_steps

# Function to create the animated plot
def create_animation(frames, time_steps):
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(text='Traces Animation with Fading Trails'),
            xaxis=dict(autorange=True),
            yaxis=dict(autorange=True),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(label='Play',
                              method='animate',
                              args=[None, dict(frame=dict(duration=10, redraw=True), fromcurrent=True)])]
            )]
        ),
        frames=frames
    )
    return fig 


# List of CSV file paths
# filepaths = ['../datasets/pendulumR0.csv','../datasets/pendulumR0_2.csv']  # Update with your actual file paths



current_directory = os.path.dirname(__file__)
print(current_directory)
file_path = os.path.join(current_directory, 'resources', 'demo', '*.csv')

print("Paths using OS:", file_path)



globpath = glob.glob('data/*.csv')  # Update with your actual directory path
  # Update with your actual directory path
print(globpath)
print(file_path)



# Read the CSV files
data = read_csv_files(globpath)



# Prepare the data for animation
frames, time_steps = prepare_data_for_animation(data, trail_length=150)

fig = create_animation(frames, time_steps)

#%%
from dash import Dash, dcc, html
#run dash in my browser



app = Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])
if __name__ == '__main__':
    app.run(use_reloader=False, debug = True)  # Turn of
    server = app.server