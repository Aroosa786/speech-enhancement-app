import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import os
from dotenv import load_dotenv
from utils.audio_processing import (
    load_sample_audio, add_noise, create_spectrogram, audio_to_base64, 
    enhance_audio_with_aic, create_vad_output
)

load_dotenv()
app = dash.Dash(__name__)
server = app.server
original_audio, sample_rate = load_sample_audio()

app.layout = html.Div([
    html.H1("Speech Enhancement Dashboard", 
            style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.Label("Noise Level (dBFS):"),
            dcc.Slider(
                id='noise-slider',
                min=-80, max=0, value=-20, step=5,
                marks={i: f'{i}dB' for i in range(-80, 1, 20)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '45%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Enhancement Level:"),
            dcc.Slider(
                id='enhancement-slider',
                min=0.0, max=1.0, value=0.7, step=0.1,
                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '45%', 'float': 'right'})
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    html.Div([
        html.Div([
            html.H3("🎤 Original Clean Speech", style={'textAlign': 'center', 'color': '#28a745'}),
            dcc.Graph(id='original-spectrogram'),
            html.Div(id='original-audio-player')
        ], style={'width': '32%', 'display': 'inline-block', 'margin': '0.5%', 
                  'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px'}),
        
        html.Div([
            html.H3("🔊 Noisy Speech", style={'textAlign': 'center', 'color': '#dc3545'}),
            dcc.Graph(id='noisy-spectrogram'),
            html.Div(id='noisy-audio-player')
        ], style={'width': '32%', 'display': 'inline-block', 'margin': '0.5%',
                  'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px'}),
        
        html.Div([
            html.H3("✨ AI Enhanced Speech", style={'textAlign': 'center', 'color': '#007bff'}),
            dcc.Graph(id='enhanced-spectrogram'),
            html.Div(id='enhanced-audio-player')
        ], style={'width': '32%', 'display': 'inline-block', 'margin': '0.5%',
                  'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px'})
    ]),
    
    html.Div([
        html.H4("🎙️ Voice Activity Detection (VAD)", style={'textAlign': 'center'}),
        dcc.Graph(id='vad-plot')
    ], style={'margin': '20px 0.5%', 'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '10px'})
    
], style={'padding': '20px', 'backgroundColor': '#f0f0f0'})

@app.callback(
    [Output('original-spectrogram', 'figure'),
     Output('noisy-spectrogram', 'figure'),
     Output('enhanced-spectrogram', 'figure'),
     Output('original-audio-player', 'children'),
     Output('noisy-audio-player', 'children'),
     Output('enhanced-audio-player', 'children'),
     Output('vad-plot', 'figure')],
    [Input('noise-slider', 'value'),
     Input('enhancement-slider', 'value')]
)
def update_audio(noise_level, enhancement_level):
    noisy_audio = add_noise(original_audio, noise_level, sample_rate)
    enhanced_audio = enhance_audio_with_aic(noisy_audio, enhancement_level, sample_rate)
    
    def make_spectrogram(audio, color_scale):
        f, t, spec = create_spectrogram(audio, sample_rate)
        return {
            'data': [go.Heatmap(
                z=spec, x=t, y=f[:len(f)//4],  
                colorscale=color_scale,
                showscale=True
            )],
            'layout': {
                'xaxis': {'title': 'Time (s)'},
                'yaxis': {'title': 'Frequency (Hz)'},
                'height': 300,
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 50}
            }
        }
    
    orig_fig = make_spectrogram(original_audio, 'Greens')
    noisy_fig = make_spectrogram(noisy_audio, 'Reds') 
    enhanced_fig = make_spectrogram(enhanced_audio, 'Blues')
    
    time_vad, vad_signal = create_vad_output(enhanced_audio, sample_rate)
    vad_fig = {
        'data': [go.Scatter(
            x=time_vad, 
            y=vad_signal, 
            mode='lines+markers',
            name='Voice Activity',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)'
        )],
        'layout': {
            'title': 'Voice Activity Detection Output',
            'xaxis': {'title': 'Time (s)'},
            'yaxis': {'title': 'Voice Activity (0=No Speech, 1=Speech)'},
            'height': 250,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    }
    
    orig_player = html.Audio(
        src=audio_to_base64(original_audio, sample_rate),
        controls=True, style={'width': '100%', 'marginTop': '10px'}
    )
    noisy_player = html.Audio(
        src=audio_to_base64(noisy_audio, sample_rate),
        controls=True, style={'width': '100%', 'marginTop': '10px'}
    )
    enhanced_player = html.Audio(
        src=audio_to_base64(enhanced_audio, sample_rate),
        controls=True, style={'width': '100%', 'marginTop': '10px'}
    )
    
    return (orig_fig, noisy_fig, enhanced_fig,
            orig_player, noisy_player, enhanced_player, vad_fig)

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
