import subprocess
import gradio as gr
from batch import run

def func(mesh_folder, output_folder):
    subprocess.Popen([
        'python', 'batch.py', 
        '--mesh-folder', mesh_folder, 
        '--output-folder', output_folder,
    ])
    return f"Processing folder: {mesh_folder}"

demo = gr.Interface(
    fn=func,
    inputs=[
        gr.Textbox(value='data/stl'),
        gr.Textbox(value='same_as_mesh_folder'),],
    outputs=["text"],
)

demo.launch()