import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("# Spralingia V2.0")
    gr.Audio(streaming=True)


demo.launch()