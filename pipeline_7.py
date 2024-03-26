from transformers import pipeline
import gradio as gr

if __name__ == "__main__":

    pipe = pipeline("image-classification", model="google/vit-base-patch16-224")

    gr.Interface.from_pipeline(pipe).launch()
