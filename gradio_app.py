import gradio as gr
from utilis import create_vit_model

vit_model, transforms = create_vit_model(num_classes=3, seed=42)

def predict(image):
    image = transforms(image).unsqueeze(0)    

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=gr.Label(num_top_classes=3, label='Predictions'))