import gradio as gr
from utilis import create_vit_model
import torch
from pathlib import Path

vit_model, transforms = create_vit_model(num_classes=3, seed=42)
class_names = ['healthy', 'maize leaf blight', 'maize streak virus']

vit_model, transforms = create_vit_model(num_classes=3, seed=42)
state_dict = Path('./Trained-models/ViT-lr-0.001-batch-16.pth')

vit_model.load_state_dict(state_dict=torch.load(f=state_dict))

def predict(image):
    image = transforms(image).unsqueeze(0)
    vit_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(vit_model(image), dim=1)
        pred_class = class_names[torch.argmax(pred_probs, dim=1)]
        pred_prob = round(pred_probs.max().item(), 2)
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

        return pred_labels_and_probs

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=gr.Label(num_top_classes=3, label='Prediction'))
demo.launch()

