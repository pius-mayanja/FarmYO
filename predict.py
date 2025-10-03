from pathlib import Path
import torch
from utilis import create_vit_model
import random
from PIL import Image
import matplotlib.pyplot as plt
import shutil

torch.manual_seed(42)

vit_model, transforms = create_vit_model(num_classes=3, seed=42)
state_dict = Path('./Trained-models/ViT-lr-0.001-batch-16.pth')

vit_model.load_state_dict(state_dict=torch.load(f=state_dict))
class_names = ['healthy', 'maize leaf blight', 'maize streak virus']

image_paths = list(Path('./MLB-MSV-Healthy-Dataset/test/').glob("*/*.jpg"))

sample_image_paths = random.sample(image_paths, k=10)

true = 0
false = 0
predictions = 0

for image_path in image_paths:
    image = Image.open(image_path)
    img = transforms(image).unsqueeze(dim=0)
    img_class = image_path.parent.stem
    vit_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(vit_model(img), dim=1)
        pred_class = class_names[torch.argmax(pred_probs, dim=1)]
        pred_prob = round(pred_probs.max().item(), 2) * 100
        predictions += 1
        if pred_class == img_class:
            true +=1 
        else:
            false += 1

print(f'Out of {len(image_paths)} predictions:\n{true} are true\n{false} are false')