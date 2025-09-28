from pathlib import Path
import json
import shutil
import random
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict

def supervisely_to_standardFormat(folder, ann_pth, destination_folder, train_ratio):
    """
    Args:
        ann_pth (str or Path): Path to the annotations folder from the supervisely folder.
        folder: (str or Path): Path to the image folder from the supervisely folder.
        destination folder: (str or Path): Path to the folder that will store data in data format.
    """
    anns_dir = Path(ann_pth)
    img_dir = Path(folder)
    data_dir = Path(destination_folder)
    demo_path = Path('./DemoFolder')

    demo_path.mkdir(parents=True, exist_ok=True)

    if not anns_dir.exists() or not anns_dir.is_dir(): # ---> check if provided path exists and it's a folder
        print(f"[ERROR] No annotations folder found at: {ann_pth}. Please check the path.")
        return
    
    if not img_dir.exists() or not img_dir.is_dir(): # ---> check if provided path exists and it's a folder
        print(f"[ERROR] No image folder found at: {folder}. Please check the path.")
        return
    
    if not data_dir.exists() or not data_dir.is_dir(): # ---> check if provided path exists and it's a folder
        print(f"[ERROR] No folder found at: {destination_folder}. Please check the path.")
        return

    ann_files = [f for f in anns_dir.iterdir() if f.is_file()] # ---> list of annotation file paths

    for ann_file in ann_files:
        if not ann_file.suffix == ".json":
            print(f"[INFO] {ann_file} is not a .json file, it's being skipped")
            continue

        image_path = img_dir/ann_file.name.replace(".json", "")

        if not image_path.exists():
            print(f"[WARNING] No matching image found for {ann_file.name} -> {image_path.name}")
        
        try:
            with open(ann_file, "r") as f:
                annotations = json.load(f)

            class_name = None
            if annotations.get('objects') and len(annotations['objects']) > 0:
                class_name = annotations['objects'][0]['classTitle']
            elif annotations.get('tags') and len(annotations['tags']) > 0:
                class_name = annotations['tags'][0]['name']
            
            if not class_name:
                print(f"[WARNING] No class name found in {ann_file.name} for image: {image_path.name}")
                continue
            
            class_folder = demo_path/str(class_name)
            class_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=image_path, dst=class_folder)

        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON in {ann_file.name}")
        except Exception as e:
            print(f'[INFO] Error occured: {e}')

    print(f'[INFO] Moved images from {folder} to {demo_path}')
    
    for class_dir in demo_path.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.*"))
        random.shuffle(images)

        total = len(images)
        train_split = int(train_ratio * total)
        test_split = int((1-train_ratio) * total)

        split_images = {
            "train": images[:train_split],
            "test": images[train_split:]
        }
        
        for split, split_image in split_images.items():
            split_folder = data_dir/split/class_dir.name
            split_folder.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_image:
                shutil.copy2(src=img_path, dst=split_folder/img_path.name)
    
    shutil.rmtree(demo_path)

    train_path = data_dir/'train'
    test_path = data_dir/'test'

    return train_path, test_path
        

def create_dataloaders(train_dir, test_dir, transforms, batch_size):
    train_data = ImageFolder(root=train_dir, transform=transforms, target_transform=None)
    test_data = ImageFolder(root=test_dir, transform=transforms)

    train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    class_names = train_data.classes

    return train_dataLoader, test_dataLoader, class_names

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true)) * 100  
    return acc

def create_effnetb0(num_classes: int=3, seed: int=42):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed=seed)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), 
                                     nn.Linear(in_features=1408, out_features=num_classes))

    return model, transforms

def create_effnetb1(num_classes: int=3, seed: int=42):
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b1(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed=seed)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), 
                                     nn.Linear(in_features=1408, out_features=num_classes))

    return model, transforms

def create_effnetb2(num_classes: int=3, seed: int=42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed=seed)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), 
                                     nn.Linear(in_features=1408, out_features=num_classes))

    return model, transforms

def create_vit_model(num_classes: int, seed: int):
    vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    vit_transforms = vit_weights.transforms()
    vit_model = torchvision.models.vit_b_16(weights=vit_weights)

    for param in vit_model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(seed)
    vit_model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))

    return vit_model, vit_transforms

def train_test(epochs, model, optimizer, train_dataloader, test_dataloader, loss_fn):
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_acc = 0, 0
        for X, y in train_dataloader:
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y_pred=y_pred, y_true=y)
            train_loss += loss.item()
            train_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        
        model.eval()
        test_acc, test_loss = 0, 0
        for X, y in test_dataloader:
            with torch.inference_mode():
                test_logits = model(X)
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                loss_ = loss_fn(test_logits, y)
                acc_ = accuracy_fn(y_pred=test_pred, y_true=y)
                test_loss += loss_.item()
                test_acc += acc_
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)

    print(f'Training process ended.\nTrain accuracy: {train_acc:2f}% | Test accuracy: {test_acc:2f}%')
    return {
        'Train Loss': f'{train_loss:4f}',
        'Train Accuracy': f'{train_acc:2f}%',
        'Test Loss': f'{test_loss:4f}',
        'Test Accuracy': f'{test_acc:2f}%',
    }


def demo_write():
    return {"names": 'I want to add things here'}