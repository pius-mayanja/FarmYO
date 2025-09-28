from torchvision import transforms
from torchvision import models
from pathlib import Path
from utilis import create_dataloaders, create_effnetb2, create_vit_model, train_test, demo_write
from torchinfo import summary
import torch
from tqdm import tqdm

data_dir = Path('./MLB-MSV-Healthy-Dataset')
supervisely = Path('./SuperviselyFormatDataSet(MLB-MSV-Healthy)/ds/img')
ann_path = Path('./SuperviselyFormatDataSet(MLB-MSV-Healthy)/ds/ann')

train_dir = data_dir/'train'
test_dir = data_dir/'test'

torch.manual_seed(42)
BATCH_SIZE = 16
EPOCHS = 10
learning_rate = 1e-4

# effnetb0_model, effnetb0_transforms = create_effnetb0(num_classes=3, seed=42)
# effnetb1_model, effnetb1_transforms = create_effnetb1(num_classes=3, seed=42)
effnetb2_model, effnetb2_transforms = create_effnetb2(num_classes=3, seed=42)
vit_model, vit_transforms = create_vit_model(num_classes=3, seed=42)

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transforms=vit_transforms,
                                                                    batch_size=BATCH_SIZE)

img, label = next(iter(train_dataloader))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer_effnet = torch.optim.Adam(effnetb2_model.parameters(), lr=learning_rate)
optimizer_vit = torch.optim.Adam(vit_model.parameters(), lr=learning_rate)

print(f'[INFO] Training on learning rate of {learning_rate} and Batch size of {BATCH_SIZE}')
vit_results = train_test(epochs=10,
                        model=vit_model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer_vit)

model_save_folder = Path('./Trained-models')
model_name = f'ViT-lr-{learning_rate}-batch-{BATCH_SIZE}.pth'
model_save_folder.mkdir(parents=True, exist_ok=True)

model_save_path = model_save_folder/model_name
assert model_name.endswith('.pth') or model_name.endswith('.pt'), '[INFO] Model save file must be a .pth or .pt file!'

torch.save(obj=vit_model.state_dict(), f=model_save_path)
print(f'[INFO] Saved {model_name} to {model_save_path}')
print("")

with open(model_save_folder/'results.txt', "a") as f:
    f.write(f'Results for learning rate of {learning_rate} and batch size of {BATCH_SIZE}:\n\t')
    f.write(str(vit_results))
    f.write(' ')

