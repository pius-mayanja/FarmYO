from utilis import supervisely_to_standardFormat
from pathlib import Path

img_dir = Path('./makerere-university-maize/ds/img')
ann_dir = Path('./makerere-university-maize/ds/ann')

output_dir = Path('./MLB-MSV-Healthy-Dataset-Big')
output_dir.mkdir(parents=True, exist_ok=True)

train_dir, test_dir = supervisely_to_standardFormat(folder=img_dir, ann_pth=ann_dir, destination_folder=output_dir, train_ratio=0.8)

print(train_dir, test_dir)