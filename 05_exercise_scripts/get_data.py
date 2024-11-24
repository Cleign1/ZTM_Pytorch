
import os
import zipfile

from pathlib import Path

import requests

# setup the data directory
data_dir = Path("data")
image_path = data_dir / "pizza_steak_sushi"

if image_path.is_dir():
    print(f'Image path {image_path} is a directory')
else:
    print(f'Image path {image_path} is not a directory')
    print('creating the directory...')
    image_path.mkdir(parents=True, exist_ok=True)
    
with open(data_dir / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading data...")
    f.write(request.content)
    
with zipfile.ZipFile(data_dir / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping data...")
    zip_ref.extractall(data_dir)
    
os.remove(data_dir / "pizza_steak_sushi.zip")
