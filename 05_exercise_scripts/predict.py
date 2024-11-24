# YOUR CODE HERE
import torch
import torchvision
import argparse

import model_builder

parser = argparse.ArgumentParser(description="butuh hyperparameter")

# setup argparser untuk image_path
parser.add_argument("--image_path", help="path to image file for prediction")

parser.add_argument("--model_path",
                    default="models/05_tiny_vgg_script_mode.pth",
                    type=str,
                    help="path to model for prediction")

args = parser.parse_args()

class_names = ["pizza", "steak", "sushi"]

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_PATH = args.image_path
print(f'[INFO] Image path: {IMG_PATH}')

def load_model(filepath=args.model_path):
    model = model_builder.TinyVGG(input_shape=3, 
                                  hidden_units=128, 
                                  output_shape=3).to(device)
    model.load_state_dict(torch.load(filepath))
    
    return model

def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
    model = load_model(filepath)
    
    image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)
    
    image = image / 255.0
    
    transform = torchvision.transforms.Resize((64, 64))
    image = transform(image)
    
    model.eval()
    with torch.inference_mode():
        image = image.to(device)
        pred_logits = model(image.unsqueeze(dim=0))
        pred_probs = torch.softmax(pred_logits, dim=1)
        
        pred_label = torch.argmax(pred_logits, dim=1)
        pred_label_class = class_names[pred_label]
        
    print(f'[INFO] Pred Class: {pred_label_class}, pred prob: {pred_probs.max():.3f}')
    
if __name__ == "__main__":
    predict_on_image()
