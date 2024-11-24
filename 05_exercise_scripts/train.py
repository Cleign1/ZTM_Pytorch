# YOUR CODE HERE
import os
import torch
from torchvision.transforms import v2 as transforms
import argparse
import data_setup, model_builder, engine, utils

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# setup parser
parser = argparse.ArgumentParser(description="butuh hyperparameter")

# setup argparser untuk num_epochs
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train model, default=10")

# setup argparser untuk batch_size
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training model, default=32")

# setup argparser untuk hidden layer
parser.add_argument("--hidden_units", type=int, default=64, help="hidden layer size for model, default=128")

# setup argparser untuk learning rate
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for model, default=0.001")

# setup argparser untuk training directory
parser.add_argument("--train_dir", type=str, default="data/train", help="training directory for model, default=data/pizza_steak_sushi/train")

# setup argparser untuk testing directory
parser.add_argument("--test_dir", type=str, default="data/test", help="testing directory for model, default=data/pizza_steak_sushi/test")

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f'[INFO] Training model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}, hidden layer size {HIDDEN_UNITS}, and learning rate {LEARNING_RATE}')

# setup directories
train_dir = args.train_dir
test_dir = args.test_dir

print(f'[INFO] Training directory: {train_dir}')
print(f'[INFO] Testing directory: {test_dir}')

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True),
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, 
                                                                               test_dir, 
                                                                               data_transform, 
                                                                               BATCH_SIZE)

model = model_builder.TinyVGG(input_shape=3 ,hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             device=device,
             epochs=NUM_EPOCHS)

utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_tiny_vgg_script_mode.pth",)
