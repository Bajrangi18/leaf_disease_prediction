import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn
from torchvision import datasets, transforms

# Define transformations for the dataset
transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)

# Load disease information
data = pd.read_csv("disease_info.csv", encoding="cp1252")

# Define CNN model
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.dense_layers(out)
        return out

# Load model
targets_size = 39
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(targets_size)
model.load_state_dict(torch.load("plant_disease_model_1.pt"))
model.to(device)
model.eval()

# Define single_prediction function
def single_prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224)).to(device)
    output = model(input_data)
    output = output.detach().cpu().numpy()  # Move to CPU and convert to numpy
    index = np.argmax(output)
    print("Original : ", image_path.split('/')[-1][:-4])  # Adjust as needed
    pred_csv = data["disease_name"][index]
    print(pred_csv)

# Test single prediction
single_prediction("test_images/Apple_ceder_apple_rust.JPG")
