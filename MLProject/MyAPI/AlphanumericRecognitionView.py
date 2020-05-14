import base64
import json
from io import BytesIO

import numpy as np
from django.contrib import messages
from django.shortcuts import render
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from .AlphanumericRecognitionForm import AlphanumericRecognitionForm1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(28, 64, (5, 5), padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 2, padding=2)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 512)
        x = self.bn(x)
        x = x.view(-1, 512)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


# Define model - ref CNN2

class MyModel:

    def __init__(self, model_weights: str, device: str):
        '''

        '''
        self.net = Net()
        self.weights = model_weights
        self.device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
        self.preprocess = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])
        self._initialize()

    def _initialize(self):
        # Load weights
        try:
            # Force loading on CPU if there is no GPU
            if (torch.cuda.is_available() == False):
                self.net.load_state_dict(
                    torch.load(self.weights, map_location=lambda storage, loc: storage)["state_dict"])
            else:
                self.net.load_state_dict(torch.load(self.weights)["state_dict"])

        except IOError:
            print("Error Loading Weights")
            return None
        self.net.eval()

        # Move to specified device
        self.net.to(self.device)

    def predict(self, path):
        # Open the Image and resize
        img = Image.open(path).convert('L')

        # Convert to tensor on device
        with torch.no_grad():
            img_tensor = self.preprocess(img)  # tensor in [0,1]
            img_tensor = 1 - img_tensor
            img_tensor = img_tensor.view(1, 28, 28, 1).to(self.device)

            # Do Inference
            probabilities = self.net(img_tensor)
            probabilities = F.softmax(probabilities, dim=1)

        return probabilities[0].cpu().numpy()


model = MyModel('MyAPI/trained_weights.pth', 'cpu')
CLASS_MAPPING = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'


def AlphanumericRecognition(request):
    if request.method == 'POST':
        form = AlphanumericRecognitionForm1(request.POST)
        if form.is_valid():
            messages.success(request, 'Invalid: Your Request.')

    form = AlphanumericRecognitionForm1()

    return render(request, 'AlphanumericRecognition.html', {'form': form})


def predict(request):
    results = {"prediction": "Empty", "probability": {}}

    # get data
    input_img = BytesIO(base64.urlsafe_b64decode(request.form['img']))

    # model.predict method takes the raw data and output a vector of probabilities
    res = model.predict(input_img)

    results["prediction"] = str(CLASS_MAPPING[np.argmax(res)])
    results["probability"] = float(np.max(res)) * 100
    # results["prediction"] = 5
    # results["probability"] = 50.424

    # output data
    return json.dumps(results)
