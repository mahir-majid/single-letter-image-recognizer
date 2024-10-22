import torch
from torch import nn

# Defining a Convolutional Neural Network to train model on EMNIST datsets
class NeuralNetworkCNN(nn.Module):
    def __init__(self):
        super(NeuralNetworkCNN, self).__init__()

        # Defining convolutional porition of Neural Network composed of 2 convolutional layers and 2 pooling layers 
        self.conv_layers = nn.Sequential (
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Defining the Adaptive Pooling Layer so that model can be designed to work with input image of any size
        # but adaptive pooling is based on dimensions of train / test dataset images that model used for training
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) 

        # Defining fully connected layers of the Neural Network
        self.fc_layers = nn.Sequential (
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 27)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        logits = self.fc_layers(x)  # Pass through fully connected layers
        return logits
    
    
# Instantiating Model to analyze EMNIST Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emnistTextModel = NeuralNetworkCNN().to(device)
print(f"Using {device} device")