from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Gathering training and test data from EMNIST dataset on various uppercase and lowercase letters
# where each image is grayscale has dimension 28 x 28 pixels
# Note: There are 27 different classes where the 0th class represents "Not a Letter" and each of
# the remaining 26 letter classes corresponds to both a letter's lowercase and uppercase version
train_data = datasets.EMNIST (
    root = './Model/EMNIST-Letters-Train',
    train = True,
    split = 'letters',
    download = True,
    transform = ToTensor()
)

test_data = datasets.EMNIST (
    root = './Model/EMNIST-Letters-Test',
    train = False,
    split = 'letters',
    download = True,
    transform = ToTensor()
)

# Initializing dataloaders on the train and test data that will be used to 
# train model to recognize all of the 26 letters
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
