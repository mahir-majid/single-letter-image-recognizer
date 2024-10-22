import torch
from torch import nn
from torch.optim import Adam
from createNeuralNetwork import device, emnistTextModel
from dataloaders import train_dataloader, test_dataloader

# Defining the Train Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    # Iterates through each batch in the dataloader allowing the model to predict a label for 
    # every image, and each iteration updates the model's parameters including all of the model's  
    # weights, biases, and kernel coefficients, based on gradient calculations with the loss function
    for batch, (batchImages, batchLabels) in enumerate(dataloader):
        batchImages, batchLabels = batchImages.to(device), batchLabels.to(device)
        predLogits = model(batchImages)
        loss = loss_fn(predLogits, batchLabels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            print(f"# of Batches processed: {batch}")
        
# Defining the Test Loop
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    totalLoss, totalCorrect = 0, 0

    # Iterates through each image in the dataloader and calculates average loss  
    # and accuracy of model's evaluation of the images
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            totalLoss += loss_fn(pred, label).item()
            totalCorrect += (pred.argmax(1) == label).type(torch.float).sum().item()

    averageLoss = round(totalLoss / size, 3)
    accuracy = round(totalCorrect / size, 2) * 100

    print(f"Average Loss: {averageLoss}")
    print(f"Accuracy: {accuracy} %")

# Configuring hyperparameters for optimization loop
epochs = 10
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(emnistTextModel.parameters(), lr = learning_rate)

# Optimization Loop
for i in range(epochs):
    print(f"Epoch: {i}")
    train_loop(train_dataloader, emnistTextModel, loss_fn, optimizer)
    test_loop(test_dataloader, emnistTextModel, loss_fn)

print("Finished Optimization Loop")

# Saving the current parameters of the emnistTextModel after completing training
model_save_path = "./Model/Model-Dictionary"
torch.save(emnistTextModel.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")