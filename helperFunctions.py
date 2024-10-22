import cv2
import torch
import numpy as np
from Model.createNeuralNetwork import device, emnistTextModel
from Model.labels import letters_labels_map

# Loads the saved weights of the emnistTextModel from the last time it completed training
# and prepares the model for evaluation
model_save_path = "./Model/Model-Dictionary"
emnistTextModel.load_state_dict(torch.load(model_save_path, weights_only = True))
emnistTextModel.to(device)
emnistTextModel.eval()

# Takes in a input of a single letter image with a clear background color where the letter is 
# predominatly present and outputs the emnistTextModel's predicted letter classification
def imageToLetterConvert(letterImage):
    # Cropping the background and reducing image size to 28 x 28 pixels
    scaledLetterImage = cv2.resize(letterImage, (28, 28))

    # Configuring image to have a black background with the letter in white
    grayscaleLetterImage = cv2.cvtColor(scaledLetterImage, cv2.COLOR_BGR2GRAY)

    minThreshold = max(0, int(grayscaleLetterImage[0][0]) - 20)
    maxThreshold = min(255, int(grayscaleLetterImage[0][0]) + 20)

    transposedGrayscaleLetterImage = np.transpose(grayscaleLetterImage)

    whiteBackgroundLetterImage = cv2.inRange(transposedGrayscaleLetterImage, minThreshold, maxThreshold)
    blackBackgroundLetterImage = cv2.bitwise_not(whiteBackgroundLetterImage)

    # Configuring a tensor to represent the original letter image and for it be compatible 
    # with the model's expected inputs
    letterTensor = torch.from_numpy(blackBackgroundLetterImage).float()
    letterTensor = letterTensor.unsqueeze(0).unsqueeze(0)
    letterTensor = letterTensor.to(device)
    letterTensor = letterTensor / 255.0

    # Getting model's predicted label and returning the output letter from the original image
    logits = emnistTextModel(letterTensor)
    _, modelPredictionIndex = torch.max(logits, dim = 1)
    modelPrediction = letters_labels_map[modelPredictionIndex.item()]

    outputLetter = modelPrediction
    return outputLetter


# Crops the background of a single letter image to ensure the letter is predominant when the
# model processes the image
def cropSingleLetterBackground(letterImage):
    # Reducing the noise in the image
    grayscaleLetterImage = cv2.cvtColor(letterImage, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaleLetterImage, (5, 5), 0)

    # Applying binary thresholding to the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detecting any contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if contours:
        # Get the largest contour, assuming it's just the letter for single letter images
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Providing padding to ensure that the letter isn't too close to the border of the image
        padding = int(w * 0.75)
        x = max(0, x - padding)  
        y = max(0, y - padding)  
        w = min(letterImage.shape[1] - x, w + 2 * padding) 
        h = min(letterImage.shape[0] - y, h + 2 * padding)  

        # Cropping the image using the padded bounding box
        cropped_image = letterImage[y:y+h, x:x+w]

        return cropped_image
    
    else:
        # Returns original letter image if no contours were found
        return letterImage
