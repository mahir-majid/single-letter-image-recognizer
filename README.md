Project Name: Single Letter Image Recognizer

Overview: The purpose of this project is to build a general-purpose algorithm that can successfully classify the letter present in any image containing a single letter and a clear background color. 
It was also intended that the algorithm worked on letters of any color and of any size in proportion to the input image size. 

To do this, a Convolutional Neural Network (CNN) letter recognition model called the emnistLetterModel was trained through PyTorch on the letters subset of the TorchVision dataset EMNIST. This dataset contains 
a training and test dataset, both with 26 balanced classes where every class corresponds to the both the uppercase and lowercase version of the apropriate letter. An optimization loop was created to train the 
model on the training dataset and evaluate the model's accuracy on the test dataset for each iteration, and in the end the emnistLetterModel yieled a 93% accuracy in classifying the images in the test dataset.

After developing the letter recognition model, a script in OpenCV was developed to take in an input image and enable the emnistLetterModel to classify the letter in the image.

Programming Languages: Python

Libraries: PyTorch, OpenCV
