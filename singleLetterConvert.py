import cv2
import numpy as np
from helperFunctions import imageToLetterConvert, cropSingleLetterBackground

# Evaluating model's ability to correctly classify images of single letters of any color
# and convert an image of a single letter to text

# Processing a random image of a single letter from folder "Single Letter Images", but feel
# free to upload any image of a single letter that has one clear background color
img_files = ["LowercaseA", "LowercaseE", "LowercaseH", "LowercaseM", "LowercaseN", "LowercaseR", 
             "UppercaseB", "UppercaseF", "UppercaseP", "UppercaseS", "UppercaseT", "UppercaseL"]
rand_file_idx = np.random.randint(0, len(img_files))
img_file = img_files[rand_file_idx]
single_letter_image_path = f'./Single Letter Images/{img_file}.png'
originalLetterImage = cv2.imread(single_letter_image_path)

# Cropping background to ensure that the single letter is predominant in the image
# when the model processes the image
croppedLetterImage = cropSingleLetterBackground(originalLetterImage)

# Capturing the model's predicted letter classification of the provided image
outputLetter = imageToLetterConvert(croppedLetterImage)
print(f"Converted Letter from Image: {outputLetter}")

cv2.imshow(outputLetter, originalLetterImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
