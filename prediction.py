from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("dogs_cats_model.keras")

# You can use loaded_model for various tasks, such as making predictions

# Example: Make predictions on a new image
import numpy as np
from tensorflow.keras.preprocessing import image

# Load a new image for prediction (replace 'new_image.jpg' with your image file)
# img_path = 'dogs-vs-cats/full/test1/cats/80.jpg'
img_path = '/Users/gigalabs/Downloads/lion-794962_640.jpeg'

# Preprocess the image (resize, rescale, etc., to match the input size of the loaded model)
img = image.load_img(img_path, target_size=(224, 224))  # Adjust the target_size as needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to the range [0, 1]

# Use the loaded model to make predictions
predictions = loaded_model.predict(img_array)

# Interpret the predictions (depends on your model and problem)
# For example, if it's a binary classification model, you can check if predictions > 0.5
if predictions[0][0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")
