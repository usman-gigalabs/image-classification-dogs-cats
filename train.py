from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the path to your dataset
train_data_dir = 'dogs-vs-cats/small/train'
test_data_dir = 'dogs-vs-cats/small/test1'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Create data generators for training and testing
train_data_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_data_gen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Load the saved model
loaded_model = load_model("dogs_cats_model.keras")

# Compile the model
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = loaded_model.fit(
    train_generator,  # Replace with your new training data generator or data
    epochs=10,            # Adjust the number of epochs as needed
    validation_data=test_generator  # Replace with your new validation data generator or data
)

# Save the trained model
model.save("dogs_cats_model.keras")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()