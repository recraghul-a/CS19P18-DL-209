import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Define image and batch parameters
IMG_SIZE = 160 # MobileNetV2 expects 160x160 images
BATCH_SIZE = 32

# 2. Set up the data directories (replace with your paths)
base_dir = 'path/to/your/dogs_vs_cats/dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# 3. Use ImageDataGenerator to load and augment data
# We'll use a data generator to handle the images in batches
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# 4. Load the pre-trained MobileNetV2 model
# We set include_top=False to exclude the final classification layers,
# as we will add our own.
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')

# 5. Freeze the convolutional base
# This prevents the pre-trained weights from being updated during training.
base_model.trainable = False

# 6. Build a new classifier on top of the base model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Use sigmoid for binary classification
])

# 7. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display the model's structure
model.summary()

# 8. Train the model with the new classifier
epochs = 10
print("\nTraining the model with transfer learning...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)

# 9. Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")