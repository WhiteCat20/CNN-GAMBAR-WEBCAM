import numpy as np
import os
import shutil
import tensorflow as tf

# Define the image data generator with augmentations
augmentation_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Directory to save the augmented images
augmented_dir = 'augmented_data'
if os.path.exists(augmented_dir):
    shutil.rmtree(augmented_dir)
os.makedirs(augmented_dir)

# Load the training dataset
train_dataset = augmentation_gen.flow_from_directory(
    'dataset 2/train',
    target_size=(200, 200),
    batch_size=1,
    class_mode='categorical',
    save_to_dir=augmented_dir,
    save_prefix='aug',
    save_format='jpeg'
)

# Generate 100 augmented images
num_augmented_images = 100
for i in range(num_augmented_images):
    train_dataset.next()

print(f"Generated {num_augmented_images} augmented images in {augmented_dir}")
