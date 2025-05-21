import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# Hardcoded input and output directories
input_dir = 'D:\\ML Bishal\\Origanal Data 1\\type_16'  # Replace with your input directory
output_dir = 'D:\\ML Bishal\Augmented Data'  # Replace with your output directory

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to add padding to an image
def add_padding(img, padding_size=50, color=(255, 255, 255)):
    width, height = img.size
    new_width = width + 2 * padding_size
    new_height = height + 2 * padding_size
    new_img = Image.new("RGB", (new_width, new_height), color)
    new_img.paste(img, (padding_size, padding_size))
    return new_img

# Create an ImageDataGenerator instance with augmentations
datagen = ImageDataGenerator(
    rotation_range=50,         # Rotation range for angle variation
    width_shift_range=0.00,    # No horizontal shift
    height_shift_range=0.00,   # No vertical shift
    shear_range=0.1,           # Small shear transformation
    zoom_range=0.1,            # Small zoom
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='constant',      # Fill empty areas with constant color
    cval=0                     # Fill with black
)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpeg', '.jpg', '.png')):  # Correct extensions
        img_path = os.path.join(input_dir, filename)

        # Load the image and add padding
        img = Image.open(img_path)
        padded_img = add_padding(img, padding_size=50)  # Adjust padding as needed

        # Convert the padded image to a numpy array
        x = tf.keras.preprocessing.image.img_to_array(padded_img)
        x = np.expand_dims(x, axis=0)  # Expand dimensions to (1, height, width, channels)

        # Extract the last portion (class name) from the input directory path
        class_name = os.path.basename(input_dir)  # Get the last folder name (class name)
        
        # Create a directory for the class in the output folder if it doesn't exist
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Generate augmented images and save them in the class-specific folder
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=class_output_dir,
                                  save_prefix='aug_' + os.path.splitext(filename)[0],
                                  save_format='jpeg'):
            i += 1
            if i >= 1:  # Limit the number of augmented images per original image to 1
                break

print(f"Augmented images saved to {output_dir}")
