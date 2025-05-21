import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# Input and output directories
input_dir = 'D:\\Eggplant Dataset\\Healthy'  # Replace with your input directory
output_dir = 'D:\\NEW MODEL\\Healthy'  # Replace with your output directory


# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to add padding to an image with a white background
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
    cval=255                   # Fill with white
)

# Process each subfolder in the input directory
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    output_subfolder_path = os.path.join(output_dir, subfolder)

    # Ensure the output subfolder exists
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)

    if os.path.isdir(subfolder_path):
        # List images in the subfolder
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        current_count = len(images)

        if current_count >= TARGET_COUNT:
            print(f"Class '{subfolder}' already has {current_count} images. Copying first {TARGET_COUNT}.")
            # Copy only the first 628 images
            for idx, filename in enumerate(images[:TARGET_COUNT]):
                img_path = os.path.join(subfolder_path, filename)
                output_path = os.path.join(output_subfolder_path, filename)
                Image.open(img_path).save(output_path)
        else:
            augment_count = TARGET_COUNT - current_count
            multiplier = int(np.ceil(augment_count / current_count))  # Ceil value for augmentation per image

            print(f"Augmenting class '{subfolder}' with {augment_count} images. Multiplier: {multiplier}")

            for filename in images:
                img_path = os.path.join(subfolder_path, filename)

                # Load the image and add padding
                img = Image.open(img_path)
                padded_img = add_padding(img, padding_size=50)  # Add padding with white background

                # Convert the padded image to a numpy array
                x = tf.keras.preprocessing.image.img_to_array(padded_img)
                x = np.expand_dims(x, axis=0)  # Expand dimensions to (1, height, width, channels)

                # Generate augmented images and save them
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=output_subfolder_path,
                                          save_prefix='aug_' + os.path.splitext(filename)[0],
                                          save_format='jpeg'):
                    i += 1
                    if i >= 8:
                        break

            # Check the total number of images after augmentation
            all_images = os.listdir(output_subfolder_path)
            if len(all_images) > TARGET_COUNT:
                # Remove extra images to ensure exactly 628
                excess_count = len(all_images) - TARGET_COUNT
                extra_files = all_images[-excess_count:]
                for extra_file in extra_files:
                    os.remove(os.path.join(output_subfolder_path, extra_file))

print(f"Augmented dataset saved to {output_dir}")
